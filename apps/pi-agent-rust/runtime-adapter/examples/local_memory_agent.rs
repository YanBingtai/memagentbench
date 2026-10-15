use std::{
    collections::BTreeMap,
    env,
    error::Error,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use pi_agent_memory::{
    ContentRef, IngestRequest, JsonlProjectionStore, MemoryItem, MemoryKind, MemoryMethod,
    MemoryRegistry, MemoryScope, Modality, OperationContext, ProjectionMemoryMethod,
};
use pi_agent_memory_agent::CoordinatorConfig;
use pi_agent_runtime::{
    bytes_digest, CommandGateway, MemoryRevision, PolicyRevision, RiskClass, SandboxExecutor,
    SandboxPolicy,
};
use pi_agent_runtime_adapter::{
    AfterRunMemoryAdapter, AfterRunMemoryConfig, FixedCommandTool, LatestUserTextPlanner,
    MemoryHookAdapter, ModalityMemoryPlanner, StructuralLatestMemoryPlanner,
};
use pi_agent_rust::{AgentLoop, Config, MemoryHookConfig, ToolContext, ToolRegistry};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let (prompt, seed_memory) = parse_args()?;
    if prompt.trim().is_empty() {
        eprintln!(
            "usage: cargo run --manifest-path runtime-adapter/Cargo.toml --example local_memory_agent -- [--seed-memory <text>] <prompt>\n\
             env: PI_AGENT_BASE_URL (default http://127.0.0.1:6273/v1),\n\
                  PI_AGENT_MODEL (default qwen3.8),\n\
                  PI_AGENT_MEMORY_FILE (default ./sessions/memory-projection.jsonl),\n\
                  PI_AGENT_MEMORY_NAMESPACE (default default),\n\
                  PI_AGENT_MEMORY_MODE (structural, semantic, image, audio, video, state, action, or trajectory),\n\
                  PI_AGENT_SANDBOX_MODE (default disabled; managed probes bubblewrap)"
        );
        return Ok(());
    }

    let current_dir = env::current_dir()?;
    let config = Config {
        base_url: env::var("PI_AGENT_BASE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:6273/v1".to_owned()),
        model: env::var("PI_AGENT_MODEL").unwrap_or_else(|_| "qwen3.8".to_owned()),
        workspace: current_dir.join("workspace"),
        session_file: current_dir.join("sessions/runtime-adapter-session.jsonl"),
        ..Config::default()
    };
    tokio::fs::create_dir_all(&config.workspace).await?;

    let mut tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))?;
    let command_gateway = build_gateway(
        &config.workspace,
        config.max_tool_output_bytes,
        config.tool_timeout_secs,
    )?;
    tools.register(FixedCommandTool::new(
        "runtime_echo",
        "Emit text through the runtime command boundary when a command demonstration is requested.",
        "/usr/bin/printf",
        command_gateway,
        16,
        RiskClass::ReadOnly,
    )?)?;
    let agent = AgentLoop::from_config(&config, tools)?
        .with_runtime_ledger_file(config.session_file.with_extension("runtime.jsonl"))
        .with_runtime_recovery();

    let memory_path = env::var_os("PI_AGENT_MEMORY_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./sessions/memory-projection.jsonl"));
    let namespace = env::var("PI_AGENT_MEMORY_NAMESPACE").unwrap_or_else(|_| "default".to_owned());
    let store = JsonlProjectionStore::open(&memory_path)?;
    let provider = ProjectionMemoryMethod::new(Arc::new(store), "jsonl-projection", "0.1.0")?;
    if let Some(seed) = seed_memory {
        seed_projection_memory(&provider, &namespace, &seed).await?;
    }
    let mut registry = MemoryRegistry::new();
    registry.register(Arc::new(provider))?;

    let registry = Arc::new(registry);
    let ingest_registry = Arc::clone(&registry);
    let ingest_namespace = namespace.clone();
    let hook_config = MemoryHookConfig {
        namespace,
        memory_revision: MemoryRevision::INITIAL,
        policy_revision: PolicyRevision::INITIAL,
        ..MemoryHookConfig::default()
    };
    let memory_mode = env::var("PI_AGENT_MEMORY_MODE").unwrap_or_else(|_| "structural".to_owned());
    let agent = match memory_mode.as_str() {
        "structural" => MemoryHookAdapter::new(
            Arc::clone(&registry),
            StructuralLatestMemoryPlanner::default(),
            CoordinatorConfig::default(),
            hook_config,
        )
        .attach(agent),
        "semantic" => MemoryHookAdapter::new(
            Arc::clone(&registry),
            LatestUserTextPlanner::default(),
            CoordinatorConfig::default(),
            hook_config,
        )
        .attach(agent),
        mode => match parse_modality(mode) {
            Some(modality) => MemoryHookAdapter::new(
                Arc::clone(&registry),
                ModalityMemoryPlanner::new(modality),
                CoordinatorConfig::default(),
                hook_config,
            )
            .attach(agent),
            None => {
                return Err(format!(
                    "PI_AGENT_MEMORY_MODE must be structural, semantic, image, audio, video, state, action, or trajectory, got {mode:?}"
                )
                .into())
            }
        },
    };
    let agent = AfterRunMemoryAdapter::new(
        ingest_registry,
        AfterRunMemoryConfig::new("jsonl-projection", ingest_namespace),
    )
    .attach(agent);

    let result = agent.run(prompt).await?;
    if let Some(content) = result.final_message.content {
        println!("{content}");
    } else {
        println!("{}", serde_json::to_string_pretty(&result.final_message)?);
    }
    Ok(())
}

fn build_gateway(
    workspace: &std::path::Path,
    max_output_bytes: usize,
    timeout_secs: u64,
) -> Result<CommandGateway, Box<dyn Error>> {
    let build = |executor, policy| {
        CommandGateway::new(executor, policy)
            .and_then(|gateway| gateway.with_timeout(Duration::from_secs(timeout_secs)))
            .and_then(|gateway| gateway.with_max_output_bytes(max_output_bytes))
    };
    match env::var("PI_AGENT_SANDBOX_MODE")
        .unwrap_or_else(|_| "disabled".to_owned())
        .as_str()
    {
        "disabled" => Ok(build(
            SandboxExecutor::from_path("/usr/bin/bwrap")?,
            SandboxPolicy::disabled(workspace),
        )?),
        "managed" => Ok(build(
            SandboxExecutor::discover_and_probe()?,
            SandboxPolicy::managed(workspace),
        )?),
        other => {
            Err(format!("PI_AGENT_SANDBOX_MODE must be disabled or managed, got {other:?}").into())
        }
    }
}

fn parse_modality(value: &str) -> Option<Modality> {
    match value {
        "image" => Some(Modality::Image),
        "audio" => Some(Modality::Audio),
        "video" => Some(Modality::Video),
        "state" => Some(Modality::State),
        "action" => Some(Modality::Action),
        "trajectory" => Some(Modality::Trajectory),
        _ => None,
    }
}

fn parse_args() -> Result<(String, Option<String>), Box<dyn Error>> {
    let mut prompt_parts = Vec::new();
    let mut seed_memory = None;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => return Ok((String::new(), None)),
            "--seed-memory" => {
                let value = args
                    .next()
                    .ok_or("--seed-memory requires a non-empty text value")?;
                if value.trim().is_empty() {
                    return Err("--seed-memory requires a non-empty text value".into());
                }
                seed_memory = Some(value);
            }
            _ => prompt_parts.push(argument),
        }
    }
    Ok((prompt_parts.join(" "), seed_memory))
}

async fn seed_projection_memory(
    provider: &ProjectionMemoryMethod,
    namespace: &str,
    text: &str,
) -> Result<(), Box<dyn Error>> {
    let item_id = Uuid::new_v4();
    let idempotency_key = format!(
        "seed:{}",
        bytes_digest(format!("{namespace}:{text}").as_bytes())
    );
    let ingest_time_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock is before unix epoch: {error}"))?
        .as_millis()
        .min(u64::MAX as u128) as u64;
    provider
        .ingest(
            IngestRequest {
                items: vec![MemoryItem {
                    id: item_id,
                    version: 1,
                    kind: MemoryKind::Semantic,
                    scope: MemoryScope::Shared,
                    modality: Modality::Text,
                    content: ContentRef::Text {
                        text: text.to_owned(),
                    },
                    namespace: namespace.to_owned(),
                    episode_id: None,
                    subject: None,
                    source: "local-memory-agent.seed".to_owned(),
                    metadata: BTreeMap::new(),
                    abstraction_level: 0,
                    event_time: None,
                    ingest_time_ms,
                    schema_version: 1,
                }],
                relations: Vec::new(),
                idempotency_key: idempotency_key.clone(),
            },
            OperationContext {
                namespace: namespace.to_owned(),
                run_id: Uuid::new_v4(),
                session_id: None,
                episode_id: None,
                timeout_ms: None,
                idempotency_key: Some(idempotency_key),
            },
        )
        .await?;
    Ok(())
}
