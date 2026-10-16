//! Run the Pi agent with a memory method hosted by another process.
//!
//! The provider manifest is discovered from the service by default. An
//! explicit JSON file can pin the manifest for an offline or reproducible run.

use std::{env, error::Error, fs, path::PathBuf, sync::Arc, time::Duration};

use pi_agent_memory::{MemoryMethod, MemoryRegistry, Modality};
use pi_agent_memory_agent::CoordinatorConfig;
use pi_agent_memory_remote::{RemoteMemoryDescriptor, RemoteMemoryMethod};
use pi_agent_runtime_adapter::{
    AfterRunMemoryAdapter, AfterRunMemoryConfig, CrossModalMemoryPlanner, LatestUserTextPlanner,
    MemoryHookAdapter, ModalityMemoryPlanner, StructuralLatestMemoryPlanner,
};
use pi_agent_rust::{AgentLoop, Config, MemoryHookConfig, ToolContext, ToolRegistry};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    run().await
}

pub async fn run() -> Result<(), Box<dyn Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let help_requested = arguments.len() == 1
        && matches!(arguments.first().map(String::as_str), Some("--help" | "-h"));
    let prompt = arguments.join(" ");
    if prompt.trim().is_empty() || help_requested {
        eprintln!(
            "usage: cargo run --manifest-path runtime-adapter/Cargo.toml \
             --bin pi-agent-remote-memory -- <prompt>\n\
             (or use --example remote_memory_agent)\n\
             env: PI_AGENT_MEMORY_URL (required),\n\
                  PI_AGENT_MEMORY_MANIFEST (optional pinned manifest JSON),\n\
                  PI_AGENT_MEMORY_NAMESPACE (default default),\n\
                  PI_AGENT_MEMORY_MODE (structural, semantic, image, audio, video, state, action, trajectory, or cross-modal:<modalities>;\n\
                    default follows provider content-search capability),\n\
                  PI_AGENT_MEMORY_TIMEOUT_SECS (default 30),\n\
                  PI_AGENT_BASE_URL (default http://127.0.0.1:6273/v1),\n\
                  PI_AGENT_MODEL (default qwen3.8),\n\
                  PI_AGENT_API_KEY_ENV (default OPENAI_API_KEY)"
        );
        return Ok(());
    }

    let base_url =
        env::var("PI_AGENT_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:6273/v1".to_owned());
    let model = env::var("PI_AGENT_MODEL").unwrap_or_else(|_| "qwen3.8".to_owned());
    let api_key_env = env::var("PI_AGENT_API_KEY_ENV")
        .ok()
        .or_else(|| Config::default().api_key_env.clone());
    let namespace = env::var("PI_AGENT_MEMORY_NAMESPACE").unwrap_or_else(|_| "default".to_owned());
    let memory_url = env::var("PI_AGENT_MEMORY_URL")
        .map_err(|_| "PI_AGENT_MEMORY_URL is required for remote_memory_agent")?;
    let timeout_secs = env::var("PI_AGENT_MEMORY_TIMEOUT_SECS")
        .unwrap_or_else(|_| "30".to_owned())
        .parse::<u64>()?;
    let timeout = Duration::from_secs(timeout_secs);
    let provider = match env::var_os("PI_AGENT_MEMORY_MANIFEST") {
        Some(path) => {
            let descriptor: RemoteMemoryDescriptor =
                serde_json::from_slice(&fs::read(PathBuf::from(path))?)?;
            RemoteMemoryMethod::new_with_config(
                memory_url,
                descriptor.manifest,
                descriptor.config,
                timeout,
            )?
        }
        None => RemoteMemoryMethod::discover(memory_url, timeout).await?,
    };
    let manifest = provider.manifest();

    let config = Config {
        base_url,
        model,
        api_key_env,
        workspace: env::current_dir()?.join("workspace"),
        session_file: env::current_dir()?.join("sessions/remote-memory-agent.jsonl"),
        ..Config::default()
    };
    tokio::fs::create_dir_all(&config.workspace).await?;
    let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))?;
    let agent = AgentLoop::from_config(&config, tools)?
        .with_runtime_ledger_file(config.session_file.with_extension("runtime.jsonl"));

    let mut registry = MemoryRegistry::new();
    registry.register(Arc::new(provider))?;
    let registry = Arc::new(registry);
    let hook_config = MemoryHookConfig {
        namespace: namespace.clone(),
        ..MemoryHookConfig::default()
    };
    let default_mode = if manifest.capabilities.supports_content_search {
        "semantic"
    } else {
        "structural"
    };
    let mode = env::var("PI_AGENT_MEMORY_MODE").unwrap_or_else(|_| default_mode.to_owned());
    let agent = match mode.as_str() {
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
        mode if mode.starts_with("cross-modal:") => {
            let modalities = parse_modalities(&mode["cross-modal:".len()..])?;
            MemoryHookAdapter::new(
                Arc::clone(&registry),
                CrossModalMemoryPlanner::new(modalities)?,
                CoordinatorConfig::default(),
                hook_config,
            )
            .attach(agent)
        }
        value => {
            let modality = parse_modality(value)
                .ok_or_else(|| format!("unsupported PI_AGENT_MEMORY_MODE: {value}"))?;
            MemoryHookAdapter::new(
                Arc::clone(&registry),
                ModalityMemoryPlanner::new(modality),
                CoordinatorConfig::default(),
                hook_config,
            )
            .attach(agent)
        }
    };
    let agent = AfterRunMemoryAdapter::new(
        registry,
        AfterRunMemoryConfig::new(&manifest.method_id, namespace),
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

fn parse_modalities(raw: &str) -> Result<Vec<Modality>, Box<dyn Error>> {
    let mut modalities = Vec::new();
    for value in raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let modality = parse_modality(value)
            .ok_or_else(|| format!("unsupported cross-modal modality: {value}"))?;
        if !modalities.contains(&modality) {
            modalities.push(modality);
        }
    }
    if modalities.is_empty() {
        return Err("cross-modal mode requires at least one modality".into());
    }
    Ok(modalities)
}
