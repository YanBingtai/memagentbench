use std::{
    collections::BTreeMap,
    collections::VecDeque,
    error::Error,
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use pi_agent_memory::{
    ContentRef, IngestRequest, JsonlProjectionStore, MemoryItem, MemoryKind, MemoryMethod,
    MemoryRegistry, MemoryScope, Modality, OperationContext, ProjectionMemoryMethod,
};
use pi_agent_memory_agent::CoordinatorConfig;
use pi_agent_runtime::{
    CommandGateway, EventSequence, JsonlLedger, RiskClass, SandboxExecutor, SandboxPolicy,
};
use pi_agent_runtime_adapter::{
    AfterRunMemoryAdapter, AfterRunMemoryConfig, FixedCommandTool, MemoryHookAdapter,
    StructuralLatestMemoryPlanner,
};
use pi_agent_rust::{
    AgentLoop, ChatFuture, ChatModel, ChatOptions, ChatRequest, FunctionCall, MemoryHookConfig,
    Message, ToolCall, ToolContext, ToolRegistry,
};
use uuid::Uuid;

struct RecordingModel {
    responses: Mutex<VecDeque<Message>>,
    requests: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl ChatModel for RecordingModel {
    fn complete<'a>(&'a self, request: ChatRequest<'a>) -> ChatFuture<'a> {
        self.requests
            .lock()
            .expect("recording model mutex should not be poisoned")
            .push(request.messages.to_vec());
        let response = self
            .responses
            .lock()
            .expect("recording model response mutex should not be poisoned")
            .pop_front()
            .expect("offline smoke response queue should not be empty");
        Box::pin(async { Ok(response) })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let namespace = "offline-smoke";
    let memory_path = temporary_memory_path();
    let runtime_path = temporary_runtime_path();
    let workspace = temporary_workspace_path();
    fs::create_dir_all(&workspace)?;
    let store = JsonlProjectionStore::open(&memory_path)?;
    let provider = ProjectionMemoryMethod::new(Arc::new(store), "jsonl-projection", "0.1.0")?;
    seed_memory(&provider, namespace).await?;

    let mut first_registry = MemoryRegistry::new();
    first_registry.register(Arc::new(provider))?;
    let first_registry = Arc::new(first_registry);
    let first_requests = Arc::new(Mutex::new(Vec::new()));
    let agent = build_agent(
        Arc::clone(&first_registry),
        workspace.clone(),
        runtime_path.clone(),
        Arc::clone(&first_requests),
        "offline-command-1",
        false,
    )?;

    agent.run("explain the parser decision").await?;
    let (injected, tool_round_trip) = {
        let first_requests = first_requests
            .lock()
            .expect("recording model mutex should not be poisoned");
        let injected = first_requests
            .first()
            .into_iter()
            .flat_map(|messages| messages.iter())
            .filter_map(|message| message.content.as_deref())
            .any(|content| content.contains("recursive descent parser"));
        let tool_round_trip = has_tool_result(&first_requests);
        (injected, tool_round_trip)
    };
    if !injected {
        return Err("offline smoke did not observe the persisted memory injection".into());
    }
    if !tool_round_trip {
        return Err("offline smoke did not observe the fixed command tool result".into());
    }
    drop(agent);
    drop(first_registry);

    let second_store = JsonlProjectionStore::open(&memory_path)?;
    let second_provider =
        ProjectionMemoryMethod::new(Arc::new(second_store), "jsonl-projection", "0.1.0")?;
    let mut second_registry = MemoryRegistry::new();
    second_registry.register(Arc::new(second_provider))?;
    let second_requests = Arc::new(Mutex::new(Vec::new()));
    let second_agent = build_agent(
        Arc::new(second_registry),
        workspace.clone(),
        runtime_path.clone(),
        Arc::clone(&second_requests),
        "offline-command-2",
        true,
    )?;
    let second_result = second_agent.run("repeat the parser decision").await?;
    let previous_run_injected = second_requests
        .lock()
        .expect("recording model request mutex should not be poisoned")
        .first()
        .into_iter()
        .flat_map(|messages| messages.iter())
        .filter_map(|message| message.content.as_deref())
        .any(|content| content.contains("offline smoke response"));
    if !previous_run_injected {
        return Err("offline smoke did not retrieve the previous run transcript".into());
    }
    if !has_tool_result(
        &second_requests
            .lock()
            .expect("recording model mutex should not be poisoned"),
    ) {
        return Err("offline recovery did not execute the fixed command tool".into());
    }
    let runtime_events = {
        let ledger = JsonlLedger::open(&runtime_path)?;
        ledger.scan_after(EventSequence::INITIAL)?
    };
    let runtime_ledger = runtime_events.len() == 8
        && runtime_events
            .iter()
            .any(|event| event.draft.kind.as_str() == "action_admitted")
        && runtime_events
            .iter()
            .any(|event| event.draft.kind.as_str() == "action_succeeded")
        && runtime_events.iter().any(|event| {
            event.draft.kind.as_str() == "memory_injected"
                && event.draft.payload["details"]["retrieval_summary"]["hit_count"] == 1
        });
    if !runtime_ledger {
        return Err("offline smoke did not observe persisted runtime action events".into());
    }

    println!(
        "offline_smoke_ok memory_injected=true command_tool_round_trip=true runtime_ledger=true recovery=true final={}",
        second_result.final_message.content.as_deref().unwrap_or_default()
    );
    let _ = fs::remove_file(memory_path);
    let _ = fs::remove_file(runtime_path);
    let _ = fs::remove_dir_all(workspace);
    Ok(())
}

fn temporary_memory_path() -> PathBuf {
    std::env::temp_dir().join(format!("pi-agent-runtime-adapter-{}.jsonl", Uuid::new_v4()))
}

fn temporary_workspace_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "pi-agent-runtime-adapter-workspace-{}",
        Uuid::new_v4()
    ))
}

fn temporary_runtime_path() -> PathBuf {
    std::env::temp_dir().join(format!("pi-agent-runtime-ledger-{}.jsonl", Uuid::new_v4()))
}

fn build_agent(
    registry: Arc<MemoryRegistry>,
    workspace: PathBuf,
    runtime_path: PathBuf,
    requests: Arc<Mutex<Vec<Vec<Message>>>>,
    tool_call_id: &str,
    recover: bool,
) -> Result<AgentLoop<RecordingModel>, Box<dyn Error>> {
    let mut tools = ToolRegistry::with_read_file(ToolContext {
        workspace,
        max_file_bytes: 4_096,
        max_tool_output_bytes: 4_096,
    })?;
    let command_gateway = CommandGateway::new(
        SandboxExecutor::from_path("/usr/bin/bwrap")?,
        SandboxPolicy::disabled("/tmp"),
    )?;
    tools.register(FixedCommandTool::new(
        "runtime_echo",
        "Emit a deterministic runtime smoke-test value.",
        "/usr/bin/printf",
        command_gateway,
        1,
        RiskClass::ReadOnly,
    )?)?;
    let mut agent = AgentLoop::new(
        RecordingModel {
            responses: Mutex::new(VecDeque::from([
                Message::assistant(
                    None,
                    Some(vec![ToolCall {
                        id: tool_call_id.to_owned(),
                        call_type: "function".to_owned(),
                        function: FunctionCall {
                            name: "runtime_echo".to_owned(),
                            arguments: r#"{"args":["tool fixture content"]}"#.to_owned(),
                        },
                    }]),
                ),
                Message::assistant(Some("offline smoke response".to_owned()), None),
            ])),
            requests,
        },
        tools,
        "offline-smoke-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )?
    .with_runtime_ledger_file(runtime_path);
    if recover {
        agent = agent.with_runtime_recovery();
    }
    let agent = MemoryHookAdapter::new(
        Arc::clone(&registry),
        StructuralLatestMemoryPlanner::default(),
        CoordinatorConfig::default(),
        MemoryHookConfig {
            namespace: "offline-smoke".to_owned(),
            max_tokens: 4_096,
            ..MemoryHookConfig::default()
        },
    )
    .attach(agent);
    Ok(AfterRunMemoryAdapter::new(
        registry,
        AfterRunMemoryConfig::new("jsonl-projection", "offline-smoke"),
    )
    .attach(agent))
}

fn has_tool_result(requests: &[Vec<Message>]) -> bool {
    requests.len() == 2
        && requests[1].iter().any(|message| {
            message.role == "tool"
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("tool fixture content"))
        })
}

async fn seed_memory(
    provider: &ProjectionMemoryMethod,
    namespace: &str,
) -> Result<(), Box<dyn Error>> {
    let idempotency_key = "offline-smoke-seed".to_owned();
    provider
        .ingest(
            IngestRequest {
                items: vec![MemoryItem {
                    id: Uuid::new_v4(),
                    version: 1,
                    kind: MemoryKind::Semantic,
                    scope: MemoryScope::Shared,
                    modality: Modality::Text,
                    content: ContentRef::Text {
                        text: "recursive descent parser".to_owned(),
                    },
                    namespace: namespace.to_owned(),
                    episode_id: None,
                    subject: None,
                    source: "offline-smoke".to_owned(),
                    metadata: BTreeMap::new(),
                    abstraction_level: 0,
                    event_time: None,
                    ingest_time_ms: 1_000,
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
