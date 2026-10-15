use pi_agent_artifacts::{ArtifactStore, FsArtifactStore, PutArtifactRequest};
use pi_agent_runtime::{
    canonical_json_digest, ActionId, AdmissionPolicy, Budget, InMemoryLedger,
    PreparationEffectClass, PreparationPlan, PreparationWorkResult, ResourceScope, Revision, RunId,
    RuntimeCoordinator,
};
use pi_agent_runtime_adapter::PreparedArtifactTool;
use pi_agent_rust::{
    AgentLoop, ChatFuture, ChatModel, ChatOptions, Config, FunctionCall, Message, RunResult,
    ToolCall, ToolRegistry,
};
use serde_json::{json, Value};
use std::{collections::VecDeque, env, error::Error, fs, sync::Mutex};

struct RecordingModel {
    responses: Mutex<VecDeque<Message>>,
}

struct TemporaryStore {
    path: std::path::PathBuf,
}

impl Drop for TemporaryStore {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

impl ChatModel for RecordingModel {
    fn complete<'a>(&'a self, request: pi_agent_rust::ChatRequest<'a>) -> ChatFuture<'a> {
        let _ = request;
        let response = self
            .responses
            .lock()
            .expect("recording model response mutex should not be poisoned")
            .pop_front()
            .expect("recording model response queue should not be empty");
        Box::pin(async move { Ok(response) })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let store_root = env::temp_dir().join(format!(
        "pi-agent-prepared-handoff-{}",
        uuid::Uuid::new_v4()
    ));
    let _temporary_store = TemporaryStore {
        path: store_root.clone(),
    };
    let store = FsArtifactStore::new(&store_root)?;
    let run_id = RunId::new();
    let action_id = ActionId::new();
    let action_arguments = json!({"episode": "episode-7"});
    let plan = PreparationPlan::new(
        run_id,
        action_id,
        Revision::INITIAL,
        "memory.retrieve_video",
        canonical_json_digest(&action_arguments),
        PreparationEffectClass::ReadOnly,
        ResourceScope::new(["memory:provider:video"]),
        Budget {
            wall_time_ms: Some(1_000),
            tokens: Some(256),
            tool_calls: Some(1),
        },
        0.95,
        1_000,
        5_000,
    )?;
    let plan_id = plan.plan_id;
    let mut coordinator = RuntimeCoordinator::new(
        run_id,
        AdmissionPolicy::deny_all(),
        InMemoryLedger::default(),
    );
    coordinator.submit_preparation(plan, 1_000)?;
    let proposal = pi_agent_runtime::ActionProposal {
        run_id,
        action_id,
        base_revision: Revision::INITIAL,
        tool_name: "memory.retrieve_video".to_owned(),
        arguments: action_arguments,
        arguments_digest: canonical_json_digest(&json!({"episode": "episode-7"})),
        risk: pi_agent_runtime::RiskClass::ReadOnly,
        idempotency_key: "prepared-handoff-smoke".to_owned(),
        created_at_ms: 1_000,
        expires_at_ms: Some(5_000),
    };
    coordinator.execute_preparation(plan_id, 1_001, 1_100, |_plan| {
        let artifact = store.put(PutArtifactRequest {
            namespace: "runtime/run-1".to_owned(),
            idempotency_key: Some("episode-7-video".to_owned()),
            media_type: "video/mp4".to_owned(),
            data: b"video bytes".to_vec(),
            codec: Some("h264".to_owned()),
            schema: Some("episode-video-v1".to_owned()),
            stream_key: Some("camera/front".to_owned()),
        })?;
        let evidence = pi_agent_runtime::EvidenceLocator {
            artifact: artifact.clone(),
            entity_path: Some("robot/gripper/object-3".to_owned()),
            time_range: None,
            frame_start: Some(36),
            frame_end: Some(42),
            step_start: Some(12),
            step_end: Some(14),
            byte_start: None,
            byte_end: None,
        };
        Ok::<_, pi_agent_artifacts::ArtifactStoreError>(PreparationWorkResult::Ready {
            artifacts: vec![artifact],
            evidence: vec![evidence],
        })
    })?;
    let consumed =
        coordinator.consume_preparation_with_resolver(plan_id, &proposal, 1_200, |artifact| {
            store.stat(artifact)
        })?;
    let mut tools = ToolRegistry::new();
    tools.register(PreparedArtifactTool::from_consumed_at(&consumed, 1_200)?)?;
    let use_local_model = env::var_os("PI_AGENT_USE_LOCAL_MODEL").is_some();
    let result = if use_local_model {
        run_with_local_model(tools).await?
    } else {
        run_with_recording_model(tools).await?
    };
    let tool_content = result
        .messages
        .iter()
        .find(|message| message.role == "tool")
        .and_then(|message| message.content.as_deref())
        .ok_or("agent did not send the prepared artifact result back to the model")?;
    let returned: Value = serde_json::from_str(tool_content)?;
    if returned["artifacts"][0]["uri"] != consumed.artifacts[0].uri
        || returned["evidence"][0]["artifact"]["sha256"] != consumed.artifacts[0].sha256
        || returned["evidence"][0]["frame_start"] != 36
    {
        return Err("agent loop did not preserve the verified artifact metadata".into());
    }
    if !use_local_model
        && result.final_message.content.as_deref() != Some("prepared artifact acknowledged")
    {
        return Err("recording model returned an unexpected final message".into());
    }
    println!(
        "prepared_handoff_smoke_ok uri={} evidence={} model_tool_round_trip=true mode={}",
        consumed.artifacts[0].uri,
        consumed.artifacts[0].sha256,
        if use_local_model {
            "local"
        } else {
            "recording"
        }
    );
    drop(store);
    Ok(())
}

async fn run_with_recording_model(tools: ToolRegistry) -> Result<RunResult, Box<dyn Error>> {
    let agent = AgentLoop::new(
        RecordingModel {
            responses: Mutex::new(VecDeque::from([
                Message::assistant(
                    None,
                    Some(vec![ToolCall {
                        id: "handoff-1".to_owned(),
                        call_type: "function".to_owned(),
                        function: FunctionCall {
                            name: "prepared_artifact".to_owned(),
                            arguments: "{}".to_owned(),
                        },
                    }]),
                ),
                Message::assistant(Some("prepared artifact acknowledged".to_owned()), None),
            ])),
        },
        tools,
        "prepared-handoff-smoke-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )?;
    Ok(agent.run("use the prepared video").await?)
}

async fn run_with_local_model(tools: ToolRegistry) -> Result<RunResult, Box<dyn Error>> {
    let current_dir = env::current_dir()?;
    let config = Config {
        base_url: env::var("PI_AGENT_BASE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:6273/v1".to_owned()),
        model: env::var("PI_AGENT_MODEL").unwrap_or_else(|_| "qwen3.8".to_owned()),
        workspace: current_dir.join("workspace"),
        session_file: current_dir.join("sessions/prepared-handoff-session.jsonl"),
        system_prompt: Some(
            "Call prepared_artifact before answering, then report the artifact URI and frame range."
                .to_owned(),
        ),
        ..Config::default()
    };
    tokio::fs::create_dir_all(&config.workspace).await?;
    let agent = AgentLoop::from_config(&config, tools)?.with_runtime_ledger_file(
        config
            .session_file
            .with_file_name("prepared-handoff-runtime.jsonl"),
    );
    Ok(agent
        .run("Use the prepared video and report its evidence.")
        .await?)
}
