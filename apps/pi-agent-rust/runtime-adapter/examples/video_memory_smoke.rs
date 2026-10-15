use std::{
    collections::BTreeMap,
    error::Error,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use pi_agent_artifacts::{
    ArtifactStore, ArtifactValidatedMemoryMethod, FsArtifactStore, PutArtifactRequest,
};
use pi_agent_memory::{
    ContentRef, EvidenceLocator, JsonlProjectionStore, MemoryItem, MemoryKind, MemoryMethod,
    MemoryProvenance, MemoryRecord, MemoryRegistry, MemoryScope, Modality, OperationContext,
    ProjectionMemoryMethod, RecordIngestRequest,
};
use pi_agent_runtime_adapter::{MemoryHookAdapter, ModalityMemoryPlanner};
use pi_agent_rust::{
    AgentLoop, ChatFuture, ChatModel, ChatOptions, ChatRequest, Message, ToolRegistry,
};
use uuid::Uuid;

struct RecordingModel {
    requests: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl ChatModel for RecordingModel {
    fn complete<'a>(&'a self, request: ChatRequest<'a>) -> ChatFuture<'a> {
        self.requests
            .lock()
            .expect("recording model mutex should not be poisoned")
            .push(request.messages.to_vec());
        Box::pin(async {
            Ok(Message::assistant(
                Some("video evidence acknowledged".to_owned()),
                None,
            ))
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let artifact_store = Arc::new(FsArtifactStore::new(directory.path().join("artifacts"))?);
    let artifact = artifact_store.put(PutArtifactRequest {
        namespace: "video-smoke".to_owned(),
        idempotency_key: Some("episode-1-front-camera".to_owned()),
        media_type: "video/mp4".to_owned(),
        data: b"video fixture bytes".to_vec(),
        codec: Some("h264".to_owned()),
        schema: Some("episode-video-v1".to_owned()),
        stream_key: Some("camera/front".to_owned()),
    })?;

    let memory_store = JsonlProjectionStore::open(directory.path().join("memory.jsonl"))?;
    let provider = Arc::new(ProjectionMemoryMethod::new(
        Arc::new(memory_store),
        "jsonl-video",
        "0.1.0",
    )?);
    let provider = Arc::new(ArtifactValidatedMemoryMethod::new(
        provider,
        Arc::clone(&artifact_store),
    ));
    let run_id = Uuid::new_v4();
    let now_ms = unix_time_ms()?;
    provider
        .ingest_records(
            RecordIngestRequest {
                records: vec![MemoryRecord {
                    item: MemoryItem {
                        id: Uuid::new_v4(),
                        version: 1,
                        kind: MemoryKind::Episodic,
                        scope: MemoryScope::Shared,
                        modality: Modality::Video,
                        content: ContentRef::Blob {
                            uri: artifact.uri.clone(),
                            media_type: artifact.media_type.clone(),
                            sha256: artifact.sha256.clone(),
                            size_bytes: artifact.bytes,
                        },
                        namespace: "video-smoke".to_owned(),
                        episode_id: Some("episode-1".to_owned()),
                        subject: None,
                        source: "video-memory-smoke".to_owned(),
                        metadata: BTreeMap::new(),
                        abstraction_level: 0,
                        event_time: None,
                        ingest_time_ms: now_ms,
                        schema_version: 1,
                    },
                    episode: None,
                    evidence: vec![EvidenceLocator {
                        artifact: artifact.clone(),
                        entity_path: Some("robot/gripper/object-3".to_owned()),
                        time_range: None,
                        frame_start: Some(120),
                        frame_end: Some(150),
                        step_start: Some(30),
                        step_end: Some(34),
                        byte_start: None,
                        byte_end: None,
                    }],
                    provenance: MemoryProvenance {
                        producer: "video-memory-smoke".to_owned(),
                        source_id: "episode-1".to_owned(),
                        source_event_id: None,
                        parent_record_ids: Vec::new(),
                        method_version: Some("0.1.0".to_owned()),
                    },
                }],
                relations: Vec::new(),
                idempotency_key: "video-smoke-record-1".to_owned(),
            },
            OperationContext {
                namespace: "video-smoke".to_owned(),
                run_id,
                session_id: None,
                episode_id: None,
                timeout_ms: None,
                idempotency_key: Some("video-smoke-record-1".to_owned()),
            },
        )
        .await?;

    let mut registry = MemoryRegistry::new();
    registry.register(provider)?;
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingModel {
            requests: Arc::clone(&requests),
        },
        ToolRegistry::new(),
        "video-memory-smoke-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )?;
    let agent = MemoryHookAdapter::new(
        Arc::new(registry),
        ModalityMemoryPlanner::new(Modality::Video),
        pi_agent_memory_agent::CoordinatorConfig::default(),
        pi_agent_rust::MemoryHookConfig {
            namespace: "video-smoke".to_owned(),
            ..pi_agent_rust::MemoryHookConfig::default()
        },
    )
    .attach(agent);
    agent.run("find the object grasping interval").await?;

    let requests = requests
        .lock()
        .expect("recording model mutex should not be poisoned");
    let system_content = requests[0]
        .iter()
        .find(|message| message.role == "system")
        .and_then(|message| message.content.as_deref())
        .ok_or("memory injection was not sent to the model")?;
    if !system_content.contains(&artifact.uri)
        || !system_content.contains("\"frame_start\":120")
        || !system_content.contains("\"frame_end\":150")
    {
        return Err("video artifact metadata was not preserved in the model context".into());
    }

    println!(
        "video_memory_smoke_ok modality=video uri={} frame_start=120 frame_end=150",
        artifact.uri
    );
    Ok(())
}

fn unix_time_ms() -> Result<u64, Box<dyn Error>> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
