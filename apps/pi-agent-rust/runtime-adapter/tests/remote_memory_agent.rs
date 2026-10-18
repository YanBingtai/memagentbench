use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use pi_agent_memory::{
    ContentRef, IngestReceipt, MemoryCapabilities, MemoryHit, MemoryKind, MemoryManifest,
    MemoryRegistry, Modality, RetrieveResponse,
};
use pi_agent_memory_agent::CoordinatorConfig;
use pi_agent_memory_remote::RemoteMemoryMethod;
use pi_agent_runtime_adapter::{
    AfterRunMemoryAdapter, AfterRunMemoryConfig, LatestUserTextPlanner, MemoryHookAdapter,
};
use pi_agent_rust::{
    AgentLoop, ChatFuture, ChatModel, ChatOptions, ChatRequest, MemoryHookConfig, Message,
    ToolRegistry,
};
use serde_json::{json, Value};
use uuid::Uuid;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

fn manifest() -> MemoryManifest {
    MemoryManifest {
        method_id: "remote.agent-test".to_owned(),
        version: "0.1.0".to_owned(),
        schema_version: 1,
        capabilities: MemoryCapabilities {
            modalities: vec![Modality::Text],
            memory_kinds: vec![MemoryKind::Semantic, MemoryKind::Episodic],
            supports_content_search: true,
            supports_time_filter: true,
            supports_visibility_cutoff: true,
            supports_graph: false,
            supports_cross_modal: false,
            supports_update: false,
            supports_delete: false,
            supports_batch: true,
            supports_record_ingest: true,
            supports_record_get: true,
            deterministic: true,
            persistent: true,
        },
    }
}

struct RecordingModel {
    requests: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl ChatModel for RecordingModel {
    fn complete<'a>(&'a self, request: ChatRequest<'a>) -> ChatFuture<'a> {
        self.requests
            .lock()
            .expect("request mutex should not be poisoned")
            .push(request.messages.to_vec());
        Box::pin(async { Ok(Message::assistant(Some("done".to_owned()), None)) })
    }
}

#[tokio::test]
async fn remote_memory_is_used_before_model_and_written_after_run() {
    let server = MockServer::start().await;
    let mut descriptor = serde_json::to_value(manifest()).expect("manifest should serialize");
    descriptor["config"] = json!({"profile": "agent-test"});
    Mock::given(method("GET"))
        .and(path("/manifest"))
        .respond_with(ResponseTemplate::new(200).set_body_json(descriptor))
        .expect(1)
        .mount(&server)
        .await;

    let retrieve_response = RetrieveResponse {
        query_id: Uuid::from_u128(20),
        hits: vec![MemoryHit {
            item_id: Uuid::from_u128(21),
            source_record_ids: vec![Uuid::from_u128(22)],
            version: 1,
            method_id: "remote.agent-test".to_owned(),
            rank: 1,
            score: Some(0.95),
            evidence: Vec::new(),
            evidence_refs: Vec::new(),
            projection: Some(ContentRef::Text {
                text: "之前的远程记忆".to_owned(),
            }),
            provenance: "remote.agent-test:22".to_owned(),
            snapshot_id: Some("snapshot-1".to_owned()),
            latency_ms: 1,
        }],
        snapshot_id: Some("snapshot-1".to_owned()),
        latency_ms: 1,
        truncated: false,
        telemetry: None,
    };
    Mock::given(method("POST"))
        .and(path("/retrieve"))
        .respond_with(ResponseTemplate::new(200).set_body_json(retrieve_response))
        .expect(1)
        .mount(&server)
        .await;

    let receipt = IngestReceipt {
        item_ids: vec![Uuid::from_u128(30)],
        relation_ids: Vec::new(),
        deduplicated: false,
        snapshot_id: Some("snapshot-2".to_owned()),
        telemetry: None,
    };
    Mock::given(method("POST"))
        .and(path("/ingest_records"))
        .respond_with(ResponseTemplate::new(200).set_body_json(receipt))
        .expect(1)
        .mount(&server)
        .await;

    let provider = RemoteMemoryMethod::discover(server.uri(), Duration::from_secs(5))
        .await
        .expect("remote manifest discovery should succeed");
    let mut registry = MemoryRegistry::new();
    registry
        .register(Arc::new(provider))
        .expect("remote provider should register");
    let registry = Arc::new(registry);

    let model_requests = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingModel {
            requests: Arc::clone(&model_requests),
        },
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("test agent should build");
    let agent = MemoryHookAdapter::new(
        Arc::clone(&registry),
        LatestUserTextPlanner::default(),
        CoordinatorConfig::default(),
        MemoryHookConfig {
            namespace: "bench/remote".to_owned(),
            max_tokens: 128,
            ..MemoryHookConfig::default()
        },
    )
    .attach(agent);
    let agent = AfterRunMemoryAdapter::new(
        Arc::clone(&registry),
        AfterRunMemoryConfig::new("remote.agent-test", "bench/remote"),
    )
    .attach(agent);

    let result = agent
        .run("查找之前的远程记忆")
        .await
        .expect("remote memory agent should complete");
    assert_eq!(result.final_message.content.as_deref(), Some("done"));

    {
        let model_requests = model_requests
            .lock()
            .expect("request mutex should not be poisoned");
        assert_eq!(model_requests.len(), 1);
        assert!(model_requests[0].iter().any(|message| {
            message.role == "system"
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("之前的远程记忆"))
        }));
    }

    let requests = server
        .received_requests()
        .await
        .expect("wiremock should return received requests");
    assert_eq!(requests.len(), 3);
    let mut paths = requests
        .iter()
        .map(|request| request.url.path().to_owned())
        .collect::<Vec<_>>();
    paths.sort();
    assert_eq!(paths, vec!["/ingest_records", "/manifest", "/retrieve"]);

    let retrieve_body: Value = serde_json::from_slice(
        &requests
            .iter()
            .find(|request| request.url.path() == "/retrieve")
            .expect("retrieve request should exist")
            .body,
    )
    .expect("retrieve request should be valid JSON");
    assert_eq!(
        retrieve_body["request"]["input"]["text"],
        "查找之前的远程记忆"
    );
    assert_eq!(retrieve_body["context"]["namespace"], "bench/remote");

    let ingest_body: Value = serde_json::from_slice(
        &requests
            .iter()
            .find(|request| request.url.path() == "/ingest_records")
            .expect("record ingest request should exist")
            .body,
    )
    .expect("record ingest request should be valid JSON");
    assert_eq!(
        ingest_body["request"]["records"][0]["item"]["namespace"],
        "bench/remote"
    );
    assert_eq!(
        ingest_body["request"]["idempotency_key"]
            .as_str()
            .map(|key| key.starts_with("agent-turn:")),
        Some(true)
    );
}
