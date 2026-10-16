use std::time::Duration;

use pi_agent_memory::{
    ContentRef, IngestReceipt, MemoryBudget, MemoryCapabilities, MemoryItem, MemoryKind,
    MemoryManifest, MemoryMethod, MemoryProvenance, MemoryQuery, MemoryRecord, MemoryScope,
    Modality, OperationContext, RetrieveResponse,
};
use pi_agent_memory_remote::RemoteMemoryMethod;
use serde_json::{json, Value};
use uuid::Uuid;
use wiremock::{matchers::method, matchers::path, Mock, MockServer, ResponseTemplate};

fn manifest() -> MemoryManifest {
    MemoryManifest {
        method_id: "remote.http-test".to_owned(),
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

fn context() -> OperationContext {
    OperationContext {
        namespace: "bench/http".to_owned(),
        run_id: Uuid::from_u128(1),
        session_id: Some(Uuid::from_u128(2)),
        episode_id: Some(Uuid::from_u128(3)),
        timeout_ms: Some(1_000),
        idempotency_key: Some("operation-1".to_owned()),
    }
}

fn query() -> MemoryQuery {
    MemoryQuery {
        input: Some(ContentRef::Text {
            text: "红色方块放在哪里".to_owned(),
        }),
        namespace: "bench/http".to_owned(),
        scope: None,
        subject: None,
        episode_id: None,
        kinds: vec![MemoryKind::Semantic],
        modalities: vec![Modality::Text],
        time_range: None,
        visibility_cutoff_ms: Some(10_000),
        graph: None,
        budget: MemoryBudget {
            max_hits: 2,
            ..MemoryBudget::default()
        },
    }
}

fn record() -> MemoryRecord {
    MemoryRecord {
        item: MemoryItem {
            id: Uuid::from_u128(10),
            version: 1,
            kind: MemoryKind::Episodic,
            scope: MemoryScope::Personalized,
            modality: Modality::Text,
            content: ContentRef::Text {
                text: "红色方块在左侧".to_owned(),
            },
            namespace: "bench/http".to_owned(),
            episode_id: Some("episode-1".to_owned()),
            subject: None,
            source: "test".to_owned(),
            metadata: Default::default(),
            abstraction_level: 0,
            event_time: None,
            ingest_time_ms: 10_000,
            schema_version: 1,
        },
        episode: None,
        evidence: Vec::new(),
        provenance: MemoryProvenance {
            producer: "http-test".to_owned(),
            source_id: "episode-1".to_owned(),
            source_event_id: Some("event-1".to_owned()),
            parent_record_ids: Vec::new(),
            method_version: Some("0.1.0".to_owned()),
        },
    }
}

#[tokio::test]
async fn remote_method_round_trips_discovery_retrieval_and_record_ingest() {
    let server = MockServer::start().await;
    let mut descriptor = serde_json::to_value(manifest()).expect("manifest should serialize");
    descriptor["config"] = json!({
        "embedding_model": "test-embedding",
        "profile": "benchmark"
    });
    Mock::given(method("GET"))
        .and(path("/manifest"))
        .respond_with(ResponseTemplate::new(200).set_body_json(descriptor))
        .expect(1)
        .mount(&server)
        .await;

    let item_id = Uuid::from_u128(11);
    let retrieve_response = RetrieveResponse {
        query_id: Uuid::from_u128(12),
        hits: vec![pi_agent_memory::MemoryHit {
            item_id,
            source_record_ids: vec![Uuid::from_u128(10)],
            version: 1,
            method_id: "remote.http-test".to_owned(),
            rank: 1,
            score: Some(0.9),
            evidence: Vec::new(),
            evidence_refs: Vec::new(),
            projection: Some(ContentRef::Text {
                text: "红色方块在左侧".to_owned(),
            }),
            provenance: "http-test:10".to_owned(),
            snapshot_id: Some("snapshot-1".to_owned()),
            latency_ms: 3,
        }],
        snapshot_id: Some("snapshot-1".to_owned()),
        latency_ms: 3,
        truncated: false,
        telemetry: None,
    };
    Mock::given(method("POST"))
        .and(path("/retrieve"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&retrieve_response))
        .expect(1)
        .mount(&server)
        .await;

    let receipt = IngestReceipt {
        item_ids: vec![Uuid::from_u128(10)],
        relation_ids: Vec::new(),
        deduplicated: false,
        snapshot_id: Some("snapshot-2".to_owned()),
        telemetry: None,
    };
    Mock::given(method("POST"))
        .and(path("/ingest_records"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&receipt))
        .expect(1)
        .mount(&server)
        .await;

    let provider = RemoteMemoryMethod::discover(server.uri(), Duration::from_secs(5))
        .await
        .expect("manifest discovery should succeed");
    assert_eq!(provider.provider_config()["profile"], "benchmark");
    assert_eq!(provider.manifest().method_id, "remote.http-test");

    let retrieved = provider
        .retrieve(query(), context())
        .await
        .expect("retrieve should use the discovered provider");
    assert_eq!(retrieved.hits[0].item_id, item_id);
    assert_eq!(
        retrieved.hits[0].source_record_ids,
        vec![Uuid::from_u128(10)]
    );

    let ingested = provider
        .ingest_records(
            pi_agent_memory::RecordIngestRequest {
                records: vec![record()],
                relations: Vec::new(),
                idempotency_key: "record-1".to_owned(),
            },
            context(),
        )
        .await
        .expect("record ingest should use the advertised capability");
    assert_eq!(ingested, receipt);

    let requests = server
        .received_requests()
        .await
        .expect("wiremock should return received requests");
    assert_eq!(requests.len(), 3);

    let retrieve_body: Value =
        serde_json::from_slice(&requests[1].body).expect("retrieve request should be valid JSON");
    assert_eq!(
        retrieve_body["request"]["input"]["text"],
        "红色方块放在哪里"
    );
    assert_eq!(retrieve_body["context"]["namespace"], "bench/http");
    assert_eq!(retrieve_body["context"]["idempotency_key"], "operation-1");

    let ingest_body: Value = serde_json::from_slice(&requests[2].body)
        .expect("record ingest request should be valid JSON");
    assert_eq!(
        ingest_body["request"]["records"][0]["item"]["id"],
        Uuid::from_u128(10).to_string()
    );
    assert_eq!(
        ingest_body["request"]["records"][0]["provenance"]["source_event_id"],
        "event-1"
    );
    assert_eq!(ingest_body["request"]["idempotency_key"], "record-1");
}
