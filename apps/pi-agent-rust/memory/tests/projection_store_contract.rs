//! Public contract tests for projection-store implementations.
//!
//! The helper is intentionally written against `ProjectionStore`, rather than
//! `JsonlProjectionStore` methods. A future SQLite, remote, or test backend can
//! call the same helper after it is constructed.

use std::{
    collections::BTreeMap,
    env,
    future::Future,
    path::PathBuf,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use pi_agent_memory::{
    ContentRef, EpisodeRef, JsonlProjectionStore, MemoryItem, MemoryKind, MemoryProvenance,
    MemoryRecord, MemoryRelation, MemoryScope, Modality, ProjectionStore, RelationQuery,
    RelationType, StorageContext, StorageError, StoreGetRequest, StoreOrder, StoreQuery,
    StoreWriteRequest, TraversalDirection,
};
use uuid::Uuid;

struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn path(name: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "pi-agent-memory-contract-{name}-{}.jsonl",
        Uuid::new_v4()
    ))
}

fn context(namespace: &str) -> StorageContext {
    StorageContext {
        namespace: namespace.to_string(),
        run_id: Uuid::new_v4(),
        snapshot_id: None,
        timeout_ms: None,
    }
}

fn record(id: u128, namespace: &str, episode_id: &str, ingest_time_ms: u64) -> MemoryRecord {
    MemoryRecord {
        item: MemoryItem {
            id: Uuid::from_u128(id),
            version: 1,
            kind: MemoryKind::Episodic,
            scope: MemoryScope::Personalized,
            modality: Modality::Text,
            content: ContentRef::Text {
                text: format!("memory-{id}"),
            },
            namespace: namespace.to_string(),
            episode_id: None,
            subject: None,
            source: "contract-test".to_string(),
            metadata: BTreeMap::new(),
            abstraction_level: 0,
            event_time: None,
            ingest_time_ms,
            schema_version: 1,
        },
        episode: Some(EpisodeRef {
            dataset_or_run_id: "dataset-1".to_string(),
            episode_id: episode_id.to_string(),
            task: "contract test".to_string(),
            time_range: None,
            dataset_revision: "v1".to_string(),
            source: "contract-test".to_string(),
        }),
        evidence: Vec::new(),
        provenance: MemoryProvenance {
            producer: "contract-test".to_string(),
            source_id: "source-1".to_string(),
            source_event_id: None,
            parent_record_ids: Vec::new(),
            method_version: Some("contract-1".to_string()),
        },
    }
}

fn relation(id: u128, ingest_time_ms: u64) -> MemoryRelation {
    MemoryRelation {
        id: Uuid::from_u128(id),
        from: Uuid::from_u128(1),
        to: Uuid::from_u128(2),
        relation_type: RelationType::DerivedFrom,
        weight: None,
        confidence: Some(1.0),
        valid_time: None,
        ingest_time_ms,
        source: "contract-test".to_string(),
        metadata: BTreeMap::new(),
        schema_version: 1,
    }
}

fn exercise_contract<S: ProjectionStore>(store: &S) {
    let namespace = "contract/run-1";
    let write = StoreWriteRequest {
        records: vec![
            record(1, namespace, "episode-1", 5),
            record(2, namespace, "episode-1", 5),
        ],
        relations: if store.manifest().capabilities.supports_relations {
            vec![relation(3, 5)]
        } else {
            Vec::new()
        },
        idempotency_key: "batch-1".to_string(),
        source_revision: Some(1),
    };
    let first = block_on(store.write(write.clone(), context(namespace))).expect("first write");
    assert!(!first.deduplicated);
    let retry = block_on(store.write(write, context(namespace))).expect("idempotent retry");
    assert!(retry.deduplicated);

    let response = block_on(store.query(
        StoreQuery {
            namespace: namespace.to_string(),
            ids: Vec::new(),
            scope: None,
            subject: None,
            episode_id: Some("episode-1".to_string()),
            kinds: Vec::new(),
            modalities: Vec::new(),
            event_time: None,
            visibility_cutoff_ms: Some(10),
            order: StoreOrder::Id,
            limit: 8,
            cursor: None,
            snapshot_id: None,
        },
        context(namespace),
    ))
    .expect("structured query");
    assert_eq!(response.records.len(), 2);
    assert_eq!(response.records[0].item.id, Uuid::from_u128(1));

    let loaded = block_on(store.get(
        StoreGetRequest {
            record_id: Uuid::from_u128(1),
            version: None,
        },
        context(namespace),
    ))
    .expect("get by id");
    assert_eq!(loaded.episode.as_ref().unwrap().episode_id, "episode-1");

    if store.manifest().capabilities.supports_relations {
        let graph = block_on(store.relations(
            RelationQuery {
                start_ids: vec![Uuid::from_u128(1)],
                relation_types: vec![RelationType::DerivedFrom],
                direction: TraversalDirection::Outgoing,
                limit: 8,
                cursor: None,
                visibility_cutoff_ms: Some(10),
                snapshot_id: None,
            },
            context(namespace),
        ))
        .expect("relation query");
        assert_eq!(graph.relations.len(), 1);
    }
}

fn exercise_invalid_requests<S: ProjectionStore>(store: &S) {
    let error = block_on(store.get(
        StoreGetRequest {
            record_id: Uuid::new_v4(),
            version: None,
        },
        StorageContext {
            namespace: String::new(),
            run_id: Uuid::new_v4(),
            snapshot_id: None,
            timeout_ms: None,
        },
    ))
    .expect_err("empty namespaces must be rejected");
    assert!(matches!(error, StorageError::InvalidRequest(_)));

    let error = block_on(store.get(
        StoreGetRequest {
            record_id: Uuid::new_v4(),
            version: None,
        },
        StorageContext {
            namespace: "contract/run-1".to_string(),
            run_id: Uuid::new_v4(),
            snapshot_id: None,
            timeout_ms: Some(0),
        },
    ))
    .expect_err("zero timeout must be rejected");
    assert!(matches!(error, StorageError::Timeout));
}

#[test]
fn jsonl_projection_store_satisfies_public_contract() {
    let file_path = path("jsonl");
    let store = JsonlProjectionStore::open(&file_path).expect("store should open");
    exercise_contract(&store);
    exercise_invalid_requests(&store);
    drop(store);

    let reopened = JsonlProjectionStore::open(&file_path).expect("store should reopen");
    let response = block_on(reopened.query(
        StoreQuery {
            namespace: "contract/run-1".to_string(),
            ids: Vec::new(),
            scope: None,
            subject: None,
            episode_id: Some("episode-1".to_string()),
            kinds: Vec::new(),
            modalities: Vec::new(),
            event_time: None,
            visibility_cutoff_ms: Some(10),
            order: StoreOrder::Id,
            limit: 8,
            cursor: None,
            snapshot_id: None,
        },
        context("contract/run-1"),
    ))
    .expect("query after restart");
    assert_eq!(response.records.len(), 2);
    drop(reopened);
    let _ = std::fs::remove_file(file_path);
}
