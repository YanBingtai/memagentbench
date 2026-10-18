//! Public conformance checks for the first projection-backed memory method.
//!
//! These tests exercise the provider boundary rather than JSONL internals:
//! retries, namespace isolation, visibility cutoffs, hit ordering, budgets,
//! and explicitly unsupported query capabilities.

use std::{
    collections::BTreeMap,
    future::Future,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use pi_agent_memory::{
    ContentRef, IngestRequest, JsonlProjectionStore, MemoryBudget, MemoryItem, MemoryKind,
    MemoryMethod, MemoryQuery, MemoryRelation, MemoryScope, Modality, OperationContext,
    ProjectionMemoryMethod, RelationType,
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

fn operation(namespace: &str, idempotency_key: Option<&str>) -> OperationContext {
    OperationContext {
        namespace: namespace.to_owned(),
        run_id: Uuid::new_v4(),
        session_id: None,
        episode_id: None,
        timeout_ms: None,
        idempotency_key: idempotency_key.map(str::to_owned),
    }
}

fn item(id: u128, namespace: &str, ingest_time_ms: u64) -> MemoryItem {
    MemoryItem {
        id: Uuid::from_u128(id),
        version: 1,
        kind: MemoryKind::Semantic,
        scope: MemoryScope::Shared,
        modality: Modality::Text,
        content: ContentRef::Text {
            text: format!("memory-{id}"),
        },
        namespace: namespace.to_owned(),
        episode_id: None,
        subject: None,
        source: "memory-method-contract".to_owned(),
        metadata: BTreeMap::new(),
        abstraction_level: 0,
        event_time: None,
        ingest_time_ms,
        schema_version: 1,
    }
}

fn item_in_episode(id: u128, namespace: &str, episode_id: &str) -> MemoryItem {
    let mut item = item(id, namespace, id as u64);
    item.episode_id = Some(episode_id.to_owned());
    item
}

fn query(namespace: &str, max_hits: usize, visibility_cutoff_ms: Option<u64>) -> MemoryQuery {
    MemoryQuery {
        input: None,
        namespace: namespace.to_owned(),
        scope: None,
        subject: None,
        episode_id: None,
        kinds: vec![MemoryKind::Semantic],
        modalities: vec![Modality::Text],
        time_range: None,
        visibility_cutoff_ms,
        graph: None,
        budget: MemoryBudget {
            max_hits,
            ..MemoryBudget::default()
        },
    }
}

fn method(path: &std::path::Path) -> ProjectionMemoryMethod {
    let store = Arc::new(JsonlProjectionStore::open(path).expect("projection should open"));
    ProjectionMemoryMethod::new(store, "projection.contract", "1.0.0")
        .expect("method manifest should be valid")
}

fn path() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("pi-agent-memory-method-{}.jsonl", Uuid::new_v4()))
}

#[test]
fn provider_contract_preserves_identity_retry_isolation_and_budget() {
    let path = path();
    let provider = method(&path);
    let namespace = "contract/run-1";
    let key = "seed-1";
    let request = IngestRequest {
        items: vec![item(1, namespace, 10), item(2, namespace, 20)],
        relations: Vec::new(),
        idempotency_key: key.to_owned(),
    };

    let first = block_on(provider.ingest(request.clone(), operation(namespace, Some(key))))
        .expect("first ingest should succeed");
    assert!(!first.deduplicated);
    let retry = block_on(provider.ingest(request, operation(namespace, Some(key))))
        .expect("same ingest should be idempotent");
    assert!(retry.deduplicated);

    let response =
        block_on(provider.retrieve(query(namespace, 1, None), operation(namespace, None)))
            .expect("bounded query should succeed");
    assert_eq!(response.hits.len(), 1);
    assert!(response.truncated);
    assert_eq!(response.hits[0].item_id, Uuid::from_u128(2));
    assert_eq!(response.hits[0].version, 1);
    assert_eq!(response.hits[0].method_id, "projection.contract");
    assert_eq!(response.hits[0].rank, 1);
    assert!(response.hits[0].score.is_none_or(f32::is_finite));

    let visible =
        block_on(provider.retrieve(query(namespace, 8, Some(15)), operation(namespace, None)))
            .expect("visibility cutoff query should succeed");
    assert_eq!(visible.hits.len(), 1);
    assert_eq!(visible.hits[0].item_id, Uuid::from_u128(1));

    let hidden =
        block_on(provider.retrieve(query(namespace, 8, Some(5)), operation(namespace, None)))
            .expect("cutoff query should succeed");
    assert!(hidden.hits.is_empty());

    let isolated = block_on(provider.retrieve(
        query("contract/run-2", 8, None),
        operation("contract/run-2", None),
    ))
    .expect("isolated namespace query should succeed");
    assert!(isolated.hits.is_empty());
    drop(provider);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}.lock", path.display()));
}

#[test]
fn provider_contract_reports_manifest_and_content_search_boundary() {
    let path = path();
    let provider = method(&path);
    let manifest = provider.manifest();

    assert!(!manifest.method_id.trim().is_empty());
    assert!(!manifest.version.trim().is_empty());
    assert!(manifest.schema_version > 0);
    assert!(manifest.capabilities.supports_graph);
    assert!(!manifest.capabilities.supports_cross_modal);

    let mut content_query = query("contract/run-1", 1, None);
    content_query.input = Some(ContentRef::Text {
        text: "semantic search".to_owned(),
    });
    let content_error =
        block_on(provider.retrieve(content_query, operation("contract/run-1", None)))
            .expect_err("content search is outside the structural provider contract");
    assert!(content_error.to_string().contains("content_search"));

    let namespace = "contract/run-1";
    let relation = MemoryRelation {
        id: Uuid::from_u128(100),
        from: Uuid::from_u128(1),
        to: Uuid::from_u128(2),
        relation_type: RelationType::ParentOf,
        weight: None,
        confidence: Some(1.0),
        valid_time: None,
        ingest_time_ms: 20,
        source: "memory-method-contract".to_owned(),
        metadata: BTreeMap::new(),
        schema_version: 1,
    };
    block_on(provider.ingest(
        IngestRequest {
            items: vec![item(1, namespace, 10), item(2, namespace, 20)],
            relations: vec![relation],
            idempotency_key: "graph-seed".to_owned(),
        },
        operation(namespace, Some("graph-seed")),
    ))
    .expect("graph seed should succeed");

    let mut graph_query = query(namespace, 2, None);
    graph_query.graph = Some(pi_agent_memory::GraphTraversal {
        start_ids: vec![Uuid::from_u128(1)],
        relation_types: vec![RelationType::ParentOf],
        direction: pi_agent_memory::TraversalDirection::Outgoing,
        max_depth: 1,
        max_nodes: 2,
    });
    let graph_response = block_on(provider.retrieve(graph_query, operation(namespace, None)))
        .expect("graph traversal should use the relation projection");
    assert_eq!(
        graph_response
            .hits
            .iter()
            .map(|hit| hit.item_id)
            .collect::<Vec<_>>(),
        vec![Uuid::from_u128(2), Uuid::from_u128(1)]
    );
    drop(provider);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}.lock", path.display()));
}

#[test]
fn provider_contract_filters_retrieval_to_one_episode() {
    let path = path();
    let provider = method(&path);
    let namespace = "contract/episodes";

    block_on(provider.ingest(
        IngestRequest {
            items: vec![
                item_in_episode(1, namespace, "episode-1"),
                item_in_episode(2, namespace, "episode-2"),
            ],
            relations: Vec::new(),
            idempotency_key: "episode-seed".to_owned(),
        },
        operation(namespace, Some("episode-seed")),
    ))
    .expect("episode seed should succeed");

    let mut query = query(namespace, 8, None);
    query.episode_id = Some("episode-2".to_owned());
    let response = block_on(provider.retrieve(query, operation(namespace, None)))
        .expect("episode-scoped retrieval should succeed");

    assert_eq!(response.hits.len(), 1);
    assert_eq!(response.hits[0].item_id, Uuid::from_u128(2));

    drop(provider);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}.lock", path.display()));
}
