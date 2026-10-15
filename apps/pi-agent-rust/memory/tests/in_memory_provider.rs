//! Deterministic in-memory provider used only by integration tests.
//!
//! This is deliberately not part of the library. It verifies that a memory
//! method can be implemented outside `src/` using only the public protocol.
//! Retrieval filters metadata and returns IDs; `get` returns the stored content.
//! Graphs, content search, event-time filtering and deadlines are unsupported.
//! There is no model, tokenizer, database or production runtime in this fixture.

use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    sync::{Arc, Mutex},
    task::{Context, Poll, Wake, Waker},
};

use pi_agent_memory::{
    ContentRef, IngestReceipt, IngestRequest, MemoryCapabilities, MemoryError, MemoryFuture,
    MemoryHit, MemoryItem, MemoryKind, MemoryManifest, MemoryMethod, MemoryQuery, MemoryScope,
    Modality, OperationContext, RetrieveResponse,
};
use uuid::Uuid;

#[derive(Default)]
struct InMemoryProvider {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    items: BTreeMap<(String, Uuid), MemoryItem>,
    // Keep the original request to distinguish a retry from reuse with new data.
    writes: BTreeMap<(String, String), IngestRequest>,
}

fn validate_context(context: &OperationContext) -> Result<(), MemoryError> {
    if context.namespace.trim().is_empty() {
        return Err(MemoryError::InvalidRequest("empty namespace".to_string()));
    }
    match context.timeout_ms {
        Some(0) => Err(MemoryError::Timeout),
        Some(_) => Err(MemoryError::UnsupportedCapability("deadline".to_string())),
        None => Ok(()),
    }
}

fn receipt(request: &IngestRequest, deduplicated: bool) -> IngestReceipt {
    IngestReceipt {
        item_ids: request.items.iter().map(|item| item.id).collect(),
        relation_ids: Vec::new(),
        deduplicated,
        snapshot_id: None,
        telemetry: None,
    }
}

impl InMemoryProvider {
    fn ingest_now(
        &self,
        request: IngestRequest,
        context: OperationContext,
    ) -> Result<IngestReceipt, MemoryError> {
        validate_context(&context)?;
        if !request.relations.is_empty() {
            return Err(MemoryError::UnsupportedCapability(
                "graph_write".to_string(),
            ));
        }
        if request.idempotency_key.trim().is_empty()
            || context
                .idempotency_key
                .as_ref()
                .is_some_and(|key| key != &request.idempotency_key)
        {
            return Err(MemoryError::InvalidRequest(
                "ingest requires a nonempty, consistent idempotency key".to_string(),
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| MemoryError::Backend("test store lock was poisoned".to_string()))?;
        let write_key = (context.namespace.clone(), request.idempotency_key.clone());
        if let Some(previous) = state.writes.get(&write_key) {
            return if previous == &request {
                Ok(receipt(previous, true))
            } else {
                Err(MemoryError::InvalidRequest(
                    "idempotency key was reused with different data".to_string(),
                ))
            };
        }

        let mut incoming_ids = BTreeSet::new();
        // Validate the whole batch before mutating the store.
        for item in &request.items {
            if item.namespace != context.namespace {
                return Err(MemoryError::InvalidRequest(format!(
                    "item {} belongs to namespace {}, expected {}",
                    item.id, item.namespace, context.namespace
                )));
            }
            if state
                .items
                .contains_key(&(context.namespace.clone(), item.id))
                || !incoming_ids.insert(item.id)
            {
                return Err(MemoryError::InvalidRequest(format!(
                    "item {} already exists",
                    item.id
                )));
            }
        }

        let result = receipt(&request, false);
        for item in &request.items {
            state
                .items
                .insert((context.namespace.clone(), item.id), item.clone());
        }
        state.writes.insert(write_key, request);
        Ok(result)
    }

    fn retrieve_now(
        &self,
        query: MemoryQuery,
        context: OperationContext,
    ) -> Result<RetrieveResponse, MemoryError> {
        validate_context(&context)?;
        if query.namespace != context.namespace {
            return Err(MemoryError::InvalidRequest(format!(
                "query namespace {} does not match context namespace {}",
                query.namespace, context.namespace
            )));
        }
        if query.graph.is_some() {
            return Err(MemoryError::UnsupportedCapability(
                "graph_traversal".to_string(),
            ));
        }
        if query.input.is_some() || query.time_range.is_some() {
            return Err(MemoryError::UnsupportedCapability(
                "content_search_or_event_time_filter".to_string(),
            ));
        }
        if let Some(timeout) = query.budget.max_latency_ms {
            return Err(if timeout == 0 {
                MemoryError::Timeout
            } else {
                MemoryError::UnsupportedCapability("deadline".to_string())
            });
        }

        let state = self
            .state
            .lock()
            .map_err(|_| MemoryError::Backend("test store lock was poisoned".to_string()))?;
        let mut matches = state
            .items
            .values()
            .filter(|item| matches_query(item, &query))
            .cloned()
            .collect::<Vec<_>>();

        matches.sort_by(|left, right| {
            right
                .ingest_time_ms
                .cmp(&left.ingest_time_ms)
                .then_with(|| left.id.cmp(&right.id))
        });

        let truncated = matches.len() > query.budget.max_hits;
        matches.truncate(query.budget.max_hits);

        let hits = matches
            .into_iter()
            .enumerate()
            .map(|(index, item)| MemoryHit {
                item_id: item.id,
                source_record_ids: vec![item.id],
                version: item.version,
                method_id: self.manifest().method_id,
                rank: index + 1,
                score: None,
                evidence: Vec::new(),
                evidence_refs: Vec::new(),
                // No evidence or context payload: byte/token budgets consume zero here.
                projection: None,
                provenance: format!("in_memory:source={}", item.source),
                snapshot_id: None,
                latency_ms: 0,
            })
            .collect();

        Ok(RetrieveResponse {
            query_id: Uuid::new_v4(),
            hits,
            snapshot_id: None,
            latency_ms: 0,
            truncated,
            telemetry: None,
        })
    }

    fn get_now(&self, item_id: Uuid, context: OperationContext) -> Result<MemoryItem, MemoryError> {
        validate_context(&context)?;
        let state = self
            .state
            .lock()
            .map_err(|_| MemoryError::Backend("test store lock was poisoned".to_string()))?;

        state
            .items
            .get(&(context.namespace, item_id))
            .cloned()
            .ok_or(MemoryError::NotFound(item_id))
    }
}

impl MemoryMethod for InMemoryProvider {
    fn manifest(&self) -> MemoryManifest {
        MemoryManifest {
            method_id: "test.in_memory".to_string(),
            version: "0.1.0".to_string(),
            schema_version: 1,
            capabilities: MemoryCapabilities {
                // Store content/references as-is; no media decoding or semantic search.
                modalities: vec![
                    Modality::Text,
                    Modality::Image,
                    Modality::Audio,
                    Modality::Video,
                    Modality::State,
                    Modality::Action,
                    Modality::Trajectory,
                    Modality::Mixed,
                ],
                memory_kinds: vec![
                    MemoryKind::Working,
                    MemoryKind::Episodic,
                    MemoryKind::Semantic,
                    MemoryKind::Procedural,
                ],
                supports_content_search: false,
                supports_time_filter: false,
                supports_visibility_cutoff: false,
                supports_graph: false,
                supports_cross_modal: false,
                supports_update: false,
                supports_delete: false,
                supports_batch: true,
                supports_record_ingest: false,
                supports_record_get: false,
                deterministic: true,
                persistent: false,
            },
        }
    }

    fn ingest<'a>(
        &'a self,
        request: IngestRequest,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        Box::pin(async move { self.ingest_now(request, context) })
    }

    fn retrieve<'a>(
        &'a self,
        query: MemoryQuery,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
        Box::pin(async move { self.retrieve_now(query, context) })
    }

    fn get<'a>(
        &'a self,
        item_id: Uuid,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
        Box::pin(async move { self.get_now(item_id, context) })
    }
}

fn matches_query(item: &MemoryItem, query: &MemoryQuery) -> bool {
    item.namespace == query.namespace
        && query
            .scope
            .as_ref()
            .is_none_or(|scope| scope == &item.scope)
        && query
            .subject
            .as_ref()
            .is_none_or(|subject| item.subject.as_ref() == Some(subject))
        && (query.kinds.is_empty() || query.kinds.contains(&item.kind))
        && (query.modalities.is_empty() || query.modalities.contains(&item.modality))
        && query
            .visibility_cutoff_ms
            .is_none_or(|cutoff| item.ingest_time_ms <= cutoff)
}

fn text_item(namespace: &str, text: &str, ingest_time_ms: u64) -> MemoryItem {
    MemoryItem {
        id: Uuid::new_v4(),
        version: 1,
        kind: MemoryKind::Semantic,
        scope: MemoryScope::Personalized,
        modality: Modality::Text,
        content: ContentRef::Text {
            text: text.to_string(),
        },
        namespace: namespace.to_string(),
        episode_id: None,
        subject: None,
        source: "integration-test".to_string(),
        metadata: BTreeMap::new(),
        abstraction_level: 0,
        event_time: None,
        ingest_time_ms,
        schema_version: 1,
    }
}

fn context(namespace: &str) -> OperationContext {
    OperationContext {
        namespace: namespace.to_string(),
        run_id: Uuid::new_v4(),
        session_id: None,
        episode_id: None,
        timeout_ms: None,
        idempotency_key: None,
    }
}

struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}

// Only for this fixture's immediately-ready futures, not a general async executor.
fn ready<F: Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut context = Context::from_waker(&waker);

    match future.as_mut().poll(&mut context) {
        Poll::Ready(output) => output,
        Poll::Pending => panic!("test provider must complete without I/O or timers"),
    }
}

#[test]
fn public_memory_protocol_supports_ingest_retrieve_and_get() {
    let provider = InMemoryProvider::default();
    let first = text_item("bench/a", "rust memory", 20);
    let second = text_item("bench/a", "agent memory", 10);
    let first_id = first.id;
    let second_id = second.id;

    let write = IngestRequest {
        items: vec![first, second],
        relations: Vec::new(),
        idempotency_key: "write-1".to_string(),
    };
    let receipt =
        ready(provider.ingest(write.clone(), context("bench/a"))).expect("ingest should succeed");

    assert_eq!(receipt.item_ids.len(), 2);
    assert!(!receipt.deduplicated);

    let retry = ready(provider.ingest(write, context("bench/a"))).expect("retry should succeed");
    assert!(retry.deduplicated);

    let response = ready(provider.retrieve(
        MemoryQuery {
            input: None,
            namespace: "bench/a".to_string(),
            scope: Some(MemoryScope::Personalized),
            subject: None,
            episode_id: None,
            kinds: vec![MemoryKind::Semantic],
            modalities: vec![Modality::Text],
            time_range: None,
            visibility_cutoff_ms: Some(15),
            graph: None,
            budget: pi_agent_memory::MemoryBudget {
                max_hits: 1,
                ..Default::default()
            },
        },
        context("bench/a"),
    ))
    .expect("retrieve should succeed");

    assert_eq!(response.hits.len(), 1);
    assert!(!response.truncated);
    assert_eq!(response.hits[0].item_id, second_id);
    assert!(response.hits[0].projection.is_none());

    let fetched = ready(provider.get(first_id, context("bench/a"))).expect("get should succeed");
    assert_eq!(fetched.id, first_id);
    assert_eq!(
        fetched.content,
        ContentRef::Text {
            text: "rust memory".to_string(),
        }
    );
}

#[test]
fn provider_rejects_namespace_crossing_and_unimplemented_graph_search() {
    let provider = InMemoryProvider::default();
    let item = text_item("bench/a", "isolated", 1);

    let error = ready(provider.ingest(
        IngestRequest {
            items: vec![item],
            relations: Vec::new(),
            idempotency_key: "write-2".to_string(),
        },
        context("bench/b"),
    ))
    .expect_err("cross-namespace ingest should fail");
    assert!(matches!(error, MemoryError::InvalidRequest(_)));

    let error = ready(provider.retrieve(
        MemoryQuery {
            input: None,
            namespace: "bench/a".to_string(),
            scope: None,
            subject: None,
            episode_id: None,
            kinds: Vec::new(),
            modalities: Vec::new(),
            time_range: None,
            visibility_cutoff_ms: None,
            graph: Some(pi_agent_memory::GraphTraversal {
                start_ids: Vec::new(),
                relation_types: Vec::new(),
                direction: pi_agent_memory::TraversalDirection::Both,
                max_depth: 1,
                max_nodes: 4,
            }),
            budget: Default::default(),
        },
        context("bench/a"),
    ))
    .expect_err("graph search should report its capability boundary");
    assert!(matches!(
        error,
        MemoryError::UnsupportedCapability(capability) if capability == "graph_traversal"
    ));
}
