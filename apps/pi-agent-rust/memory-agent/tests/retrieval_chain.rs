use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

use pi_agent_memory::{
    ContentRef, IngestReceipt, IngestRequest, MemoryBudget, MemoryCapabilities, MemoryError,
    MemoryFuture, MemoryItem, MemoryKind, MemoryManifest, MemoryMethod, MemoryQuery,
    MemoryRegistry, Modality, OperationContext, RetrieveResponse,
};
use pi_agent_memory_agent::{
    CoordinatorConfig, MemoryAgentCoordinator, MemoryEpoch, MemoryNeed, NeedHorizon, PolicyEpoch,
    TaskState, ToolObservation, ToolOutcome,
};
use pi_agent_runtime::{InjectionDecision, Revision, RunId};
use uuid::Uuid;

struct StubProvider {
    manifest: MemoryManifest,
    result: Result<RetrieveResponse, MemoryError>,
}

impl MemoryMethod for StubProvider {
    fn manifest(&self) -> MemoryManifest {
        self.manifest.clone()
    }

    fn ingest<'a>(
        &'a self,
        _request: IngestRequest,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        Box::pin(std::future::ready(Err(MemoryError::UnsupportedCapability(
            "integration test provider is read-only".to_string(),
        ))))
    }

    fn retrieve<'a>(
        &'a self,
        _query: MemoryQuery,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
        Box::pin(std::future::ready(self.result.clone()))
    }

    fn get<'a>(
        &'a self,
        item_id: Uuid,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
        Box::pin(std::future::ready(Err(MemoryError::NotFound(item_id))))
    }
}

fn manifest(method_id: &str) -> MemoryManifest {
    MemoryManifest {
        method_id: method_id.to_string(),
        version: "test-1".to_string(),
        schema_version: 1,
        capabilities: MemoryCapabilities {
            modalities: vec![Modality::Text],
            memory_kinds: vec![MemoryKind::Semantic],
            supports_graph: false,
            supports_cross_modal: false,
            supports_update: false,
            supports_delete: false,
            supports_batch: false,
            deterministic: true,
            persistent: false,
        },
    }
}

fn hit(method_id: &str, item_id: u128, text: &str, score: f32) -> pi_agent_memory::MemoryHit {
    pi_agent_memory::MemoryHit {
        item_id: Uuid::from_u128(item_id),
        version: 1,
        method_id: method_id.to_string(),
        rank: 0,
        score: Some(score),
        evidence: Vec::new(),
        projection: Some(ContentRef::Text {
            text: text.to_string(),
        }),
        provenance: format!("{method_id}:record"),
        snapshot_id: Some(format!("{method_id}:snapshot")),
        latency_ms: 1,
    }
}

fn response(method_id: &str, item_id: u128, text: &str, score: f32) -> RetrieveResponse {
    RetrieveResponse {
        query_id: Uuid::from_u128(item_id + 10_000),
        hits: vec![hit(method_id, item_id, text, score)],
        snapshot_id: Some(format!("{method_id}:snapshot")),
        latency_ms: 1,
        truncated: false,
    }
}

fn state(run_id: RunId) -> TaskState {
    TaskState {
        run_id,
        turn_id: Uuid::from_u128(20),
        namespace: "benchmark/integration".to_string(),
        runtime_revision: Revision::new(7),
        memory_epoch: MemoryEpoch::new(3),
        policy_epoch: PolicyEpoch::new(2),
        goal_summary: "fix the parser".to_string(),
        latest_query: Some("find the previous parser fix".to_string()),
        current_plan: None,
        recent_tools: vec![ToolObservation {
            tool_name: "read_file".to_string(),
            outcome: ToolOutcome::Succeeded,
            summary: "opened parser.rs".to_string(),
        }],
        active_modalities: vec![Modality::Text],
        remaining_deadline_ms: Some(500),
        context_token_budget: 128,
    }
}

fn need() -> MemoryNeed {
    MemoryNeed {
        need_id: Uuid::from_u128(30),
        query: MemoryQuery {
            input: Some(ContentRef::Text {
                text: "previous parser fix".to_string(),
            }),
            namespace: "benchmark/integration".to_string(),
            scope: None,
            subject: None,
            kinds: vec![MemoryKind::Semantic],
            modalities: vec![Modality::Text],
            time_range: None,
            visibility_cutoff_ms: Some(1_000),
            graph: None,
            budget: MemoryBudget {
                max_hits: 4,
                max_context_tokens: 128,
                ..MemoryBudget::default()
            },
        },
        provider_hints: Vec::new(),
        horizon: NeedHorizon::CurrentTurn,
        confidence: 0.9,
        estimated_benefit: 0.8,
        created_at_ms: 1_000,
        expires_at_ms: Some(1_500),
    }
}

fn operation(run_id: RunId, namespace: &str) -> OperationContext {
    OperationContext {
        namespace: namespace.to_string(),
        run_id: run_id.as_uuid(),
        session_id: None,
        episode_id: None,
        timeout_ms: Some(200),
        idempotency_key: Some("integration/run-1".to_string()),
    }
}

fn block_on<F: Future>(mut future: F) -> F::Output {
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    fn wake(_: *const ()) {}
    fn drop(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake, drop);

    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut context = Context::from_waker(&waker);
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
    }
}

#[test]
fn provider_result_reaches_runtime_injection_decision() {
    let mut registry = MemoryRegistry::new();
    registry
        .register(Arc::new(StubProvider {
            manifest: manifest("semantic-text"),
            result: Ok(response(
                "semantic-text",
                101,
                "previous parser fix used an explicit state transition",
                0.95,
            )),
        }))
        .unwrap();

    let run_id = RunId::new();
    let task = state(run_id);
    let mut coordinator =
        MemoryAgentCoordinator::new(&registry, CoordinatorConfig::default()).unwrap();
    let prepared =
        block_on(coordinator.prepare(&need(), &task, operation(run_id, &task.namespace), 1_100))
            .expect("provider result should prepare a proposal");

    assert_eq!(prepared.report.hits.len(), 1);
    assert_eq!(prepared.projected_count, 1);
    assert_eq!(prepared.proposal.candidates.len(), 1);

    let decision = prepared
        .proposal
        .evaluate(&MemoryAgentCoordinator::hook_context(&task, 1_100))
        .expect("runtime hook should evaluate the proposal");
    match decision {
        InjectionDecision::Inject { injections, .. } => {
            assert_eq!(injections.len(), 1);
            assert!(injections[0]
                .projection
                .content
                .contains("explicit state transition"));
        }
        InjectionDecision::Skip { reason, .. } => panic!("unexpected injection skip: {reason:?}"),
    }
}

#[test]
fn one_provider_failure_does_not_hide_another_provider_result() {
    let mut registry = MemoryRegistry::new();
    registry
        .register(Arc::new(StubProvider {
            manifest: manifest("a-success"),
            result: Ok(response("a-success", 201, "successful memory", 0.8)),
        }))
        .unwrap();
    registry
        .register(Arc::new(StubProvider {
            manifest: manifest("b-timeout"),
            result: Err(MemoryError::Timeout),
        }))
        .unwrap();

    let run_id = RunId::new();
    let task = state(run_id);
    let mut coordinator =
        MemoryAgentCoordinator::new(&registry, CoordinatorConfig::default()).unwrap();
    let prepared =
        block_on(coordinator.prepare(&need(), &task, operation(run_id, &task.namespace), 1_100))
            .expect("partial provider failure should remain fail-soft");

    assert_eq!(prepared.report.hits.len(), 1);
    assert_eq!(prepared.report.provider_failures.len(), 1);
    assert_eq!(prepared.report.provider_failures[0].method_id, "b-timeout");
}

#[test]
fn namespace_mismatch_is_rejected_before_provider_access() {
    let mut registry = MemoryRegistry::new();
    registry
        .register(Arc::new(StubProvider {
            manifest: manifest("semantic-text"),
            result: Ok(response("semantic-text", 301, "should not be read", 0.9)),
        }))
        .unwrap();

    let run_id = RunId::new();
    let task = state(run_id);
    let mut coordinator =
        MemoryAgentCoordinator::new(&registry, CoordinatorConfig::default()).unwrap();
    let error =
        block_on(coordinator.prepare(&need(), &task, operation(run_id, "benchmark/other"), 1_100))
            .expect_err("namespace mismatch must be rejected");

    assert!(matches!(
        error,
        pi_agent_memory_agent::CoordinatorError::NamespaceMismatch
    ));
}
