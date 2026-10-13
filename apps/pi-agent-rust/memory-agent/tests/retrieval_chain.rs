use std::{future::Future, sync::Arc};

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
    delay_ms: u64,
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
        let result = self.result.clone();
        let delay_ms = self.delay_ms;
        Box::pin(async move {
            if delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
            result
        })
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

fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap()
        .block_on(future)
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
            delay_ms: 0,
        }))
        .unwrap();

    let run_id = RunId::new();
    let task = state(run_id);
    let mut coordinator =
        MemoryAgentCoordinator::new(&registry, CoordinatorConfig::default()).unwrap();
    let runtime_context = coordinator.hook_context_for(&task, 1_100);
    let prepared = block_on(coordinator.prepare(
        &need(),
        &task,
        operation(run_id, &task.namespace),
        runtime_context,
    ))
    .expect("provider result should prepare a proposal");

    assert_eq!(prepared.report.hits.len(), 1);
    assert_eq!(prepared.projected_count, 1);
    assert_eq!(prepared.proposal.candidates.len(), 1);

    let decision = prepared
        .proposal
        .evaluate(&coordinator.hook_context_for(&task, 1_100))
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
            delay_ms: 0,
        }))
        .unwrap();
    registry
        .register(Arc::new(StubProvider {
            manifest: manifest("b-timeout"),
            result: Err(MemoryError::Timeout),
            delay_ms: 0,
        }))
        .unwrap();

    let run_id = RunId::new();
    let task = state(run_id);
    let mut coordinator =
        MemoryAgentCoordinator::new(&registry, CoordinatorConfig::default()).unwrap();
    let runtime_context = coordinator.hook_context_for(&task, 1_100);
    let prepared = block_on(coordinator.prepare(
        &need(),
        &task,
        operation(run_id, &task.namespace),
        runtime_context,
    ))
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
            delay_ms: 0,
        }))
        .unwrap();

    let run_id = RunId::new();
    let task = state(run_id);
    let mut coordinator =
        MemoryAgentCoordinator::new(&registry, CoordinatorConfig::default()).unwrap();
    let runtime_context = coordinator.hook_context_for(&task, 1_100);
    let error = block_on(coordinator.prepare(
        &need(),
        &task,
        operation(run_id, "benchmark/other"),
        runtime_context,
    ))
    .expect_err("namespace mismatch must be rejected");

    assert!(matches!(
        error,
        pi_agent_memory_agent::CoordinatorError::NamespaceMismatch
    ));
}

#[test]
fn unauthorized_provider_failure_is_fail_closed() {
    let mut registry = MemoryRegistry::new();
    registry
        .register(Arc::new(StubProvider {
            manifest: manifest("restricted"),
            result: Err(MemoryError::Unauthorized),
            delay_ms: 0,
        }))
        .unwrap();

    let run_id = RunId::new();
    let task = state(run_id);
    let mut requested = need();
    requested.provider_hints = vec!["restricted".to_string()];
    let mut coordinator =
        MemoryAgentCoordinator::new(&registry, CoordinatorConfig::default()).unwrap();
    let runtime_context = coordinator.hook_context_for(&task, 1_100);
    let error = block_on(coordinator.prepare(
        &requested,
        &task,
        operation(run_id, &task.namespace),
        runtime_context,
    ))
    .expect_err("unauthorized memory must stop the retrieval plan");

    assert!(matches!(
        error,
        pi_agent_memory_agent::CoordinatorError::Retrieval(
            pi_agent_memory_agent::RetrievalError::ProviderRejected {
                error: MemoryError::Unauthorized,
                ..
            }
        )
    ));
}

#[test]
fn global_top_k_keeps_the_best_hit_across_providers() {
    let mut registry = MemoryRegistry::new();
    registry
        .register(Arc::new(StubProvider {
            manifest: manifest("a-low"),
            result: Ok(response("a-low", 401, "low", 0.2)),
            delay_ms: 0,
        }))
        .unwrap();
    registry
        .register(Arc::new(StubProvider {
            manifest: manifest("b-high"),
            result: Ok(response("b-high", 402, "high", 0.99)),
            delay_ms: 0,
        }))
        .unwrap();

    let run_id = RunId::new();
    let task = state(run_id);
    let mut requested = need();
    requested.query.budget.max_hits = 1;
    let mut coordinator =
        MemoryAgentCoordinator::new(&registry, CoordinatorConfig::default()).unwrap();
    let runtime_context = coordinator.hook_context_for(&task, 1_100);
    let prepared = block_on(coordinator.prepare(
        &requested,
        &task,
        operation(run_id, &task.namespace),
        runtime_context,
    ))
    .expect("both providers should be queried");

    assert_eq!(prepared.report.hits.len(), 1);
    assert_eq!(prepared.report.hits[0].method_id, "b-high");
}

#[test]
fn provider_contract_and_deadline_failures_are_observable() {
    let mut registry = MemoryRegistry::new();
    registry
        .register(Arc::new(StubProvider {
            manifest: manifest("slow"),
            result: Ok(response("slow", 501, "late", 0.9)),
            delay_ms: 50,
        }))
        .unwrap();

    let run_id = RunId::new();
    let task = state(run_id);
    let mut requested = need();
    requested.provider_hints = vec!["slow".to_string()];
    let mut context = operation(run_id, &task.namespace);
    context.timeout_ms = Some(5);
    let mut coordinator =
        MemoryAgentCoordinator::new(&registry, CoordinatorConfig::default()).unwrap();
    let runtime_context = coordinator.hook_context_for(&task, 1_100);
    let prepared = block_on(coordinator.prepare(&requested, &task, context, runtime_context))
        .expect("a timed out provider should be fail-soft");
    assert!(prepared.report.hits.is_empty());
    assert!(matches!(
        prepared.report.provider_failures[0].error,
        MemoryError::Timeout
    ));

    let mut mismatch_registry = MemoryRegistry::new();
    mismatch_registry
        .register(Arc::new(StubProvider {
            manifest: manifest("claimed"),
            result: Ok(response("other", 502, "spoofed", 0.9)),
            delay_ms: 0,
        }))
        .unwrap();
    let mut mismatch =
        MemoryAgentCoordinator::new(&mismatch_registry, CoordinatorConfig::default()).unwrap();
    requested.provider_hints = vec!["claimed".to_string()];
    let runtime_context = mismatch.hook_context_for(&task, 1_100);
    let error = block_on(mismatch.prepare(
        &requested,
        &task,
        operation(run_id, &task.namespace),
        runtime_context,
    ))
    .expect_err("provider identity spoofing must be rejected");
    assert!(matches!(
        error,
        pi_agent_memory_agent::CoordinatorError::Retrieval(
            pi_agent_memory_agent::RetrievalError::ProviderContractViolation { .. }
        )
    ));
}

#[test]
fn expired_need_is_rejected_before_provider_access() {
    let mut registry = MemoryRegistry::new();
    registry
        .register(Arc::new(StubProvider {
            manifest: manifest("semantic-text"),
            result: Ok(response("semantic-text", 601, "expired", 0.9)),
            delay_ms: 0,
        }))
        .unwrap();
    let run_id = RunId::new();
    let task = state(run_id);
    let mut requested = need();
    requested.expires_at_ms = Some(1_000);
    let mut coordinator =
        MemoryAgentCoordinator::new(&registry, CoordinatorConfig::default()).unwrap();
    let runtime_context = coordinator.hook_context_for(&task, 1_100);
    let error = block_on(coordinator.prepare(
        &requested,
        &task,
        operation(run_id, &task.namespace),
        runtime_context,
    ))
    .expect_err("expired needs must not reach providers");
    assert!(matches!(
        error,
        pi_agent_memory_agent::CoordinatorError::Retrieval(
            pi_agent_memory_agent::RetrievalError::ExpiredNeed
        )
    ));
}
