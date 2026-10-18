//! Connects memory-need prediction to version-bound preparation workers.
//!
//! A predicted need is still only a candidate. The scheduler binds it to the
//! runtime revision and expiry, runs the existing memory coordinator in the
//! background, and keeps the resulting proposal until the caller is ready to
//! pass it through the normal runtime injection gate.

use std::{collections::BTreeMap, fmt, future::Future, sync::Arc};

use pi_agent_memory::{MemoryRegistry, Modality, OperationContext};
use pi_agent_memory_agent::{
    CoordinatorConfig as MemoryCoordinatorConfig, HeuristicNeedPredictor, MemoryAgentCoordinator,
    MemoryEpoch, MemoryNeed, NeedPredictor, PolicyEpoch, PredictionError, PreparedMemory,
    TaskState,
};
use pi_agent_runtime::{
    canonical_json_digest, ActionId, Budget, EventLedger, MemoryHookContext, MemoryProposal,
    PreparationEffectClass, PreparationError, PreparationPlan, PreparationStatus,
    PreparationWorkResult, ResourceScope, RunId, RuntimeCoordinator,
};
use pi_agent_rust::{AgentLoop, ChatModel, MemoryHookConfig, MemoryHookFuture, MemoryHookRequest};
use serde_json::json;
use thiserror::Error;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::preparation_scheduler::{
    PreparationScheduler, PreparationSchedulerError, PreparationTask,
};
use crate::prepared_memory_store::{PreparedMemoryStore, PreparedMemoryStoreError};

/// Errors raised while turning a predicted need into a prepared memory.
#[derive(Debug, Error)]
pub enum SpeculativeMemoryError {
    #[error("memory prediction failed: {0}")]
    Prediction(#[from] PredictionError),
    #[error("memory preparation plan is invalid: {0}")]
    InvalidPlan(String),
    #[error("memory preparation scheduler failed: {0}")]
    Scheduler(#[from] PreparationSchedulerError),
    #[error("memory preparation worker failed: {0}")]
    WorkerFailed(String),
    #[error("prepared memory for plan {0} was not found")]
    MissingPreparedMemory(Uuid),
    #[error("prepared memory store failed: {0}")]
    PreparedStore(#[from] PreparedMemoryStoreError),
}

/// Predicts recency-based needs for providers that expose structural filters
/// but do not implement semantic content search.
#[derive(Debug, Clone, Copy, Default)]
pub struct StructuralNeedPredictor;

impl NeedPredictor for StructuralNeedPredictor {
    fn predict(&self, state: &TaskState, now_ms: u64) -> Result<Vec<MemoryNeed>, PredictionError> {
        let mut needs = HeuristicNeedPredictor::default().predict(state, now_ms)?;
        for need in &mut needs {
            need.query.input = None;
        }
        Ok(needs)
    }
}

/// A predicted need together with its asynchronously prepared result.
pub struct SpeculativeMemoryTask {
    plan_id: Uuid,
    need_id: Uuid,
    task: PreparationTask,
    prepared: Arc<Mutex<BTreeMap<Uuid, PreparedMemory>>>,
}

impl SpeculativeMemoryTask {
    pub const fn plan_id(&self) -> Uuid {
        self.plan_id
    }

    pub const fn need_id(&self) -> Uuid {
        self.need_id
    }

    /// Waits for preparation and returns the proposal produced by the memory
    /// coordinator. The caller still decides when to evaluate and inject it.
    pub async fn wait(self) -> Result<PreparedMemory, SpeculativeMemoryError> {
        let plan_id = self.plan_id;
        let result = match self.task.wait().await {
            Ok(result) => result,
            Err(error) => {
                self.prepared.lock().await.remove(&plan_id);
                return Err(error.into());
            }
        };
        match result {
            PreparationWorkResult::Ready { .. } => self
                .prepared
                .lock()
                .await
                .remove(&plan_id)
                .ok_or(SpeculativeMemoryError::MissingPreparedMemory(plan_id)),
            PreparationWorkResult::Failed { error }
            | PreparationWorkResult::Unknown { reason: error } => {
                self.prepared.lock().await.remove(&plan_id);
                Err(SpeculativeMemoryError::WorkerFailed(error))
            }
        }
    }
}

/// Runs predicted memory retrievals through the runtime preparation boundary.
pub struct SpeculativeMemoryScheduler<L, P> {
    scheduler: PreparationScheduler<L>,
    registry: Arc<MemoryRegistry>,
    predictor: Arc<P>,
    coordinator_config: MemoryCoordinatorConfig,
    prepared: Arc<Mutex<BTreeMap<Uuid, PreparedMemory>>>,
    prepared_store: Option<PreparedMemoryStore>,
}

impl<L, P> SpeculativeMemoryScheduler<L, P> {
    pub fn new(
        coordinator: RuntimeCoordinator<L>,
        max_concurrency: usize,
        registry: Arc<MemoryRegistry>,
        predictor: P,
        coordinator_config: MemoryCoordinatorConfig,
    ) -> Result<Self, PreparationSchedulerError> {
        Ok(Self {
            scheduler: PreparationScheduler::new(coordinator, max_concurrency)?,
            registry,
            predictor: Arc::new(predictor),
            coordinator_config,
            prepared: Arc::new(Mutex::new(BTreeMap::new())),
            prepared_store: None,
        })
    }

    pub fn coordinator(&self) -> Arc<Mutex<RuntimeCoordinator<L>>> {
        self.scheduler.coordinator()
    }

    /// Enables durable snapshots for proposals produced by this scheduler.
    pub fn with_prepared_memory_store(mut self, store: PreparedMemoryStore) -> Self {
        self.prepared_store = Some(store);
        self
    }
}

impl<L, P> SpeculativeMemoryScheduler<L, P>
where
    L: EventLedger + Send + 'static,
    P: NeedPredictor + 'static,
{
    /// Brings the preparation coordinator up to the revision observed by the
    /// main agent and invalidates older in-flight plans.
    pub async fn synchronize_runtime(
        &self,
        context: &MemoryHookContext,
    ) -> Result<(), SpeculativeMemoryError> {
        let coordinator = self.scheduler.coordinator();
        let mut coordinator = coordinator.lock().await;
        coordinator
            .synchronize_revision(context.run_id, context.current_revision, context.now_ms)
            .map_err(|error| {
                SpeculativeMemoryError::Scheduler(PreparationSchedulerError::Coordinator(error))
            })
    }

    /// Reconciles preparation plans restored after a worker-process restart.
    pub async fn recover_interrupted(&self, now_ms: u64) -> Result<usize, SpeculativeMemoryError> {
        self.scheduler
            .recover_interrupted(now_ms)
            .await
            .map_err(SpeculativeMemoryError::Scheduler)
    }

    async fn consume_read_only(
        &self,
        plan_id: Uuid,
        context: &MemoryHookContext,
    ) -> Result<(), SpeculativeMemoryError> {
        self.scheduler
            .consume_read_only(plan_id, context.now_ms)
            .await
            .map_err(SpeculativeMemoryError::Scheduler)?;
        if let Some(store) = &self.prepared_store {
            store
                .remove(
                    context.run_id,
                    plan_id,
                    context.current_revision,
                    context.now_ms,
                )
                .await?;
        }
        Ok(())
    }

    /// Restores one durable proposal whose runtime identity still matches the
    /// current hook context. Stale or already-terminal plans are discarded.
    pub async fn restore_ready_proposal(
        &self,
        context: &MemoryHookContext,
    ) -> Result<Option<MemoryProposal>, SpeculativeMemoryError> {
        let Some(store) = &self.prepared_store else {
            return Ok(None);
        };
        let proposals = store.proposals_for_run(context.run_id).await?;
        for (plan_id, proposal) in proposals {
            if proposal.namespace != context.namespace
                || proposal.base_revision != context.current_revision
                || proposal.memory_revision != context.memory_revision
                || proposal.policy_revision != context.policy_revision
            {
                store
                    .remove(
                        context.run_id,
                        plan_id,
                        context.current_revision,
                        context.now_ms,
                    )
                    .await?;
                continue;
            }
            let plan = {
                let coordinator = self.scheduler.coordinator();
                let coordinator = coordinator.lock().await;
                coordinator.preparations().get(plan_id).cloned()
            };
            let Some(plan) = plan else {
                store
                    .remove(
                        context.run_id,
                        plan_id,
                        context.current_revision,
                        context.now_ms,
                    )
                    .await?;
                continue;
            };
            if plan.status != PreparationStatus::Ready
                || !plan.artifacts.is_empty()
                || !plan.evidence.is_empty()
            {
                if plan.status != PreparationStatus::Ready {
                    store
                        .remove(
                            context.run_id,
                            plan_id,
                            context.current_revision,
                            context.now_ms,
                        )
                        .await?;
                }
                continue;
            }
            match self.consume_read_only(plan_id, context).await {
                Ok(()) => return Ok(Some(proposal)),
                Err(SpeculativeMemoryError::Scheduler(PreparationSchedulerError::Coordinator(
                    pi_agent_runtime::CoordinatorError::Preparation(PreparationError::Expired(_)),
                ))) => {
                    // The runtime has durably invalidated the expired plan. The
                    // proposal snapshot is no longer admissible, so remove its
                    // projection tombstone and let this hook continue without
                    // failing the agent request.
                    store
                        .remove(
                            context.run_id,
                            plan_id,
                            context.current_revision,
                            context.now_ms,
                        )
                        .await?;
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
        Ok(None)
    }

    /// Applies provider-specific evidence to restored unknown preparations.
    pub async fn reconcile_unknown<F, Fut, E>(
        &self,
        now_ms: u64,
        reconciler: F,
    ) -> Result<usize, SpeculativeMemoryError>
    where
        F: Fn(PreparationPlan) -> Fut + Send + Sync,
        Fut: Future<Output = Result<PreparationWorkResult, E>> + Send,
        E: fmt::Display,
    {
        self.scheduler
            .reconcile_unknown(now_ms, reconciler)
            .await
            .map_err(SpeculativeMemoryError::Scheduler)
    }

    /// Predicts needs from one task snapshot and submits all of them without
    /// waiting for any individual provider retrieval to finish.
    pub async fn predict_and_submit(
        &self,
        state: TaskState,
        runtime_context: MemoryHookContext,
        now_ms: u64,
    ) -> Result<Vec<SpeculativeMemoryTask>, SpeculativeMemoryError> {
        validate_context(&state, &runtime_context)?;
        let needs = self.predictor.predict(&state, now_ms)?;
        let mut tasks = Vec::with_capacity(needs.len());
        for need in needs {
            let plan = preparation_plan(
                &need,
                &state,
                &runtime_context,
                now_ms,
                self.coordinator_config.proposal_ttl_ms,
            )?;
            let plan_id = plan.plan_id;
            let operation = operation_for(&need, &state);
            let registry = Arc::clone(&self.registry);
            let prepared = Arc::clone(&self.prepared);
            let worker_need = need.clone();
            let worker_state = state.clone();
            let worker_context = runtime_context.clone();
            let worker_revision = runtime_context.current_revision;
            let worker_now_ms = runtime_context.now_ms;
            let coordinator_config = self.coordinator_config;
            let prepared_store = self.prepared_store.clone();
            let task = self
                .scheduler
                .submit(plan, now_ms, move |_plan| async move {
                    let mut coordinator =
                        MemoryAgentCoordinator::new(&registry, coordinator_config)
                            .map_err(|error| error.to_string())?;
                    let result = coordinator
                        .prepare(&worker_need, &worker_state, operation, worker_context)
                        .await
                        .map_err(|error| error.to_string())?;
                    if let Some(store) = prepared_store {
                        store
                            .put(plan_id, &result.proposal, worker_revision, worker_now_ms)
                            .await
                            .map_err(|error| error.to_string())?;
                    }
                    prepared.lock().await.insert(plan_id, result);
                    Ok::<_, String>(PreparationWorkResult::Ready {
                        artifacts: Vec::new(),
                        evidence: Vec::new(),
                    })
                })
                .await?;
            tasks.push(SpeculativeMemoryTask {
                plan_id,
                need_id: need.need_id,
                task,
                prepared: Arc::clone(&self.prepared),
            });
        }
        Ok(tasks)
    }
}

struct PendingSpeculation {
    run_id: RunId,
    tasks: Vec<SpeculativeMemoryTask>,
}

/// Installs speculative memory preparation into the main agent loop.
///
/// The first hook invocation schedules predicted retrievals and immediately
/// lets the model run. The following invocation waits for that work, keeps a
/// proposal only when it still matches the current runtime and memory
/// revisions, and then starts the next prediction window. Retrieval failures
/// are best effort: the agent remains usable while the runtime ledger still
/// records every preparation transition.
///
/// The scheduler is deliberately bound to one runtime [`RunId`]. Construct it
/// with the same identity passed to [`AgentLoop::run_with_context`]; this keeps
/// preparation events and model/tool events in one auditable run without
/// silently rebinding a live coordinator to another run.
pub struct SpeculativeMemoryHookAdapter<L, P> {
    scheduler: Arc<SpeculativeMemoryScheduler<L, P>>,
    hook_config: MemoryHookConfig,
    pending: Arc<Mutex<Option<PendingSpeculation>>>,
}

impl<L, P> SpeculativeMemoryHookAdapter<L, P> {
    pub fn new(scheduler: SpeculativeMemoryScheduler<L, P>, hook_config: MemoryHookConfig) -> Self {
        Self {
            scheduler: Arc::new(scheduler),
            hook_config,
            pending: Arc::new(Mutex::new(None)),
        }
    }
}

impl<L, P> SpeculativeMemoryHookAdapter<L, P>
where
    L: EventLedger + Send + 'static,
    P: NeedPredictor + 'static,
{
    /// Attaches the asynchronous prediction window to an existing agent.
    pub fn attach<M>(self, agent: AgentLoop<M>) -> AgentLoop<M>
    where
        M: ChatModel,
    {
        let scheduler = Arc::clone(&self.scheduler);
        let pending = Arc::clone(&self.pending);
        agent.with_memory_hook(self.hook_config, move |request| {
            let scheduler = Arc::clone(&scheduler);
            let pending = Arc::clone(&pending);
            Box::pin(async move {
                scheduler
                    .synchronize_runtime(&request.context)
                    .await
                    .map_err(|error| error.to_string())?;

                let previous = pending.lock().await.take();
                let mut proposal = if let Some(previous) = previous {
                    if previous.run_id == request.context.run_id {
                        await_matching_proposal(&scheduler, previous.tasks, &request.context).await
                    } else {
                        None
                    }
                } else {
                    None
                };
                if proposal.is_none() {
                    proposal = scheduler
                        .restore_ready_proposal(&request.context)
                        .await
                        .map_err(|error| error.to_string())?;
                }

                let state = task_state_from_request(&request);
                let tasks = scheduler
                    .predict_and_submit(state, request.context.clone(), request.context.now_ms)
                    .await
                    .map_err(|error| error.to_string())?;
                if !tasks.is_empty() {
                    pending.lock().await.replace(PendingSpeculation {
                        run_id: request.context.run_id,
                        tasks,
                    });
                }
                Ok(proposal)
            }) as MemoryHookFuture
        })
    }
}

async fn await_matching_proposal<L, P>(
    scheduler: &SpeculativeMemoryScheduler<L, P>,
    tasks: Vec<SpeculativeMemoryTask>,
    context: &MemoryHookContext,
) -> Option<pi_agent_runtime::MemoryProposal>
where
    L: EventLedger + Send + 'static,
    P: NeedPredictor + 'static,
{
    let mut selected = None;
    for task in tasks {
        let plan_id = task.plan_id();
        if let Ok(prepared) = task.wait().await {
            let proposal = prepared.proposal;
            if proposal.run_id == context.run_id
                && proposal.namespace == context.namespace
                && proposal.base_revision == context.current_revision
                && proposal.memory_revision == context.memory_revision
                && proposal.policy_revision == context.policy_revision
                && scheduler.consume_read_only(plan_id, context).await.is_ok()
                && selected.is_none()
            {
                selected = Some(proposal);
            }
        }
    }
    selected
}

fn task_state_from_request(request: &MemoryHookRequest) -> TaskState {
    let latest_query = request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .and_then(|message| message.content.as_deref())
        .map(str::trim)
        .filter(|content| !content.is_empty())
        .map(str::to_owned);
    let goal_summary = latest_query
        .clone()
        .unwrap_or_else(|| "agent task".to_owned());
    TaskState {
        run_id: request.context.run_id,
        turn_id: request.context.turn_id.unwrap_or_else(Uuid::new_v4),
        namespace: request.context.namespace.clone(),
        runtime_revision: request.context.current_revision,
        memory_epoch: MemoryEpoch::new(request.context.memory_revision.value()),
        policy_epoch: PolicyEpoch::new(request.context.policy_revision.value()),
        goal_summary,
        latest_query,
        current_plan: None,
        recent_tools: Vec::new(),
        active_modalities: vec![Modality::Text],
        remaining_deadline_ms: None,
        context_token_budget: request.context.max_tokens,
    }
}

fn validate_context(
    state: &TaskState,
    runtime_context: &MemoryHookContext,
) -> Result<(), SpeculativeMemoryError> {
    if state.run_id != runtime_context.run_id {
        return Err(SpeculativeMemoryError::InvalidPlan(
            "task and runtime context run ids differ".to_owned(),
        ));
    }
    if state.namespace != runtime_context.namespace {
        return Err(SpeculativeMemoryError::InvalidPlan(
            "task and runtime context namespaces differ".to_owned(),
        ));
    }
    if state.runtime_revision != runtime_context.current_revision {
        return Err(SpeculativeMemoryError::InvalidPlan(
            "task and runtime context revisions differ".to_owned(),
        ));
    }
    if state.memory_epoch.value() != runtime_context.memory_revision.value() {
        return Err(SpeculativeMemoryError::InvalidPlan(
            "task and runtime memory revisions differ".to_owned(),
        ));
    }
    if state.policy_epoch.value() != runtime_context.policy_revision.value() {
        return Err(SpeculativeMemoryError::InvalidPlan(
            "task and runtime policy revisions differ".to_owned(),
        ));
    }
    Ok(())
}

fn preparation_plan(
    need: &MemoryNeed,
    state: &TaskState,
    runtime_context: &MemoryHookContext,
    now_ms: u64,
    default_ttl_ms: u64,
) -> Result<PreparationPlan, SpeculativeMemoryError> {
    let arguments = json!({
        "need_id": need.need_id,
        "query": need.query,
        "horizon": need.horizon,
    });
    let mut resources = vec![format!("memory:namespace:{}", state.namespace)];
    resources.extend(
        need.provider_hints
            .iter()
            .map(|provider| format!("memory:provider:{provider}")),
    );
    if need.provider_hints.is_empty() {
        resources.push("memory:provider:any".to_owned());
    }
    let expires_at_ms = need
        .expires_at_ms
        .unwrap_or_else(|| now_ms.saturating_add(default_ttl_ms));
    PreparationPlan::new(
        state.run_id,
        ActionId::from_uuid(need.need_id),
        runtime_context.current_revision,
        "memory.retrieve",
        canonical_json_digest(&arguments),
        PreparationEffectClass::ReadOnly,
        ResourceScope::new(resources),
        Budget {
            wall_time_ms: state.remaining_deadline_ms,
            tokens: Some(need.query.budget.max_context_tokens as u64),
            tool_calls: Some(1),
        },
        need.confidence,
        now_ms,
        expires_at_ms,
    )
    .map_err(|error| SpeculativeMemoryError::InvalidPlan(error.to_string()))
}

fn operation_for(need: &MemoryNeed, state: &TaskState) -> OperationContext {
    OperationContext {
        namespace: state.namespace.clone(),
        run_id: state.run_id.as_uuid(),
        session_id: None,
        episode_id: need
            .query
            .episode_id
            .as_deref()
            .and_then(|value| Uuid::parse_str(value).ok()),
        timeout_ms: state.remaining_deadline_ms,
        idempotency_key: Some(format!("speculative-memory:{}", need.need_id)),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    use pi_agent_memory::{
        IngestReceipt, IngestRequest, MemoryCapabilities, MemoryError, MemoryFuture, MemoryItem,
        MemoryKind, MemoryManifest, MemoryMethod, MemoryQuery, Modality, RetrieveResponse,
    };
    use pi_agent_memory_agent::{
        CoordinatorConfig, HeuristicNeedPredictor, MemoryEpoch, NeedPredictor, PolicyEpoch,
        TaskState,
    };
    use pi_agent_runtime::{
        ActionId, AdmissionPolicy, Budget, InMemoryLedger, MemoryHookContext, MemoryProposal,
        MemoryRevision, PolicyRevision, PreparationEffectClass, PreparationPlan, PreparationStatus,
        ResourceScope, Revision, RunId, RuntimeCoordinator,
    };
    use tempfile::tempdir;
    use uuid::Uuid;

    use super::{SpeculativeMemoryScheduler, StructuralNeedPredictor};
    use crate::PreparedMemoryStore;

    struct EmptyTextProvider;

    impl MemoryMethod for EmptyTextProvider {
        fn manifest(&self) -> MemoryManifest {
            MemoryManifest {
                method_id: "empty-text".to_owned(),
                version: "1".to_owned(),
                schema_version: 1,
                capabilities: MemoryCapabilities {
                    modalities: vec![Modality::Text],
                    memory_kinds: vec![
                        MemoryKind::Working,
                        MemoryKind::Episodic,
                        MemoryKind::Semantic,
                        MemoryKind::Procedural,
                    ],
                    supports_content_search: true,
                    supports_time_filter: true,
                    supports_visibility_cutoff: true,
                    supports_graph: false,
                    supports_cross_modal: false,
                    supports_update: false,
                    supports_delete: false,
                    supports_batch: false,
                    supports_record_ingest: false,
                    supports_record_get: false,
                    deterministic: true,
                    persistent: false,
                },
            }
        }

        fn ingest<'a>(
            &'a self,
            _request: IngestRequest,
            _context: pi_agent_memory::OperationContext,
        ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
            Box::pin(std::future::ready(Err(MemoryError::UnsupportedCapability(
                "test provider is read-only".to_owned(),
            ))))
        }

        fn retrieve<'a>(
            &'a self,
            _query: MemoryQuery,
            _context: pi_agent_memory::OperationContext,
        ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
            Box::pin(std::future::ready(Ok(RetrieveResponse {
                query_id: Uuid::new_v4(),
                hits: Vec::new(),
                snapshot_id: Some("empty".to_owned()),
                latency_ms: 0,
                truncated: false,
                telemetry: None,
            })))
        }

        fn get<'a>(
            &'a self,
            _item_id: Uuid,
            _context: pi_agent_memory::OperationContext,
        ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
            Box::pin(std::future::ready(Err(MemoryError::InvalidRequest(
                "test provider has no items".to_owned(),
            ))))
        }
    }

    fn task(run_id: RunId) -> TaskState {
        TaskState {
            run_id,
            turn_id: Uuid::new_v4(),
            namespace: "bench/speculative".to_owned(),
            runtime_revision: Revision::INITIAL,
            memory_epoch: MemoryEpoch::INITIAL,
            policy_epoch: PolicyEpoch::INITIAL,
            goal_summary: "inspect memory".to_owned(),
            latest_query: Some("find prior memory".to_owned()),
            current_plan: Some("prepare the next retrieval".to_owned()),
            recent_tools: Vec::new(),
            active_modalities: vec![Modality::Text],
            remaining_deadline_ms: Some(1_000),
            context_token_budget: 512,
        }
    }

    fn empty_proposal(context: &MemoryHookContext) -> MemoryProposal {
        MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: context.namespace.clone(),
            base_revision: context.current_revision,
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: Vec::new(),
            created_at_ms: context.now_ms,
            expires_at_ms: Some(context.now_ms + 1_000),
            retrieval_summary: None,
        }
    }

    #[test]
    fn structural_predictor_removes_semantic_query_input() {
        let state = task(RunId::new());
        let needs = StructuralNeedPredictor
            .predict(&state, 1_000)
            .expect("structural prediction should reuse the heuristic bounds");
        assert!(!needs.is_empty());
        assert!(needs.iter().all(|need| need.query.input.is_none()));
    }

    #[tokio::test]
    async fn predictor_outputs_are_prepared_through_runtime_scheduler() {
        let run_id = RunId::new();
        let state = task(run_id);
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("test clock should be valid")
            .as_millis()
            .min(u64::MAX as u128) as u64;
        let context = MemoryHookContext {
            run_id,
            turn_id: Some(state.turn_id),
            namespace: state.namespace.clone(),
            current_revision: Revision::INITIAL,
            memory_revision: MemoryRevision::INITIAL,
            policy_revision: PolicyRevision::INITIAL,
            now_ms,
            max_tokens: 512,
            max_candidates: 8,
            max_projection_bytes: 8_192,
        };
        let mut registry = pi_agent_memory::MemoryRegistry::new();
        registry
            .register(Arc::new(EmptyTextProvider))
            .expect("test provider should register");
        let scheduler = SpeculativeMemoryScheduler::new(
            RuntimeCoordinator::new(
                run_id,
                AdmissionPolicy::deny_all(),
                InMemoryLedger::default(),
            ),
            2,
            Arc::new(registry),
            HeuristicNeedPredictor::default(),
            CoordinatorConfig::default(),
        )
        .expect("scheduler should be created");

        let tasks = scheduler
            .predict_and_submit(state, context.clone(), now_ms)
            .await
            .expect("predicted needs should be submitted");
        assert_eq!(tasks.len(), 2);
        let first = tasks[0].need_id();
        let second = tasks[1].need_id();
        let proposal = super::await_matching_proposal(&scheduler, tasks, &context)
            .await
            .expect("an empty retrieval still produces a proposal");
        assert!(proposal.candidates.is_empty());
        assert_ne!(first, second);

        let coordinator = scheduler.coordinator();
        let coordinator = coordinator.lock().await;
        assert_eq!(coordinator.preparations().len(), 2);
        assert!(coordinator
            .preparations()
            .plans()
            .all(|plan| plan.status == pi_agent_runtime::PreparationStatus::Consumed));
    }

    #[tokio::test]
    async fn durable_ready_proposal_is_restored_and_consumed() {
        let run_id = RunId::new();
        let state = task(run_id);
        let context = MemoryHookContext {
            run_id,
            turn_id: Some(state.turn_id),
            namespace: state.namespace.clone(),
            current_revision: Revision::INITIAL,
            memory_revision: MemoryRevision::INITIAL,
            policy_revision: PolicyRevision::INITIAL,
            now_ms: 1_000,
            max_tokens: 512,
            max_candidates: 8,
            max_projection_bytes: 8_192,
        };
        let directory = tempdir().expect("temporary proposal directory should be created");
        let store = PreparedMemoryStore::open(directory.path().join("prepared.jsonl"))
            .expect("prepared memory store should open");
        let scheduler = SpeculativeMemoryScheduler::new(
            RuntimeCoordinator::new(
                run_id,
                AdmissionPolicy::deny_all(),
                InMemoryLedger::default(),
            ),
            1,
            Arc::new(pi_agent_memory::MemoryRegistry::new()),
            StructuralNeedPredictor,
            CoordinatorConfig::default(),
        )
        .expect("scheduler should be created")
        .with_prepared_memory_store(store.clone());
        let plan = PreparationPlan::new(
            run_id,
            ActionId::new(),
            Revision::INITIAL,
            "memory.retrieve",
            "sha256:test",
            PreparationEffectClass::ReadOnly,
            ResourceScope::new(["memory:test"]),
            Budget {
                wall_time_ms: Some(1_000),
                tokens: Some(128),
                tool_calls: Some(1),
            },
            0.9,
            context.now_ms,
            context.now_ms + 2_000,
        )
        .expect("preparation plan should be valid");
        let plan_id = plan.plan_id;
        {
            let coordinator = scheduler.coordinator();
            let mut coordinator = coordinator.lock().await;
            coordinator
                .submit_preparation(plan, context.now_ms)
                .expect("plan should be submitted");
            coordinator
                .start_preparation(plan_id, context.now_ms + 1)
                .expect("plan should start");
            coordinator
                .mark_preparation_ready_with_artifacts_and_evidence(
                    plan_id,
                    Vec::new(),
                    Vec::new(),
                    context.now_ms + 2,
                )
                .expect("plan should be ready");
        }
        let proposal = empty_proposal(&context);
        store
            .put(plan_id, &proposal, context.current_revision, context.now_ms)
            .await
            .expect("proposal should be durable");

        assert_eq!(
            scheduler
                .restore_ready_proposal(&context)
                .await
                .expect("proposal should restore"),
            Some(proposal)
        );
        assert!(store
            .get(run_id, plan_id)
            .await
            .expect("consume tombstone should be readable")
            .is_none());
        let coordinator = scheduler.coordinator();
        let coordinator = coordinator.lock().await;
        assert_eq!(
            coordinator.preparations().get(plan_id).unwrap().status,
            PreparationStatus::Consumed
        );
        store.close().await.expect("proposal store should close");
    }

    #[tokio::test]
    async fn stale_proposal_is_removed_before_restore() {
        let run_id = RunId::new();
        let state = task(run_id);
        let context = MemoryHookContext {
            run_id,
            turn_id: Some(state.turn_id),
            namespace: state.namespace.clone(),
            current_revision: Revision::INITIAL,
            memory_revision: MemoryRevision::INITIAL,
            policy_revision: PolicyRevision::INITIAL,
            now_ms: 1_000,
            max_tokens: 512,
            max_candidates: 8,
            max_projection_bytes: 8_192,
        };
        let directory = tempdir().expect("temporary proposal directory should be created");
        let store = PreparedMemoryStore::open(directory.path().join("prepared.jsonl"))
            .expect("prepared memory store should open");
        let scheduler = SpeculativeMemoryScheduler::new(
            RuntimeCoordinator::new(
                run_id,
                AdmissionPolicy::deny_all(),
                InMemoryLedger::default(),
            ),
            1,
            Arc::new(pi_agent_memory::MemoryRegistry::new()),
            StructuralNeedPredictor,
            CoordinatorConfig::default(),
        )
        .expect("scheduler should be created")
        .with_prepared_memory_store(store.clone());
        let plan_id = Uuid::new_v4();
        let mut proposal = empty_proposal(&context);
        proposal.memory_revision = MemoryRevision::new(1);
        store
            .put(plan_id, &proposal, context.current_revision, context.now_ms)
            .await
            .expect("stale proposal should be durable");

        assert_eq!(
            scheduler
                .restore_ready_proposal(&context)
                .await
                .expect("stale proposal should be skipped"),
            None
        );
        assert!(store
            .get(run_id, plan_id)
            .await
            .expect("stale proposal tombstone should be readable")
            .is_none());
        store.close().await.expect("proposal store should close");
    }
}
