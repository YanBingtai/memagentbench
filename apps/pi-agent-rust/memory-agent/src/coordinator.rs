//! End-to-end orchestration for one memory need.

use std::{error::Error, fmt};

use pi_agent_memory::{MemoryRegistry, OperationContext};
use pi_agent_runtime::{
    MemoryCandidate, MemoryHookContext, MemoryProposal, MemoryRevision, PolicyRevision, Revision,
};
use uuid::Uuid;

use crate::{
    contracts::{MemoryEpoch, MemoryNeed, PolicyEpoch, TaskState},
    projection::{ProjectionError, ProjectionRenderer},
    retrieval::{RetrievalCoordinator, RetrievalError, RetrievalReport},
    working_set::{WorkingSet, WorkingSetConfig, WorkingSetError},
};

/// Conservative defaults for a request-local memory control plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinatorConfig {
    pub max_working_set_candidates: usize,
    pub max_projection_chars: usize,
    pub proposal_ttl_ms: u64,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            max_working_set_candidates: 16,
            max_projection_chars: crate::projection::DEFAULT_MAX_CHARS,
            proposal_ttl_ms: 2_000,
        }
    }
}

/// Output of retrieval and preparation. Runtime still performs the final
/// revision, expiry, and token-budget decision through `MemoryProposal::evaluate`.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedMemory {
    pub report: RetrievalReport,
    pub projected_count: usize,
    pub skipped_count: usize,
    pub proposal: MemoryProposal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorError {
    InvalidConfiguration(String),
    RunMismatch,
    NamespaceMismatch,
    Retrieval(RetrievalError),
    Projection(ProjectionError),
    WorkingSet(WorkingSetError),
}

impl fmt::Display for CoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => write!(
                formatter,
                "invalid memory coordinator configuration: {message}"
            ),
            Self::RunMismatch => {
                write!(formatter, "memory operation run does not match task state")
            }
            Self::NamespaceMismatch => write!(
                formatter,
                "memory operation namespace does not match task state"
            ),
            Self::Retrieval(error) => write!(formatter, "memory retrieval failed: {error}"),
            Self::Projection(error) => write!(formatter, "memory projection failed: {error}"),
            Self::WorkingSet(error) => write!(formatter, "memory working set failed: {error}"),
        }
    }
}

impl Error for CoordinatorError {}

impl From<RetrievalError> for CoordinatorError {
    fn from(error: RetrievalError) -> Self {
        Self::Retrieval(error)
    }
}

impl From<ProjectionError> for CoordinatorError {
    fn from(error: ProjectionError) -> Self {
        Self::Projection(error)
    }
}

impl From<WorkingSetError> for CoordinatorError {
    fn from(error: WorkingSetError) -> Self {
        Self::WorkingSet(error)
    }
}

/// Coordinates providers without owning or mutating the runtime prompt.
pub struct MemoryAgentCoordinator<'a> {
    retrieval: RetrievalCoordinator<'a>,
    renderer: ProjectionRenderer,
    working_set: WorkingSet,
    config: CoordinatorConfig,
    active_run: Option<pi_agent_runtime::RunId>,
    active_turn: Option<Uuid>,
    active_runtime_revision: Option<Revision>,
    active_runtime_memory_revision: Option<MemoryRevision>,
    active_runtime_policy_revision: Option<PolicyRevision>,
    active_memory_epoch: Option<MemoryEpoch>,
    active_policy_epoch: Option<PolicyEpoch>,
}

impl<'a> MemoryAgentCoordinator<'a> {
    pub fn new(
        registry: &'a MemoryRegistry,
        config: CoordinatorConfig,
    ) -> Result<Self, CoordinatorError> {
        if config.proposal_ttl_ms == 0 {
            return Err(CoordinatorError::InvalidConfiguration(
                "proposal_ttl_ms must be positive".to_string(),
            ));
        }
        let renderer = ProjectionRenderer::new(config.max_projection_chars)?;
        let working_set = WorkingSet::new(WorkingSetConfig {
            max_candidates: config.max_working_set_candidates,
        })?;
        Ok(Self {
            retrieval: RetrievalCoordinator::new(registry),
            renderer,
            working_set,
            config,
            active_run: None,
            active_turn: None,
            active_runtime_revision: None,
            active_runtime_memory_revision: None,
            active_runtime_policy_revision: None,
            active_memory_epoch: None,
            active_policy_epoch: None,
        })
    }

    pub fn working_set(&self) -> &WorkingSet {
        &self.working_set
    }

    /// Retrieves, renders, ranks, and prepares a versioned runtime proposal.
    ///
    /// `runtime_context` is the authoritative snapshot for namespace,
    /// runtime revision, policy revision, memory revision, and hook limits.
    /// The memory agent never derives those commit-gate revisions from its
    /// own prediction epochs on this path.
    pub async fn prepare(
        &mut self,
        need: &MemoryNeed,
        state: &TaskState,
        mut operation: OperationContext,
        runtime_context: MemoryHookContext,
    ) -> Result<PreparedMemory, CoordinatorError> {
        if runtime_context.run_id != state.run_id {
            return Err(CoordinatorError::RunMismatch);
        }
        if runtime_context.namespace != state.namespace {
            return Err(CoordinatorError::NamespaceMismatch);
        }
        let now_ms = runtime_context.now_ms;
        if operation.run_id != state.run_id.as_uuid() {
            return Err(CoordinatorError::RunMismatch);
        }
        if operation.namespace != state.namespace || need.query.namespace != state.namespace {
            return Err(CoordinatorError::NamespaceMismatch);
        }
        operation.timeout_ms = match (operation.timeout_ms, state.remaining_deadline_ms) {
            (Some(operation), Some(task)) => Some(operation.min(task)),
            (None, task) => task,
            (operation, None) => operation,
        };
        if self.active_run != Some(state.run_id)
            || self.active_turn != Some(state.turn_id)
            || self.active_runtime_revision != Some(runtime_context.current_revision)
            || self.active_runtime_memory_revision != Some(runtime_context.memory_revision)
            || self.active_runtime_policy_revision != Some(runtime_context.policy_revision)
            || self.active_memory_epoch != Some(state.memory_epoch)
            || self.active_policy_epoch != Some(state.policy_epoch)
        {
            self.working_set = WorkingSet::new(WorkingSetConfig {
                max_candidates: self.config.max_working_set_candidates,
            })?;
            self.active_run = Some(state.run_id);
            self.active_turn = Some(state.turn_id);
            self.active_runtime_revision = Some(runtime_context.current_revision);
            self.active_runtime_memory_revision = Some(runtime_context.memory_revision);
            self.active_runtime_policy_revision = Some(runtime_context.policy_revision);
            self.active_memory_epoch = Some(state.memory_epoch);
            self.active_policy_epoch = Some(state.policy_epoch);
        }

        let report = self.retrieval.retrieve_at(need, operation, now_ms).await?;
        let mut projected_count = 0;
        let mut skipped_count = 0;
        for hit in &report.hits {
            let Some(projection) = self.renderer.render(hit)? else {
                skipped_count += 1;
                continue;
            };
            let provider_score = hit
                .score
                .filter(|score| score.is_finite())
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let score = (0.7 * provider_score + 0.3 * need.estimated_benefit).clamp(0.0, 1.0);
            let expires_at_ms = match (
                need.expires_at_ms,
                self.config.proposal_ttl_ms.checked_add(now_ms),
            ) {
                (Some(need_expiry), Some(ttl_expiry)) => Some(need_expiry.min(ttl_expiry)),
                (Some(need_expiry), None) => Some(need_expiry),
                (None, Some(ttl_expiry)) => Some(ttl_expiry),
                (None, None) => None,
            };
            let candidate = MemoryCandidate {
                candidate_id: Uuid::new_v4(),
                projection,
                score,
                confidence: need.confidence.clamp(0.0, 1.0),
                expires_at_ms,
            };
            self.working_set.upsert(candidate, now_ms)?;
            projected_count += 1;
        }

        let proposal_expiry = Some(now_ms.saturating_add(self.config.proposal_ttl_ms));
        let proposal = self.working_set.proposal(&runtime_context, proposal_expiry);
        Ok(PreparedMemory {
            report,
            projected_count,
            skipped_count,
            proposal,
        })
    }

    /// Builds an adapter context for callers that have not yet threaded the
    /// runtime-owned revision snapshot into this boundary. Production callers
    /// should pass the context created by the runtime hook directly.
    pub fn hook_context_for(&self, state: &TaskState, now_ms: u64) -> MemoryHookContext {
        MemoryHookContext {
            run_id: state.run_id,
            namespace: state.namespace.clone(),
            current_revision: state.runtime_revision,
            memory_revision: pi_agent_runtime::MemoryRevision::new(state.memory_epoch.value()),
            policy_revision: pi_agent_runtime::PolicyRevision::new(state.policy_epoch.value()),
            now_ms,
            max_tokens: state.context_token_budget,
            max_candidates: u32::try_from(self.config.max_working_set_candidates)
                .unwrap_or(u32::MAX),
            max_projection_bytes: u32::try_from(self.config.max_projection_chars.saturating_mul(4))
                .unwrap_or(u32::MAX),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{future::Future, sync::Arc};

    use pi_agent_memory::{
        ContentRef, IngestReceipt, IngestRequest, MemoryCapabilities, MemoryError, MemoryFuture,
        MemoryItem, MemoryKind, MemoryManifest, MemoryMethod, MemoryQuery, MemoryRegistry,
        Modality, OperationContext, RetrieveResponse,
    };
    use pi_agent_runtime::{Revision, RunId};

    use super::*;
    use crate::contracts::{MemoryEpoch, NeedHorizon, PolicyEpoch};

    struct Provider;

    impl MemoryMethod for Provider {
        fn manifest(&self) -> MemoryManifest {
            MemoryManifest {
                method_id: "text".to_string(),
                version: "1".to_string(),
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
        fn ingest<'a>(
            &'a self,
            _: IngestRequest,
            _: OperationContext,
        ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
            Box::pin(std::future::ready(Err(MemoryError::UnsupportedCapability(
                "test".into(),
            ))))
        }
        fn retrieve<'a>(
            &'a self,
            _: MemoryQuery,
            _: OperationContext,
        ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
            Box::pin(std::future::ready(Ok(RetrieveResponse {
                query_id: Uuid::new_v4(),
                hits: vec![pi_agent_memory::MemoryHit {
                    item_id: Uuid::from_u128(1),
                    version: 1,
                    method_id: "text".into(),
                    rank: 0,
                    score: Some(0.9),
                    evidence: vec![],
                    projection: Some(ContentRef::Text {
                        text: "fix parser".into(),
                    }),
                    provenance: "record-1".into(),
                    snapshot_id: Some("s1".into()),
                    latency_ms: 1,
                }],
                snapshot_id: Some("s1".into()),
                latency_ms: 1,
                truncated: false,
            })))
        }
        fn get<'a>(
            &'a self,
            _: Uuid,
            _: OperationContext,
        ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
            Box::pin(std::future::ready(Err(MemoryError::NotFound(Uuid::nil()))))
        }
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
            .block_on(future)
    }

    fn state(run_id: RunId) -> TaskState {
        TaskState {
            run_id,
            turn_id: Uuid::new_v4(),
            namespace: "test".into(),
            runtime_revision: Revision::new(4),
            memory_epoch: MemoryEpoch::new(2),
            policy_epoch: PolicyEpoch::new(3),
            goal_summary: "fix".into(),
            latest_query: Some("parser".into()),
            current_plan: None,
            recent_tools: vec![],
            active_modalities: vec![Modality::Text],
            remaining_deadline_ms: Some(100),
            context_token_budget: 64,
        }
    }

    fn need() -> MemoryNeed {
        MemoryNeed {
            need_id: Uuid::new_v4(),
            query: MemoryQuery {
                input: Some(ContentRef::Text {
                    text: "parser".into(),
                }),
                namespace: "test".into(),
                scope: None,
                subject: None,
                kinds: vec![MemoryKind::Semantic],
                modalities: vec![Modality::Text],
                time_range: None,
                visibility_cutoff_ms: None,
                graph: None,
                budget: Default::default(),
            },
            provider_hints: vec![],
            horizon: NeedHorizon::CurrentTurn,
            confidence: 0.8,
            estimated_benefit: 0.7,
            created_at_ms: 1,
            expires_at_ms: None,
        }
    }

    #[test]
    fn prepare_builds_runtime_proposal_from_provider_hit() {
        let mut registry = MemoryRegistry::new();
        registry.register(Arc::new(Provider)).unwrap();
        let run_id = RunId::new();
        let task = state(run_id);
        let operation = OperationContext {
            namespace: "test".into(),
            run_id: run_id.as_uuid(),
            session_id: None,
            episode_id: None,
            timeout_ms: Some(100),
            idempotency_key: Some("run/need".into()),
        };
        let mut coordinator =
            MemoryAgentCoordinator::new(&registry, CoordinatorConfig::default()).unwrap();
        let runtime_context = coordinator.hook_context_for(&task, 10);
        let prepared =
            block_on(coordinator.prepare(&need(), &task, operation, runtime_context)).unwrap();
        assert_eq!(prepared.projected_count, 1);
        assert_eq!(prepared.proposal.candidates.len(), 1);
        let decision = prepared
            .proposal
            .evaluate(&coordinator.hook_context_for(&task, 10))
            .unwrap();
        assert!(matches!(
            decision,
            pi_agent_runtime::InjectionDecision::Inject { .. }
        ));
    }
}
