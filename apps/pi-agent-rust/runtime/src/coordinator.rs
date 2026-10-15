//! Small executable control loop for side-effecting actions.
//!
//! [`RuntimeCoordinator`] is the first runtime vertical slice. It admits an
//! action, durably records the grant before execution, invokes a caller-owned
//! executor, and records the outcome before advancing the runtime revision.
//! The executor is intentionally a callback: the runtime controls authority
//! and recovery boundaries without depending on a particular tool, model, or
//! memory provider.

use crate::{
    bytes_digest, ActionId, ActionProposal, AdmissionContext, AdmissionError, AdmissionPolicy,
    AdmissionRequest, CommitGrant, EventDraft, EventDraftError, EventKind, EventLedger,
    EventSequence, InjectionDecision, LedgerError, PreparationError, PreparationManager,
    PreparationPlan, PreparationWorkResult, Revision, RunId,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashMap, fmt, future::Future};
use uuid::Uuid;

/// Event emitted after an action receives a durable commit grant.
pub const ACTION_ADMITTED_EVENT: &str = "action_admitted";
/// Event emitted after the caller-owned executor returns successfully.
pub const ACTION_SUCCEEDED_EVENT: &str = "action_succeeded";
/// Event emitted after the caller-owned executor reports a failure.
pub const ACTION_FAILED_EVENT: &str = "action_failed";
/// Event emitted when execution stops before success or failure is known.
pub const ACTION_UNKNOWN_EVENT: &str = "action_unknown";
/// Event emitted when an unresolved action is closed using external evidence.
pub const ACTION_RECONCILED_EVENT: &str = "action_reconciled";
/// Event emitted when a validated memory proposal is injected into a request.
pub const MEMORY_INJECTED_EVENT: &str = "memory_injected";
/// Event emitted when a memory proposal is evaluated but not injected.
pub const MEMORY_SKIPPED_EVENT: &str = "memory_skipped";
/// Event containing a durable snapshot of one preparation plan state.
pub const PREPARATION_UPDATED_EVENT: &str = "preparation_updated";
/// Default bound for speculative preparation plans owned by one run.
pub const DEFAULT_MAX_PREPARATION_PLANS: usize = 64;

/// Outcome of one admitted action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionResult {
    Succeeded {
        grant: CommitGrant,
        output: Value,
        outcome_sequence: EventSequence,
    },
    Failed {
        grant: CommitGrant,
        error: String,
        outcome_sequence: EventSequence,
    },
    Unknown {
        grant: CommitGrant,
        reason: String,
        outcome_sequence: EventSequence,
    },
}

/// External evidence used to close an action whose execution outcome was not
/// available to the original process.
#[derive(Debug, Clone, PartialEq)]
pub enum ReconciliationOutcome {
    /// The provider confirms that the action took effect.
    Succeeded { output: Value, evidence: String },
    /// The provider confirms that the action did not take effect.
    Failed { error: String, evidence: String },
}

/// Result of evaluating a memory proposal at the runtime boundary.
///
/// The decision is returned together with the canonical ledger sequence that
/// records it, so callers can correlate model-request state without parsing
/// the storage backend themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryDecisionReceipt {
    pub decision: InjectionDecision,
    pub event_sequence: EventSequence,
}

struct TerminalEventSpec<'a> {
    success_kind: &'a str,
    failure_kind: &'a str,
    reconciliation: Option<(&'a str, &'a str)>,
}

impl ExecutionResult {
    /// Returns whether the external executor completed successfully.
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Succeeded { .. })
    }

    /// Returns the grant that authorized this execution attempt.
    pub fn grant(&self) -> &CommitGrant {
        match self {
            Self::Succeeded { grant, .. }
            | Self::Failed { grant, .. }
            | Self::Unknown { grant, .. } => grant,
        }
    }
}

/// Runtime-owned coordinator state.
pub struct RuntimeCoordinator<L> {
    ledger: L,
    policy: AdmissionPolicy,
    run_id: RunId,
    current_revision: Revision,
    pending_actions: Vec<CommitGrant>,
    preparations: PreparationManager,
}

impl<L> RuntimeCoordinator<L> {
    /// Creates a coordinator at the initial runtime revision.
    pub fn new(run_id: RunId, policy: AdmissionPolicy, ledger: L) -> Self {
        Self {
            ledger,
            policy,
            run_id,
            current_revision: Revision::INITIAL,
            pending_actions: Vec::new(),
            preparations: PreparationManager::new(DEFAULT_MAX_PREPARATION_PLANS)
                .expect("the default preparation capacity is positive"),
        }
    }

    /// Returns the run identity enforced for every proposal.
    pub const fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Returns the revision used for the next proposal.
    pub const fn current_revision(&self) -> Revision {
        self.current_revision
    }

    /// Returns the policy used by this coordinator.
    pub fn policy(&self) -> &AdmissionPolicy {
        &self.policy
    }

    /// Borrows the underlying ledger for inspection or provider-specific
    /// recovery tooling.
    pub fn ledger(&self) -> &L {
        &self.ledger
    }

    /// Mutably borrows the underlying ledger for controlled maintenance.
    pub fn ledger_mut(&mut self) -> &mut L {
        &mut self.ledger
    }

    /// Creates a coordinator from a caller-verified checkpoint with no pending
    /// actions or preparations. Use [`Self::recover`] to rebuild both from an
    /// existing ledger.
    pub fn from_revision(
        run_id: RunId,
        policy: AdmissionPolicy,
        current_revision: Revision,
        ledger: L,
    ) -> Self {
        Self {
            ledger,
            policy,
            run_id,
            current_revision,
            pending_actions: Vec::new(),
            preparations: PreparationManager::new(DEFAULT_MAX_PREPARATION_PLANS)
                .expect("the default preparation capacity is positive"),
        }
    }

    /// Consumes the coordinator and returns its owned parts.
    pub fn into_parts(self) -> (L, AdmissionPolicy, RunId, Revision) {
        (self.ledger, self.policy, self.run_id, self.current_revision)
    }

    /// Returns grants whose external outcome is not yet known.
    pub fn pending_actions(&self) -> &[CommitGrant] {
        &self.pending_actions
    }

    /// Returns preparation plans owned by this runtime run.
    pub fn preparations(&self) -> &PreparationManager {
        &self.preparations
    }
}

impl<L: EventLedger> RuntimeCoordinator<L> {
    /// Submits a plan after binding it to this run's current revision.
    pub fn submit_preparation(
        &mut self,
        plan: PreparationPlan,
        now_ms: u64,
    ) -> Result<Uuid, CoordinatorError> {
        self.validate_preparation_binding(&plan)?;
        let mut next = self.preparations.clone();
        let plan_id = next
            .submit(plan, now_ms)
            .map_err(CoordinatorError::Preparation)?;
        let snapshot = next
            .get(plan_id)
            .expect("submitted preparation must be present")
            .clone();
        self.append_preparation_update(&snapshot, now_ms)?;
        self.preparations = next;
        Ok(plan_id)
    }

    /// Marks a submitted preparation as started by an external worker.
    pub fn start_preparation(
        &mut self,
        plan_id: Uuid,
        now_ms: u64,
    ) -> Result<(), CoordinatorError> {
        let mut next = self.preparations.clone();
        if let Err(error) = next.start(plan_id, self.run_id, self.current_revision, now_ms) {
            if matches!(&error, PreparationError::Expired(_)) {
                return self.persist_invalidated_preparation(next, plan_id, now_ms, error);
            }
            return Err(CoordinatorError::Preparation(error));
        }
        let snapshot = next
            .get(plan_id)
            .expect("started preparation must be present")
            .clone();
        self.append_preparation_update(&snapshot, now_ms)?;
        self.preparations = next;
        Ok(())
    }

    /// Marks a worker result with no external artifact outputs as ready.
    pub fn mark_preparation_ready(
        &mut self,
        plan_id: Uuid,
        now_ms: u64,
    ) -> Result<(), CoordinatorError> {
        self.mark_preparation_ready_with_artifacts(plan_id, Vec::new(), now_ms)
    }

    /// Marks a worker result as ready and persists its output references.
    pub fn mark_preparation_ready_with_artifacts(
        &mut self,
        plan_id: Uuid,
        artifacts: Vec<crate::ArtifactRef>,
        now_ms: u64,
    ) -> Result<(), CoordinatorError> {
        self.mark_preparation_ready_with_artifacts_and_evidence(
            plan_id,
            artifacts,
            Vec::new(),
            now_ms,
        )
    }

    /// Marks a worker result as ready and persists precise artifact evidence.
    pub fn mark_preparation_ready_with_artifacts_and_evidence(
        &mut self,
        plan_id: Uuid,
        artifacts: Vec<crate::ArtifactRef>,
        evidence: Vec<pi_agent_memory::EvidenceLocator>,
        now_ms: u64,
    ) -> Result<(), CoordinatorError> {
        let mut next = self.preparations.clone();
        if let Err(error) = next.mark_ready_with_artifacts_and_evidence(
            plan_id,
            self.run_id,
            self.current_revision,
            now_ms,
            artifacts,
            evidence,
        ) {
            if matches!(&error, PreparationError::Expired(_)) {
                return self.persist_invalidated_preparation(next, plan_id, now_ms, error);
            }
            return Err(CoordinatorError::Preparation(error));
        }
        let snapshot = next
            .get(plan_id)
            .expect("ready preparation must be present")
            .clone();
        self.append_preparation_update(&snapshot, now_ms)?;
        self.preparations = next;
        Ok(())
    }

    /// Runs one preparation worker after its start event is durable.
    ///
    /// Worker errors are recorded as a failed preparation, just like action
    /// execution errors are recorded as failed actions. A caller that loses
    /// contact with the worker can return `Unknown` instead.
    pub fn execute_preparation<F, E>(
        &mut self,
        plan_id: Uuid,
        started_at_ms: u64,
        completed_at_ms: u64,
        worker: F,
    ) -> Result<PreparationWorkResult, CoordinatorError>
    where
        F: FnOnce(&PreparationPlan) -> Result<PreparationWorkResult, E>,
        E: fmt::Display,
    {
        self.start_preparation(plan_id, started_at_ms)?;
        let plan = self
            .preparations
            .get(plan_id)
            .expect("started preparation must be present")
            .clone();
        let result = worker(&plan).unwrap_or_else(|error| PreparationWorkResult::Failed {
            error: error.to_string(),
        });
        self.finish_preparation(plan_id, completed_at_ms, result)
    }

    /// Async equivalent of [`Self::execute_preparation`] for networked memory
    /// providers and other concurrent preparation workers.
    pub async fn execute_preparation_async<F, Fut, E>(
        &mut self,
        plan_id: Uuid,
        started_at_ms: u64,
        completed_at_ms: u64,
        worker: F,
    ) -> Result<PreparationWorkResult, CoordinatorError>
    where
        F: FnOnce(&PreparationPlan) -> Fut,
        Fut: Future<Output = Result<PreparationWorkResult, E>>,
        E: fmt::Display,
    {
        self.start_preparation(plan_id, started_at_ms)?;
        let plan = self
            .preparations
            .get(plan_id)
            .expect("started preparation must be present")
            .clone();
        let result = worker(&plan)
            .await
            .unwrap_or_else(|error| PreparationWorkResult::Failed {
                error: error.to_string(),
            });
        self.finish_preparation(plan_id, completed_at_ms, result)
    }

    /// Consumes a prepared result only when the final action still matches it.
    pub fn consume_preparation(
        &mut self,
        plan_id: Uuid,
        proposal: &ActionProposal,
        now_ms: u64,
    ) -> Result<PreparationPlan, CoordinatorError> {
        self.consume_preparation_checked(plan_id, proposal, now_ms, |plan| {
            if plan.artifacts.is_empty() {
                Ok(())
            } else {
                Err(CoordinatorError::ArtifactVerificationRequired(plan.plan_id))
            }
        })
    }

    /// Consumes a preparation after checking every output reference through a
    /// caller-owned artifact backend.
    pub fn consume_preparation_with_resolver<F, E>(
        &mut self,
        plan_id: Uuid,
        proposal: &ActionProposal,
        now_ms: u64,
        resolver: F,
    ) -> Result<PreparationPlan, CoordinatorError>
    where
        F: Fn(&crate::ArtifactRef) -> Result<crate::ArtifactRef, E>,
        E: fmt::Display,
    {
        self.consume_preparation_checked(plan_id, proposal, now_ms, |plan| {
            for artifact in &plan.artifacts {
                let resolved =
                    resolver(artifact).map_err(|error| CoordinatorError::ArtifactResolution {
                        artifact_id: artifact.artifact_id,
                        error: error.to_string(),
                    })?;
                if resolved != *artifact {
                    return Err(CoordinatorError::ArtifactMetadataMismatch {
                        artifact_id: artifact.artifact_id,
                    });
                }
            }
            Ok(())
        })
    }

    fn consume_preparation_checked<F>(
        &mut self,
        plan_id: Uuid,
        proposal: &ActionProposal,
        now_ms: u64,
        verify: F,
    ) -> Result<PreparationPlan, CoordinatorError>
    where
        F: FnOnce(&PreparationPlan) -> Result<(), CoordinatorError>,
    {
        let mut next = self.preparations.clone();
        let consumed = match next.consume_ready(plan_id, proposal, now_ms) {
            Ok(consumed) => consumed,
            Err(error) => {
                if matches!(&error, PreparationError::Expired(_)) {
                    self.persist_invalidated_preparation(next, plan_id, now_ms, error.clone())?;
                }
                return Err(CoordinatorError::Preparation(error));
            }
        };
        verify(&consumed)?;
        self.append_preparation_update(&consumed, now_ms)?;
        self.preparations = next;
        Ok(consumed)
    }

    /// Releases a preparation without granting any external commit authority.
    pub fn release_preparation(
        &mut self,
        plan_id: Uuid,
        now_ms: u64,
    ) -> Result<(), CoordinatorError> {
        let mut next = self.preparations.clone();
        next.release(plan_id)
            .map_err(CoordinatorError::Preparation)?;
        let snapshot = next
            .get(plan_id)
            .expect("released preparation must be present")
            .clone();
        self.append_preparation_update(&snapshot, now_ms)?;
        self.preparations = next;
        Ok(())
    }

    /// Records that an external preparation worker failed.
    pub fn fail_preparation(&mut self, plan_id: Uuid, now_ms: u64) -> Result<(), CoordinatorError> {
        self.fail_preparation_with_reason(plan_id, "preparation worker failed", now_ms)
    }

    /// Records a known worker failure and preserves its reason for recovery.
    pub fn fail_preparation_with_reason(
        &mut self,
        plan_id: Uuid,
        error: impl Into<String>,
        now_ms: u64,
    ) -> Result<(), CoordinatorError> {
        let error = error.into();
        let mut next = self.preparations.clone();
        next.mark_failed(plan_id, error)
            .map_err(CoordinatorError::Preparation)?;
        let snapshot = next
            .get(plan_id)
            .expect("failed preparation must be present")
            .clone();
        self.append_preparation_update(&snapshot, now_ms)?;
        self.preparations = next;
        Ok(())
    }

    /// Records that the preparation worker outcome is not known.
    pub fn mark_preparation_unknown(
        &mut self,
        plan_id: Uuid,
        now_ms: u64,
    ) -> Result<(), CoordinatorError> {
        self.mark_preparation_unknown_with_reason(
            plan_id,
            "preparation worker outcome unknown",
            now_ms,
        )
    }

    /// Records that a worker stopped without establishing whether preparation
    /// completed, preserving the reason for reconciliation after recovery.
    pub fn mark_preparation_unknown_with_reason(
        &mut self,
        plan_id: Uuid,
        reason: impl Into<String>,
        now_ms: u64,
    ) -> Result<(), CoordinatorError> {
        let reason = reason.into();
        let mut next = self.preparations.clone();
        next.mark_unknown(plan_id, reason)
            .map_err(CoordinatorError::Preparation)?;
        let snapshot = next
            .get(plan_id)
            .expect("unknown preparation must be present")
            .clone();
        self.append_preparation_update(&snapshot, now_ms)?;
        self.preparations = next;
        Ok(())
    }

    fn finish_preparation(
        &mut self,
        plan_id: Uuid,
        now_ms: u64,
        result: PreparationWorkResult,
    ) -> Result<PreparationWorkResult, CoordinatorError> {
        match &result {
            PreparationWorkResult::Ready {
                artifacts,
                evidence,
            } => self.mark_preparation_ready_with_artifacts_and_evidence(
                plan_id,
                artifacts.clone(),
                evidence.clone(),
                now_ms,
            )?,
            PreparationWorkResult::Failed { error } => {
                self.fail_preparation_with_reason(plan_id, error.clone(), now_ms)?
            }
            PreparationWorkResult::Unknown { reason } => {
                self.mark_preparation_unknown_with_reason(plan_id, reason.clone(), now_ms)?
            }
        }
        Ok(result)
    }

    fn persist_invalidated_preparation(
        &mut self,
        next: PreparationManager,
        plan_id: Uuid,
        now_ms: u64,
        error: PreparationError,
    ) -> Result<(), CoordinatorError> {
        let snapshot = next
            .get(plan_id)
            .expect("expired preparation must remain present")
            .clone();
        self.append_preparation_update(&snapshot, now_ms)?;
        self.preparations = next;
        Err(CoordinatorError::Preparation(error))
    }

    /// Rebuilds the coordinator revision from the events belonging to a run.
    ///
    /// An admission event keeps the current revision, while a terminal
    /// outcome records the next revision. Incomplete actions, including
    /// side-effecting actions with unknown outcomes, remain pending and must
    /// be reconciled before admitting more work in this run.
    pub fn recover(
        run_id: RunId,
        policy: AdmissionPolicy,
        ledger: L,
    ) -> Result<Self, CoordinatorError> {
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .map_err(CoordinatorError::Ledger)?
            .into_iter()
            .filter(|event| event.draft.run_id == run_id)
            .collect::<Vec<_>>();
        let state = replay_run(run_id, events)?;
        let mut coordinator = Self::from_revision(run_id, policy, state.revision, ledger);
        coordinator.pending_actions = state.pending_actions;
        for plan in state.preparations {
            coordinator
                .preparations
                .restore(plan)
                .map_err(CoordinatorError::Preparation)?;
        }
        Ok(coordinator)
    }

    /// Recovers the latest run represented by a ledger.
    ///
    /// This is the restart-friendly form of [`Self::recover`]. It selects the
    /// only run with unresolved actions. If several runs are unresolved, the
    /// runtime refuses to guess which external side effect should be
    /// reconciled. When every run is complete, the run with the latest event
    /// is returned so its revision can continue monotonically.
    pub fn recover_latest(
        policy: AdmissionPolicy,
        ledger: L,
    ) -> Result<Option<Self>, CoordinatorError> {
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .map_err(CoordinatorError::Ledger)?;
        if events.is_empty() {
            return Ok(None);
        }

        let mut by_run: HashMap<RunId, Vec<crate::ledger::LedgerEvent>> = HashMap::new();
        for event in events {
            by_run.entry(event.draft.run_id).or_default().push(event);
        }

        let mut states = Vec::with_capacity(by_run.len());
        for (run_id, run_events) in by_run {
            states.push((run_id, replay_run(run_id, run_events)?));
        }

        let mut unresolved: Vec<_> = states
            .iter()
            .filter(|(_, state)| !state.pending_actions.is_empty())
            .map(|(run_id, _)| *run_id)
            .collect();
        unresolved.sort_by_key(|run_id| run_id.as_uuid());
        if unresolved.len() > 1 {
            return Err(CoordinatorError::AmbiguousRecovery { runs: unresolved });
        }

        let selected_run = unresolved.first().copied().or_else(|| {
            states
                .iter()
                .max_by_key(|(_, state)| state.last_sequence)
                .map(|(run_id, _)| *run_id)
        });
        let Some(selected_run) = selected_run else {
            return Ok(None);
        };
        let (_, state) = states
            .into_iter()
            .find(|(run_id, _)| *run_id == selected_run)
            .expect("selected recovery run must have a replay state");
        let mut coordinator = Self::from_revision(selected_run, policy, state.revision, ledger);
        coordinator.pending_actions = state.pending_actions;
        for plan in state.preparations {
            coordinator
                .preparations
                .restore(plan)
                .map_err(CoordinatorError::Preparation)?;
        }
        Ok(Some(coordinator))
    }

    /// Admits a proposal and records the grant before any external call.
    pub fn admit(
        &mut self,
        request: AdmissionRequest,
        now_ms: u64,
    ) -> Result<CommitGrant, CoordinatorError> {
        if request.proposal.run_id != self.run_id {
            return Err(CoordinatorError::RunMismatch {
                expected: self.run_id,
                actual: request.proposal.run_id,
            });
        }

        if let Some(pending) = self.pending_actions.first() {
            return Err(CoordinatorError::PendingAction {
                action_id: pending.action_id,
                tool_name: pending.tool_name.clone(),
            });
        }
        // Reserve room for the outcome before authorizing any external call.
        self.next_revision()?;

        let grant = self.policy.admit(
            request,
            AdmissionContext {
                current_revision: self.current_revision,
                now_ms,
            },
        )?;

        let payload = serde_json::to_value(&grant).map_err(CoordinatorError::Serialize)?;
        self.append_and_flush(
            ACTION_ADMITTED_EVENT,
            self.current_revision,
            now_ms,
            payload,
            Some(grant.idempotency_key.clone()),
        )?;
        self.pending_actions.push(grant.clone());

        Ok(grant)
    }

    /// Executes one admitted proposal through a caller-owned side-effect
    /// boundary and records either success or failure before advancing state.
    pub fn execute<F, E>(
        &mut self,
        request: AdmissionRequest,
        now_ms: u64,
        executor: F,
    ) -> Result<ExecutionResult, CoordinatorError>
    where
        F: FnOnce(&CommitGrant, &ActionProposal) -> Result<Value, E>,
        E: fmt::Display,
    {
        let next_revision = self
            .current_revision
            .checked_next()
            .ok_or(CoordinatorError::RevisionExhausted)?;
        let proposal = request.proposal.clone();
        let grant = self.admit(request, now_ms)?;
        let execution = executor(&grant, &proposal).map_err(|error| error.to_string());
        self.finish_execution(grant, next_revision, now_ms, execution)
    }

    /// Async equivalent of [`Self::execute`] for networked or concurrent
    /// tool and memory providers.
    pub async fn execute_async<F, Fut, E>(
        &mut self,
        request: AdmissionRequest,
        now_ms: u64,
        executor: F,
    ) -> Result<ExecutionResult, CoordinatorError>
    where
        F: FnOnce(&CommitGrant, &ActionProposal) -> Fut,
        Fut: Future<Output = Result<Value, E>>,
        E: fmt::Display,
    {
        let next_revision = self
            .current_revision
            .checked_next()
            .ok_or(CoordinatorError::RevisionExhausted)?;
        let proposal = request.proposal.clone();
        let grant = self.admit(request, now_ms)?;
        let execution = executor(&grant, &proposal)
            .await
            .map_err(|error| error.to_string());
        self.finish_execution(grant, next_revision, now_ms, execution)
    }

    /// Records a successful result for a grant returned by [`Self::admit`].
    pub fn complete(
        &mut self,
        grant: CommitGrant,
        output: Value,
        now_ms: u64,
    ) -> Result<ExecutionResult, CoordinatorError> {
        let next_revision = self.next_revision()?;
        self.finish_execution(grant, next_revision, now_ms, Ok(output))
    }

    /// Records a known executor failure for a grant returned by [`Self::admit`].
    pub fn fail(
        &mut self,
        grant: CommitGrant,
        error: impl Into<String>,
        now_ms: u64,
    ) -> Result<ExecutionResult, CoordinatorError> {
        let next_revision = self.next_revision()?;
        self.finish_execution(grant, next_revision, now_ms, Err(error.into()))
    }

    /// Records that execution stopped before its external outcome was known.
    ///
    /// This is used for cancellation, timeout, or process handoff. The
    /// runtime deliberately does not guess whether the provider applied the
    /// side effect; callers must reconcile the action before retrying it.
    pub fn mark_unknown(
        &mut self,
        grant: CommitGrant,
        reason: impl Into<String>,
        now_ms: u64,
    ) -> Result<ExecutionResult, CoordinatorError> {
        self.validate_pending(&grant)?;
        let next_revision = self.next_revision()?;
        let reason = reason.into();
        let outcome_sequence = self.append_and_flush(
            ACTION_UNKNOWN_EVENT,
            next_revision,
            now_ms,
            json!({
                "grant_id": grant.grant_id,
                "action_id": grant.action_id,
                "tool_name": grant.tool_name,
                "status": "unknown",
                "reason": reason,
            }),
            Some(format!(
                "action:{}:{}",
                grant.grant_id, ACTION_UNKNOWN_EVENT
            )),
        )?;
        // Keep the grant pending until a caller reconciles the external
        // outcome. Even a declared read can hide provider-specific effects.
        self.advance_revision(next_revision, now_ms);

        Ok(ExecutionResult::Unknown {
            grant,
            reason,
            outcome_sequence,
        })
    }

    /// Closes a pending action after an external system has established its
    /// outcome. The evidence is persisted so a later recovery can distinguish
    /// reconciliation from the original executor result.
    pub fn reconcile(
        &mut self,
        grant: CommitGrant,
        outcome: ReconciliationOutcome,
        now_ms: u64,
    ) -> Result<ExecutionResult, CoordinatorError> {
        self.validate_pending(&grant)?;
        let evidence = match &outcome {
            ReconciliationOutcome::Succeeded { evidence, .. }
            | ReconciliationOutcome::Failed { evidence, .. } => evidence,
        };
        if evidence.trim().is_empty() {
            return Err(CoordinatorError::EmptyReconciliationEvidence);
        }

        let next_revision = self.next_revision()?;
        match outcome {
            ReconciliationOutcome::Succeeded { output, evidence } => self
                .finish_execution_with_events(
                    grant,
                    next_revision,
                    now_ms,
                    Ok(output),
                    TerminalEventSpec {
                        success_kind: ACTION_RECONCILED_EVENT,
                        failure_kind: ACTION_RECONCILED_EVENT,
                        reconciliation: Some((&evidence, "succeeded")),
                    },
                ),
            ReconciliationOutcome::Failed { error, evidence } => self.finish_execution_with_events(
                grant,
                next_revision,
                now_ms,
                Err(error),
                TerminalEventSpec {
                    success_kind: ACTION_RECONCILED_EVENT,
                    failure_kind: ACTION_RECONCILED_EVENT,
                    reconciliation: Some((&evidence, "failed")),
                },
            ),
        }
    }

    /// Persists the last-mile decision for a memory proposal.
    ///
    /// Memory decisions are observational: they use the current action
    /// revision but do not advance it. A later tool action still has to pass
    /// the normal admission path, while benchmark consumers can correlate
    /// retrieval and injection behavior from one ledger.
    pub fn record_memory_decision(
        &mut self,
        context: &crate::MemoryHookContext,
        decision: &InjectionDecision,
        now_ms: u64,
    ) -> Result<EventSequence, CoordinatorError> {
        self.record_memory_decision_with_summary(context, decision, now_ms, None)
    }

    /// Persists a memory decision together with provider-neutral retrieval
    /// facts supplied by the memory agent.
    pub fn record_memory_decision_with_summary(
        &mut self,
        context: &crate::MemoryHookContext,
        decision: &InjectionDecision,
        now_ms: u64,
        retrieval_summary: Option<&crate::MemoryRetrievalSummary>,
    ) -> Result<EventSequence, CoordinatorError> {
        if context.run_id != self.run_id {
            return Err(CoordinatorError::RunMismatch {
                expected: self.run_id,
                actual: context.run_id,
            });
        }
        if context.current_revision != self.current_revision {
            return Err(CoordinatorError::RevisionConflict {
                expected: self.current_revision,
                actual: context.current_revision,
            });
        }

        let (kind, status, mut details, proposal_id) = match decision {
            InjectionDecision::Inject {
                proposal_id,
                injections,
                total_token_cost,
            } => (
                MEMORY_INJECTED_EVENT,
                "injected",
                json!({
                    "proposal_id": proposal_id,
                    "injections": injections
                        .iter()
                        .map(|injection| {
                            json!({
                                "candidate_id": injection.candidate_id,
                                "projection_id": injection.projection.projection_id,
                                "source_memory_id": injection.projection.source_memory_id,
                                "source_version": injection.projection.source_version,
                                "token_cost": injection.projection.token_cost,
                                "provenance": injection.projection.provenance,
                                "evidence_refs": injection.projection.evidence_refs,
                                "content_digest": bytes_digest(
                                    injection.projection.content.as_bytes()
                                ),
                            })
                        })
                        .collect::<Vec<_>>(),
                    "total_token_cost": total_token_cost,
                }),
                *proposal_id,
            ),
            InjectionDecision::Skip {
                proposal_id,
                reason,
            } => (
                MEMORY_SKIPPED_EVENT,
                "skipped",
                json!({
                    "proposal_id": proposal_id,
                    "reason": reason,
                }),
                *proposal_id,
            ),
        };
        if let Some(summary) = retrieval_summary {
            details["retrieval_summary"] =
                serde_json::to_value(summary).expect("memory retrieval summary is serializable");
        }
        let payload = json!({
            "status": status,
            "run_id": context.run_id,
            "turn_id": context.turn_id,
            "namespace": context.namespace,
            "runtime_revision": context.current_revision,
            "memory_revision": context.memory_revision,
            "policy_revision": context.policy_revision,
            "details": details,
        });
        let idempotency_key = format!(
            "memory:{}:run-{}:runtime-{}:memory-{}:policy-{}",
            proposal_id,
            context.run_id.as_uuid(),
            context.current_revision.value(),
            context.memory_revision.value(),
            context.policy_revision.value(),
        );
        self.append_and_flush(
            kind,
            self.current_revision,
            now_ms,
            payload,
            Some(idempotency_key),
        )
    }

    /// Evaluates and audits one memory proposal at the runtime boundary.
    ///
    /// Keeping these operations together prevents a caller from consuming a
    /// proposal without leaving the corresponding benchmark/audit event.
    pub fn evaluate_memory_proposal(
        &mut self,
        proposal: &crate::MemoryProposal,
        context: &crate::MemoryHookContext,
        now_ms: u64,
    ) -> Result<InjectionDecision, CoordinatorError> {
        Ok(self
            .evaluate_memory_proposal_with_receipt(proposal, context, now_ms)?
            .decision)
    }

    /// Evaluates and audits a memory proposal, returning its ledger receipt.
    pub fn evaluate_memory_proposal_with_receipt(
        &mut self,
        proposal: &crate::MemoryProposal,
        context: &crate::MemoryHookContext,
        now_ms: u64,
    ) -> Result<MemoryDecisionReceipt, CoordinatorError> {
        let decision = proposal
            .evaluate(context)
            .map_err(CoordinatorError::MemoryProposal)?;
        let event_sequence = self.record_memory_decision_with_summary(
            context,
            &decision,
            now_ms,
            proposal.retrieval_summary.as_ref(),
        )?;
        Ok(MemoryDecisionReceipt {
            decision,
            event_sequence,
        })
    }

    fn next_revision(&self) -> Result<Revision, CoordinatorError> {
        self.current_revision
            .checked_next()
            .ok_or(CoordinatorError::RevisionExhausted)
    }

    fn validate_pending(&self, grant: &CommitGrant) -> Result<(), CoordinatorError> {
        if grant.run_id != self.run_id {
            return Err(CoordinatorError::RunMismatch {
                expected: self.run_id,
                actual: grant.run_id,
            });
        }
        if !self.pending_actions.contains(grant) {
            return Err(CoordinatorError::UnrecognizedGrant(grant.action_id));
        }
        Ok(())
    }

    fn remove_pending(&mut self, action_id: ActionId) {
        self.pending_actions
            .retain(|pending| pending.action_id != action_id);
    }

    fn finish_execution(
        &mut self,
        grant: CommitGrant,
        next_revision: Revision,
        now_ms: u64,
        execution: Result<Value, String>,
    ) -> Result<ExecutionResult, CoordinatorError> {
        self.finish_execution_with_events(
            grant,
            next_revision,
            now_ms,
            execution,
            TerminalEventSpec {
                success_kind: ACTION_SUCCEEDED_EVENT,
                failure_kind: ACTION_FAILED_EVENT,
                reconciliation: None,
            },
        )
    }

    fn finish_execution_with_events(
        &mut self,
        grant: CommitGrant,
        next_revision: Revision,
        now_ms: u64,
        execution: Result<Value, String>,
        events: TerminalEventSpec<'_>,
    ) -> Result<ExecutionResult, CoordinatorError> {
        self.validate_pending(&grant)?;
        match execution {
            Ok(output) => {
                let payload = match events.reconciliation {
                    Some((evidence, outcome)) => json!({
                        "grant_id": grant.grant_id,
                        "action_id": grant.action_id,
                        "tool_name": grant.tool_name,
                        "status": "reconciled",
                        "outcome": outcome,
                        "evidence": evidence,
                        "output": output.clone(),
                    }),
                    None => json!({
                        "grant_id": grant.grant_id,
                        "action_id": grant.action_id,
                        "tool_name": grant.tool_name,
                        "status": "succeeded",
                        "output": output.clone(),
                    }),
                };
                let outcome_sequence = self.append_and_flush(
                    events.success_kind,
                    next_revision,
                    now_ms,
                    payload,
                    Some(format!("action:{}:{}", grant.grant_id, events.success_kind)),
                )?;
                self.remove_pending(grant.action_id);
                self.advance_revision(next_revision, now_ms);

                Ok(ExecutionResult::Succeeded {
                    grant,
                    output,
                    outcome_sequence,
                })
            }
            Err(error) => {
                let payload = match events.reconciliation {
                    Some((evidence, outcome)) => json!({
                        "grant_id": grant.grant_id,
                        "action_id": grant.action_id,
                        "tool_name": grant.tool_name,
                        "status": "reconciled",
                        "outcome": outcome,
                        "evidence": evidence,
                        "error": error,
                    }),
                    None => json!({
                        "grant_id": grant.grant_id,
                        "action_id": grant.action_id,
                        "tool_name": grant.tool_name,
                        "status": "failed",
                        "error": error,
                    }),
                };
                let outcome_sequence = self.append_and_flush(
                    events.failure_kind,
                    next_revision,
                    now_ms,
                    payload,
                    Some(format!("action:{}:{}", grant.grant_id, events.failure_kind)),
                )?;
                self.remove_pending(grant.action_id);
                self.advance_revision(next_revision, now_ms);

                Ok(ExecutionResult::Failed {
                    grant,
                    error,
                    outcome_sequence,
                })
            }
        }
    }

    /// Closes the underlying ledger after all events have been flushed.
    pub fn close(&mut self) -> Result<(), CoordinatorError> {
        self.ledger.close().map_err(CoordinatorError::Ledger)
    }

    fn append_and_flush(
        &mut self,
        kind: &str,
        revision: Revision,
        now_ms: u64,
        payload: Value,
        idempotency_key: Option<String>,
    ) -> Result<EventSequence, CoordinatorError> {
        let kind = EventKind::new(kind).map_err(CoordinatorError::Draft)?;
        let draft = EventDraft::new(self.run_id, revision, kind, payload, now_ms);
        let draft = match idempotency_key {
            Some(key) => draft
                .with_idempotency_key(key)
                .map_err(CoordinatorError::Draft)?,
            None => draft,
        };
        let receipt = self
            .ledger
            .append(vec![draft])
            .map_err(CoordinatorError::Ledger)?;
        self.ledger
            .flush(receipt.last_sequence)
            .map_err(CoordinatorError::Ledger)?;
        Ok(receipt.last_sequence)
    }

    fn validate_preparation_binding(&self, plan: &PreparationPlan) -> Result<(), CoordinatorError> {
        if plan.run_id != self.run_id {
            return Err(CoordinatorError::RunMismatch {
                expected: self.run_id,
                actual: plan.run_id,
            });
        }
        if plan.base_revision != self.current_revision {
            return Err(CoordinatorError::RevisionConflict {
                expected: self.current_revision,
                actual: plan.base_revision,
            });
        }
        Ok(())
    }

    fn append_preparation_update(
        &mut self,
        plan: &PreparationPlan,
        now_ms: u64,
    ) -> Result<EventSequence, CoordinatorError> {
        let payload = json!({
            "plan": plan,
            "status": preparation_status_name(plan.status),
        });
        self.append_and_flush(
            PREPARATION_UPDATED_EVENT,
            self.current_revision,
            now_ms,
            payload,
            Some(format!(
                "preparation:{}:{}",
                plan.plan_id,
                preparation_status_name(plan.status)
            )),
        )
    }

    fn advance_revision(&mut self, next_revision: Revision, now_ms: u64) {
        self.current_revision = next_revision;
        self.preparations
            .invalidate_stale(self.run_id, next_revision, now_ms);
    }
}

fn preparation_status_name(status: crate::PreparationStatus) -> &'static str {
    match status {
        crate::PreparationStatus::Planned => "planned",
        crate::PreparationStatus::Starting => "starting",
        crate::PreparationStatus::Ready => "ready",
        crate::PreparationStatus::Consumed => "consumed",
        crate::PreparationStatus::Invalidated => "invalidated",
        crate::PreparationStatus::Released => "released",
        crate::PreparationStatus::Failed => "failed",
        crate::PreparationStatus::Unknown => "unknown",
    }
}

#[derive(Deserialize)]
struct OutcomeIdentity {
    grant_id: uuid::Uuid,
    action_id: ActionId,
}

#[derive(Default)]
struct ReplayState {
    revision: Revision,
    last_sequence: EventSequence,
    pending_actions: Vec<CommitGrant>,
    preparations: Vec<PreparationPlan>,
}

fn replay_run(
    run_id: RunId,
    events: impl IntoIterator<Item = crate::ledger::LedgerEvent>,
) -> Result<ReplayState, CoordinatorError> {
    let mut state = ReplayState::default();
    for event in events {
        state.last_sequence = state.last_sequence.max(event.sequence);
        state.revision = state.revision.max(event.draft.revision);
        match event.draft.kind.as_str() {
            ACTION_ADMITTED_EVENT => {
                let grant: CommitGrant =
                    serde_json::from_value(event.draft.payload).map_err(|source| {
                        CoordinatorError::Deserialize {
                            kind: ACTION_ADMITTED_EVENT.to_owned(),
                            source,
                        }
                    })?;
                if grant.run_id != run_id {
                    return Err(CoordinatorError::RunMismatch {
                        expected: run_id,
                        actual: grant.run_id,
                    });
                }
                state.pending_actions.push(grant);
            }
            ACTION_SUCCEEDED_EVENT
            | ACTION_FAILED_EVENT
            | ACTION_UNKNOWN_EVENT
            | ACTION_RECONCILED_EVENT => {
                let kind = event.draft.kind.as_str().to_owned();
                let outcome: OutcomeIdentity = serde_json::from_value(event.draft.payload)
                    .map_err(|source| CoordinatorError::Deserialize { kind, source })?;
                let index = state
                    .pending_actions
                    .iter()
                    .position(|grant| {
                        grant.action_id == outcome.action_id && grant.grant_id == outcome.grant_id
                    })
                    .ok_or(CoordinatorError::UnrecognizedGrant(outcome.action_id))?;
                if event.draft.kind.as_str() != ACTION_UNKNOWN_EVENT {
                    state.pending_actions.remove(index);
                }
            }
            PREPARATION_UPDATED_EVENT => {
                let payload = event.draft.payload;
                let plan: PreparationPlan =
                    serde_json::from_value(payload.get("plan").cloned().unwrap_or(Value::Null))
                        .map_err(|source| CoordinatorError::Deserialize {
                            kind: PREPARATION_UPDATED_EVENT.to_owned(),
                            source,
                        })?;
                if plan.run_id != run_id {
                    return Err(CoordinatorError::RunMismatch {
                        expected: run_id,
                        actual: plan.run_id,
                    });
                }
                if let Some(existing) = state
                    .preparations
                    .iter_mut()
                    .find(|existing| existing.plan_id == plan.plan_id)
                {
                    *existing = plan;
                } else {
                    state.preparations.push(plan);
                }
            }
            _ => {}
        }
    }
    for plan in &mut state.preparations {
        if plan.base_revision != state.revision
            && !matches!(
                plan.status,
                crate::PreparationStatus::Consumed
                    | crate::PreparationStatus::Invalidated
                    | crate::PreparationStatus::Released
                    | crate::PreparationStatus::Failed
                    | crate::PreparationStatus::Unknown
            )
        {
            plan.status = crate::PreparationStatus::Invalidated;
        }
    }
    Ok(state)
}

/// Errors raised by the coordinator before an execution result is available.
#[derive(Debug)]
pub enum CoordinatorError {
    RunMismatch {
        expected: RunId,
        actual: RunId,
    },
    PendingAction {
        action_id: ActionId,
        tool_name: String,
    },
    AmbiguousRecovery {
        runs: Vec<RunId>,
    },
    UnrecognizedGrant(ActionId),
    RevisionConflict {
        expected: Revision,
        actual: Revision,
    },
    ArtifactVerificationRequired(Uuid),
    ArtifactResolution {
        artifact_id: Uuid,
        error: String,
    },
    ArtifactMetadataMismatch {
        artifact_id: Uuid,
    },
    MemoryProposal(crate::MemoryProposalError),
    Preparation(PreparationError),
    Admission(AdmissionError),
    Draft(EventDraftError),
    Ledger(LedgerError),
    Serialize(serde_json::Error),
    Deserialize {
        kind: String,
        source: serde_json::Error,
    },
    RevisionExhausted,
    EmptyReconciliationEvidence,
}

impl fmt::Display for CoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RunMismatch { expected, actual } => {
                write!(
                    formatter,
                    "proposal run {} does not match coordinator run {}",
                    actual.as_uuid(),
                    expected.as_uuid()
                )
            }
            Self::PendingAction {
                action_id,
                tool_name,
            } => write!(
                formatter,
                "action {} for tool {tool_name} requires reconciliation before admitting more work",
                action_id.as_uuid()
            ),
            Self::AmbiguousRecovery { runs } => write!(
                formatter,
                "cannot recover automatically: {} runs have unresolved actions ({})",
                runs.len(),
                runs.iter()
                    .map(|run| run.as_uuid().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::UnrecognizedGrant(action_id) => write!(
                formatter,
                "grant for action {} does not match a pending action",
                action_id.as_uuid()
            ),
            Self::RevisionConflict { expected, actual } => write!(
                formatter,
                "runtime revision conflict: expected {}, received {}",
                expected.value(),
                actual.value()
            ),
            Self::ArtifactVerificationRequired(plan_id) => write!(
                formatter,
                "preparation {plan_id} has artifact outputs; consume it with an artifact resolver"
            ),
            Self::ArtifactResolution { artifact_id, error } => write!(
                formatter,
                "artifact {artifact_id} could not be resolved: {error}"
            ),
            Self::ArtifactMetadataMismatch { artifact_id } => write!(
                formatter,
                "resolved artifact {artifact_id} metadata does not match the prepared reference"
            ),
            Self::MemoryProposal(error) => error.fmt(formatter),
            Self::Preparation(error) => error.fmt(formatter),
            Self::Admission(error) => error.fmt(formatter),
            Self::Draft(error) => error.fmt(formatter),
            Self::Ledger(error) => error.fmt(formatter),
            Self::Serialize(error) => {
                write!(formatter, "failed to serialize runtime event: {error}")
            }
            Self::Deserialize { kind, source } => {
                write!(formatter, "invalid {kind} runtime event: {source}")
            }
            Self::RevisionExhausted => formatter.write_str("runtime revision is exhausted"),
            Self::EmptyReconciliationEvidence => {
                formatter.write_str("reconciliation evidence must not be empty")
            }
        }
    }
}

impl std::error::Error for CoordinatorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(error) => Some(error),
            Self::Draft(error) => Some(error),
            Self::Ledger(error) => Some(error),
            Self::Serialize(error) => Some(error),
            Self::Deserialize { source, .. } => Some(source),
            Self::MemoryProposal(error) => Some(error),
            Self::Preparation(error) => Some(error),
            Self::RunMismatch { .. }
            | Self::PendingAction { .. }
            | Self::AmbiguousRecovery { .. }
            | Self::UnrecognizedGrant(_)
            | Self::RevisionConflict { .. }
            | Self::ArtifactVerificationRequired(_)
            | Self::ArtifactResolution { .. }
            | Self::ArtifactMetadataMismatch { .. }
            | Self::RevisionExhausted
            | Self::EmptyReconciliationEvidence => None,
        }
    }
}

impl From<AdmissionError> for CoordinatorError {
    fn from(error: AdmissionError) -> Self {
        Self::Admission(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Budget, InMemoryLedger, JsonlLedger, ResourceScope, RiskClass};

    fn request(run_id: RunId) -> AdmissionRequest {
        AdmissionRequest {
            proposal: ActionProposal {
                run_id,
                action_id: crate::ActionId::new(),
                base_revision: Revision::INITIAL,
                tool_name: "memory.search".to_owned(),
                arguments: json!({"query": "rust"}),
                arguments_digest: crate::canonical_json_digest(&json!({"query": "rust"})),
                risk: RiskClass::ReadOnly,
                idempotency_key: "run/action/1".to_owned(),
                created_at_ms: 900,
                expires_at_ms: Some(2_000),
            },
            resource_scope: ResourceScope::new(["memory:provider:text"]),
            budget: Budget {
                wall_time_ms: Some(100),
                tokens: Some(128),
                tool_calls: Some(1),
            },
        }
    }

    fn coordinator(run_id: RunId) -> RuntimeCoordinator<InMemoryLedger> {
        let policy = AdmissionPolicy::deny_all()
            .allow_tool("memory.search")
            .allow_resources(ResourceScope::new(["memory:provider:text"]))
            .with_budget_limit(Budget {
                wall_time_ms: Some(500),
                tokens: Some(512),
                tool_calls: Some(2),
            });
        RuntimeCoordinator::new(run_id, policy, InMemoryLedger::default())
    }

    fn preparation_plan(request: &AdmissionRequest) -> PreparationPlan {
        PreparationPlan::new(
            request.proposal.run_id,
            request.proposal.action_id,
            request.proposal.base_revision,
            request.proposal.tool_name.clone(),
            request.proposal.arguments_digest.clone(),
            crate::PreparationEffectClass::Draft,
            ResourceScope::new(["workspace:draft"]),
            Budget {
                wall_time_ms: Some(100),
                tokens: Some(64),
                tool_calls: Some(1),
            },
            0.8,
            900,
            2_000,
        )
        .expect("test preparation should be valid")
    }

    #[test]
    fn successful_execution_is_granted_logged_and_versioned() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let result = coordinator
            .execute(request(run_id), 1_000, |grant, proposal| {
                assert_eq!(grant.tool_name, proposal.tool_name);
                Ok::<_, &'static str>(json!({"hits": 2}))
            })
            .expect("execution should complete");

        assert!(result.is_success());
        assert_eq!(coordinator.current_revision(), Revision::new(1));
        let events = coordinator
            .ledger()
            .scan_after(EventSequence::INITIAL)
            .expect("events should be readable");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].draft.kind.as_str(), ACTION_ADMITTED_EVENT);
        assert_eq!(
            events[0].draft.idempotency_key.as_deref(),
            Some("run/action/1")
        );
        assert_eq!(events[1].draft.kind.as_str(), ACTION_SUCCEEDED_EVENT);
        assert_eq!(events[1].draft.revision, Revision::new(1));
        let terminal_key = format!(
            "action:{}:{}",
            result.grant().grant_id,
            ACTION_SUCCEEDED_EVENT
        );
        assert_eq!(
            events[1].draft.idempotency_key.as_deref(),
            Some(terminal_key.as_str())
        );
        assert!(matches!(
            &result,
            ExecutionResult::Succeeded { output, outcome_sequence, .. }
                if output == &json!({"hits": 2}) && *outcome_sequence == EventSequence::new(2)
        ));
    }

    #[test]
    fn action_revision_change_invalidates_unconsumed_preparation() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let request = request(run_id);
        let plan = preparation_plan(&request);
        let plan_id = coordinator
            .submit_preparation(plan, 1_000)
            .expect("preparation should be bound to the current run");
        coordinator
            .start_preparation(plan_id, 1_001)
            .expect("preparation should start");
        coordinator
            .mark_preparation_ready(plan_id, 1_002)
            .expect("preparation should become ready");

        coordinator
            .execute(request, 1_003, |_grant, _proposal| {
                Ok::<_, &'static str>(json!({"ok": true}))
            })
            .expect("action should complete");

        assert_eq!(coordinator.current_revision(), Revision::new(1));
        assert_eq!(
            coordinator.preparations().get(plan_id).unwrap().status,
            crate::PreparationStatus::Invalidated
        );
    }

    #[test]
    fn preparation_lifecycle_is_recovered_and_can_be_consumed() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let request = request(run_id);
        let proposal = request.proposal.clone();
        let plan_id = coordinator
            .submit_preparation(preparation_plan(&request), 1_000)
            .expect("preparation should be durable");
        coordinator
            .start_preparation(plan_id, 1_001)
            .expect("starting state should be durable");
        coordinator
            .mark_preparation_ready(plan_id, 1_002)
            .expect("ready state should be durable");

        let (ledger, policy, _, _) = coordinator.into_parts();
        let mut recovered = RuntimeCoordinator::recover(run_id, policy, ledger)
            .expect("preparation state should be restored from the ledger");
        assert_eq!(
            recovered.preparations().get(plan_id).unwrap().status,
            crate::PreparationStatus::Ready
        );

        recovered
            .consume_preparation(plan_id, &proposal, 1_003)
            .expect("a matching action should consume the restored preparation");
        let (ledger, policy, _, _) = recovered.into_parts();
        let recovered = RuntimeCoordinator::recover(run_id, policy, ledger)
            .expect("consumed state should remain durable after another restart");
        assert_eq!(
            recovered.preparations().get(plan_id).unwrap().status,
            crate::PreparationStatus::Consumed
        );
    }

    #[test]
    fn preparation_worker_outcomes_are_durable() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let first_request = request(run_id);
        let second_request = request(run_id);
        let failed_id = coordinator
            .submit_preparation(preparation_plan(&first_request), 1_000)
            .expect("failed preparation should be durable");
        let unknown_id = coordinator
            .submit_preparation(preparation_plan(&second_request), 1_000)
            .expect("unknown preparation should be durable");

        let failed = coordinator
            .execute_preparation(failed_id, 1_001, 1_002, |_plan| {
                Err::<PreparationWorkResult, _>("provider disconnected")
            })
            .expect("worker failure should be recorded");
        assert_eq!(
            failed,
            PreparationWorkResult::Failed {
                error: "provider disconnected".to_owned()
            }
        );
        let unknown = coordinator
            .execute_preparation(unknown_id, 1_001, 1_002, |_plan| {
                Ok::<_, &str>(PreparationWorkResult::Unknown {
                    reason: "worker process exited".to_owned(),
                })
            })
            .expect("unknown worker outcome should be recorded");
        assert_eq!(
            unknown,
            PreparationWorkResult::Unknown {
                reason: "worker process exited".to_owned()
            }
        );

        let (ledger, policy, _, _) = coordinator.into_parts();
        let recovered = RuntimeCoordinator::recover(run_id, policy, ledger)
            .expect("worker outcomes should survive recovery");
        assert_eq!(
            recovered.preparations().get(failed_id).unwrap().outcome,
            Some(crate::PreparationOutcome::Failed {
                error: "provider disconnected".to_owned()
            })
        );
        assert_eq!(
            recovered.preparations().get(unknown_id).unwrap().outcome,
            Some(crate::PreparationOutcome::Unknown {
                reason: "worker process exited".to_owned()
            })
        );
    }

    #[test]
    fn preparation_worker_artifacts_survive_recovery() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let request = request(run_id);
        let artifact = crate::ArtifactRef {
            artifact_id: Uuid::from_u128(70),
            uri: "cas://runtime/run-1/sha256:video".to_owned(),
            media_type: "video/mp4".to_owned(),
            sha256: "sha256:video".to_owned(),
            bytes: 4_096,
            codec: Some("h264".to_owned()),
            schema: Some("episode-video-v1".to_owned()),
            stream_key: Some("camera/front".to_owned()),
            storage_revision: Some("fs-v1".to_owned()),
        };
        let evidence = crate::EvidenceLocator {
            artifact: artifact.clone(),
            entity_path: Some("robot/gripper/object-3".to_owned()),
            time_range: Some(pi_agent_memory::TimeRange {
                clock_domain: "episode_ms".to_owned(),
                start_ms: 1_200,
                end_ms: Some(1_400),
            }),
            frame_start: Some(36),
            frame_end: Some(42),
            step_start: Some(12),
            step_end: Some(14),
            byte_start: None,
            byte_end: None,
        };
        let plan_id = coordinator
            .submit_preparation(preparation_plan(&request), 1_000)
            .expect("artifact preparation should be durable");

        coordinator
            .execute_preparation(plan_id, 1_001, 1_200, |_plan| {
                Ok::<_, &str>(PreparationWorkResult::Ready {
                    artifacts: vec![artifact.clone()],
                    evidence: vec![evidence.clone()],
                })
            })
            .expect("artifact result should be recorded");
        assert_eq!(
            coordinator.preparations().get(plan_id).unwrap().artifacts,
            vec![artifact.clone()]
        );

        let error = coordinator
            .consume_preparation(plan_id, &request.proposal, 1_300)
            .expect_err("artifact output must be verified before consumption");
        assert!(matches!(
            error,
            CoordinatorError::ArtifactVerificationRequired(id) if id == plan_id
        ));
        assert_eq!(
            coordinator.preparations().get(plan_id).unwrap().status,
            crate::PreparationStatus::Ready
        );

        let error = coordinator
            .consume_preparation_with_resolver(plan_id, &request.proposal, 1_300, |_artifact| {
                Err::<crate::ArtifactRef, _>("object store unavailable")
            })
            .expect_err("unavailable artifact must keep the plan ready");
        assert!(matches!(
            error,
            CoordinatorError::ArtifactResolution { artifact_id, .. }
                if artifact_id == artifact.artifact_id
        ));

        let mut changed = artifact.clone();
        changed.storage_revision = Some("different-store".to_owned());
        let error = coordinator
            .consume_preparation_with_resolver(
                plan_id,
                &request.proposal,
                1_300,
                move |_artifact| Ok::<_, &str>(changed.clone()),
            )
            .expect_err("changed artifact metadata must be rejected");
        assert!(matches!(
            error,
            CoordinatorError::ArtifactMetadataMismatch { artifact_id }
                if artifact_id == artifact.artifact_id
        ));

        coordinator
            .consume_preparation_with_resolver(plan_id, &request.proposal, 1_300, |prepared| {
                Ok::<_, &str>(prepared.clone())
            })
            .expect("matching artifact metadata should allow consumption");
        assert_eq!(
            coordinator.preparations().get(plan_id).unwrap().status,
            crate::PreparationStatus::Consumed
        );

        let (ledger, policy, _, _) = coordinator.into_parts();
        let recovered = RuntimeCoordinator::recover(run_id, policy, ledger)
            .expect("artifact references should survive recovery");
        assert_eq!(
            recovered.preparations().get(plan_id).unwrap().artifacts,
            vec![artifact]
        );
        assert_eq!(
            recovered.preparations().get(plan_id).unwrap().evidence,
            vec![evidence]
        );
    }

    #[test]
    fn expired_worker_result_is_durably_invalidated() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let request = request(run_id);
        let plan_id = coordinator
            .submit_preparation(preparation_plan(&request), 1_000)
            .expect("preparation should be durable");

        let error = coordinator
            .execute_preparation(plan_id, 1_001, 2_000, |_plan| {
                Ok::<_, &str>(PreparationWorkResult::Ready {
                    artifacts: Vec::new(),
                    evidence: Vec::new(),
                })
            })
            .expect_err("a result after expiry must not become reusable");
        assert!(matches!(
            error,
            CoordinatorError::Preparation(PreparationError::Expired(id)) if id == plan_id
        ));
        assert_eq!(
            coordinator.preparations().get(plan_id).unwrap().status,
            crate::PreparationStatus::Invalidated
        );
        let (ledger, policy, _, _) = coordinator.into_parts();
        let recovered = RuntimeCoordinator::recover(run_id, policy, ledger)
            .expect("invalidated preparation should be recoverable");
        assert_eq!(
            recovered.preparations().get(plan_id).unwrap().status,
            crate::PreparationStatus::Invalidated
        );
    }

    #[test]
    fn failed_execution_is_logged_and_advances_revision() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let result = coordinator
            .execute(request(run_id), 1_000, |_grant, _proposal| {
                Err::<Value, _>("provider unavailable")
            })
            .expect("tool failure should be a recorded outcome");

        assert!(
            matches!(result, ExecutionResult::Failed { ref error, .. } if error == "provider unavailable")
        );
        assert_eq!(coordinator.current_revision(), Revision::new(1));
        let events = coordinator
            .ledger()
            .scan_after(EventSequence::INITIAL)
            .expect("events should be readable");
        assert_eq!(events[1].draft.kind.as_str(), ACTION_FAILED_EVENT);
    }

    #[test]
    fn interrupted_execution_is_logged_as_unknown_and_advances_revision() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let grant = coordinator
            .admit(request(run_id), 1_000)
            .expect("admission should complete");

        let result = coordinator
            .mark_unknown(grant, "provider timed out", 1_001)
            .expect("unknown outcome should be durable");

        assert!(matches!(
            result,
            ExecutionResult::Unknown {
                ref reason,
                outcome_sequence,
                ..
            } if reason == "provider timed out" && outcome_sequence == EventSequence::new(2)
        ));
        assert_eq!(coordinator.current_revision(), Revision::new(1));
        let events = coordinator
            .ledger()
            .scan_after(EventSequence::INITIAL)
            .expect("events should be readable");
        assert_eq!(events[1].draft.kind.as_str(), ACTION_UNKNOWN_EVENT);
        assert_eq!(events[1].draft.revision, Revision::new(1));
        assert_eq!(
            events[1].draft.idempotency_key.as_deref(),
            Some(
                format!(
                    "action:{}:{}",
                    result.grant().grant_id,
                    ACTION_UNKNOWN_EVENT
                )
                .as_str()
            )
        );
        assert_eq!(coordinator.pending_actions().len(), 1);
    }

    #[test]
    fn duplicate_unknown_outcome_is_rejected_without_advancing_revision() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let grant = coordinator
            .admit(request(run_id), 1_000)
            .expect("admission should complete");
        coordinator
            .mark_unknown(grant.clone(), "provider timed out", 1_001)
            .expect("first unknown outcome should be durable");

        let error = coordinator
            .mark_unknown(grant, "provider timed out again", 1_002)
            .expect_err("the same unknown outcome must not be appended twice");

        assert!(matches!(
            error,
            CoordinatorError::Ledger(LedgerError::DuplicateIdempotencyKey(key))
                if key.ends_with(":action_unknown")
        ));
        assert_eq!(coordinator.current_revision(), Revision::new(1));
        assert_eq!(coordinator.pending_actions().len(), 1);
    }

    #[test]
    fn reconciliation_records_evidence_and_closes_unknown_action() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let grant = coordinator
            .admit(request(run_id), 1_000)
            .expect("admission should be durable");

        let error = coordinator
            .reconcile(
                grant.clone(),
                ReconciliationOutcome::Failed {
                    error: "provider confirmed no operation".to_owned(),
                    evidence: "  ".to_owned(),
                },
                1_001,
            )
            .expect_err("blank reconciliation evidence must be rejected");
        assert!(matches!(
            error,
            CoordinatorError::EmptyReconciliationEvidence
        ));

        coordinator
            .mark_unknown(grant.clone(), "process handoff", 1_001)
            .expect("unknown outcome should be durable");
        let result = coordinator
            .reconcile(
                grant,
                ReconciliationOutcome::Failed {
                    error: "provider confirmed no operation".to_owned(),
                    evidence: "provider status query: operation absent".to_owned(),
                },
                1_002,
            )
            .expect("external evidence should close the action");

        assert!(matches!(
            result,
            ExecutionResult::Failed {
                ref error,
                outcome_sequence,
                ..
            } if error == "provider confirmed no operation"
                && outcome_sequence == EventSequence::new(3)
        ));
        assert_eq!(coordinator.current_revision(), Revision::new(2));
        assert!(coordinator.pending_actions().is_empty());
        let event = coordinator
            .ledger()
            .scan_after(EventSequence::new(2))
            .expect("reconciliation event should be readable")
            .pop()
            .expect("one reconciliation event should exist");
        assert_eq!(event.draft.kind.as_str(), ACTION_RECONCILED_EVENT);
        assert_eq!(event.draft.payload["status"], "reconciled");
        assert_eq!(event.draft.payload["outcome"], "failed");
        let terminal_key = format!(
            "action:{}:{}",
            result.grant().grant_id,
            ACTION_RECONCILED_EVENT
        );
        assert_eq!(
            event.draft.idempotency_key.as_deref(),
            Some(terminal_key.as_str())
        );
        assert_eq!(
            event.draft.payload["evidence"],
            "provider status query: operation absent"
        );
    }

    #[test]
    fn reconciliation_can_confirm_success() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let grant = coordinator
            .admit(request(run_id), 1_000)
            .expect("admission should be durable");
        coordinator
            .mark_unknown(grant.clone(), "process handoff", 1_001)
            .expect("unknown outcome should be durable");

        let result = coordinator
            .reconcile(
                grant,
                ReconciliationOutcome::Succeeded {
                    output: json!({"provider_id": "op-7"}),
                    evidence: "provider status query: operation committed".to_owned(),
                },
                1_002,
            )
            .expect("positive external evidence should close the action");

        assert!(matches!(
            result,
            ExecutionResult::Succeeded { output, .. }
                if output == json!({"provider_id": "op-7"})
        ));
        assert!(coordinator.pending_actions().is_empty());
        let event = coordinator
            .ledger()
            .scan_after(EventSequence::new(2))
            .expect("reconciliation event should be readable")
            .pop()
            .expect("one reconciliation event should exist");
        assert_eq!(event.draft.kind.as_str(), ACTION_RECONCILED_EVENT);
        assert_eq!(event.draft.payload["outcome"], "succeeded");
        assert_eq!(
            event.draft.payload["output"],
            json!({"provider_id": "op-7"})
        );
    }

    #[test]
    fn memory_decision_is_audited_without_advancing_action_revision() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let context = crate::MemoryHookContext {
            run_id,
            turn_id: Some(uuid::Uuid::from_u128(99)),
            namespace: "benchmark/text".to_owned(),
            current_revision: Revision::INITIAL,
            memory_revision: crate::MemoryRevision::new(3),
            policy_revision: crate::PolicyRevision::new(2),
            now_ms: 1_000,
            max_tokens: 128,
            max_candidates: 4,
            max_projection_bytes: 4_096,
        };
        let proposal_id = uuid::Uuid::from_u128(7);
        let candidate_id = uuid::Uuid::from_u128(8);
        let retrieval_summary = crate::MemoryRetrievalSummary {
            hit_count: 1,
            truncated: true,
            provider_failures: vec![crate::MemoryProviderFailure {
                method_id: "video.timeline".to_owned(),
                code: "timeout".to_owned(),
                message: "provider deadline exceeded".to_owned(),
                retryable: true,
            }],
            ..crate::MemoryRetrievalSummary::default()
        };
        let sequence = coordinator
            .record_memory_decision_with_summary(
                &context,
                &InjectionDecision::Inject {
                    proposal_id,
                    injections: vec![crate::MemoryInjection {
                        candidate_id,
                        projection: crate::MemoryProjection {
                            projection_id: uuid::Uuid::from_u128(9),
                            source_memory_id: uuid::Uuid::from_u128(10),
                            source_version: 1,
                            content: "parser fix".to_owned(),
                            evidence_refs: vec![serde_json::json!({
                                "artifact_id": "artifact-1",
                                "frame_start": 12,
                                "frame_end": 18,
                            })],
                            provenance: "text-memory:record-1".to_owned(),
                            token_cost: 2,
                        },
                    }],
                    total_token_cost: 2,
                },
                1_001,
                Some(&retrieval_summary),
            )
            .expect("memory decision should be durable");

        assert_eq!(sequence, EventSequence::new(1));
        assert_eq!(coordinator.current_revision(), Revision::INITIAL);
        let event = coordinator
            .ledger()
            .scan_after(EventSequence::INITIAL)
            .expect("memory event should be readable")
            .pop()
            .expect("one memory event should exist");
        assert_eq!(event.draft.kind.as_str(), MEMORY_INJECTED_EVENT);
        assert_eq!(
            event.draft.payload["details"]["proposal_id"],
            serde_json::json!(proposal_id)
        );
        assert_eq!(
            event.draft.payload["turn_id"],
            serde_json::json!(uuid::Uuid::from_u128(99))
        );
        assert_eq!(
            event.draft.payload["details"]["injections"][0]["candidate_id"],
            serde_json::json!(candidate_id)
        );
        assert_eq!(
            event.draft.payload["details"]["injections"][0]["projection_id"],
            serde_json::json!(uuid::Uuid::from_u128(9))
        );
        assert_eq!(
            event.draft.payload["details"]["injections"][0]["evidence_refs"],
            serde_json::json!([{
                "artifact_id": "artifact-1",
                "frame_start": 12,
                "frame_end": 18,
            }])
        );
        assert_eq!(
            event.draft.payload["details"]["injections"][0]["content_digest"],
            serde_json::json!(bytes_digest(b"parser fix"))
        );
        assert_eq!(
            event.draft.payload["details"]["retrieval_summary"],
            serde_json::json!({
                "hit_count": 1,
                "truncated": true,
                "provider_failures": [{
                    "method_id": "video.timeline",
                    "code": "timeout",
                    "message": "provider deadline exceeded",
                    "retryable": true,
                }],
            })
        );
        let expected_key = format!(
            "memory:{proposal_id}:run-{}:runtime-0:memory-3:policy-2",
            run_id.as_uuid()
        );
        assert_eq!(
            event.draft.idempotency_key.as_deref(),
            Some(expected_key.as_str())
        );
    }

    #[test]
    fn jsonl_reopen_preserves_memory_trace_evidence() {
        let run_id = RunId::new();
        let path = std::env::temp_dir().join(format!(
            "pi-agent-runtime-memory-trace-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let policy = AdmissionPolicy::deny_all();
        let context = crate::MemoryHookContext {
            run_id,
            turn_id: Some(uuid::Uuid::from_u128(99)),
            namespace: "benchmark/video".to_owned(),
            current_revision: Revision::INITIAL,
            memory_revision: crate::MemoryRevision::new(3),
            policy_revision: crate::PolicyRevision::new(2),
            now_ms: 1_000,
            max_tokens: 128,
            max_candidates: 4,
            max_projection_bytes: 4_096,
        };
        let proposal_id = uuid::Uuid::from_u128(7);
        let summary = crate::MemoryRetrievalSummary {
            query_digest: Some("sha256:query".to_owned()),
            snapshot_id: Some("snapshot:7".to_owned()),
            hit_count: 1,
            truncated: false,
            hits: vec![crate::MemoryRetrievalHit {
                item_id: uuid::Uuid::from_u128(8),
                source_record_ids: Vec::new(),
                version: 2,
                method_id: "video.timeline".to_owned(),
                rank: 1,
                snapshot_id: Some("snapshot:7".to_owned()),
                evidence_ids: vec!["evidence:frame-12".to_owned()],
            }],
            provider_failures: Vec::new(),
        };

        let ledger = JsonlLedger::open(&path).expect("JSONL ledger should open");
        let mut coordinator = RuntimeCoordinator::new(run_id, policy, ledger);
        coordinator
            .record_memory_decision_with_summary(
                &context,
                &InjectionDecision::Skip {
                    proposal_id,
                    reason: crate::InjectionSkipReason::NoEligibleCandidate,
                },
                1_001,
                Some(&summary),
            )
            .expect("memory trace should be durable");
        coordinator
            .close()
            .expect("first ledger handle should close");

        let ledger = JsonlLedger::open(&path).expect("JSONL ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("reopened ledger should be readable");
        let event = events
            .iter()
            .find(|event| event.draft.kind.as_str() == MEMORY_SKIPPED_EVENT)
            .expect("reopened ledger should contain the memory decision");
        let restored: crate::MemoryRetrievalSummary =
            serde_json::from_value(event.draft.payload["details"]["retrieval_summary"].clone())
                .expect("memory trace should deserialize after reopen");
        assert_eq!(restored, summary);

        let mut ledger = ledger;
        ledger.close().expect("reopened ledger should close");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}.lock", path.display()));
    }

    #[test]
    fn duplicate_memory_decision_is_rejected_at_the_same_revision() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let context = crate::MemoryHookContext {
            run_id,
            turn_id: None,
            namespace: "benchmark/text".to_owned(),
            current_revision: Revision::INITIAL,
            memory_revision: crate::MemoryRevision::new(3),
            policy_revision: crate::PolicyRevision::new(2),
            now_ms: 1_000,
            max_tokens: 128,
            max_candidates: 4,
            max_projection_bytes: 4_096,
        };
        let decision = InjectionDecision::Skip {
            proposal_id: uuid::Uuid::from_u128(7),
            reason: crate::InjectionSkipReason::NoEligibleCandidate,
        };

        coordinator
            .record_memory_decision(&context, &decision, 1_001)
            .expect("first memory decision should be durable");
        let error = coordinator
            .record_memory_decision(&context, &decision, 1_002)
            .expect_err("the same decision must not be counted twice");
        assert!(matches!(
            error,
            CoordinatorError::Ledger(LedgerError::DuplicateIdempotencyKey(key))
                if key.starts_with("memory:00000000-0000-0000-0000-000000000007:")
        ));
    }

    #[test]
    fn same_memory_proposal_id_is_allowed_in_a_different_run() {
        let first_run = RunId::new();
        let second_run = RunId::new();
        let mut first = coordinator(first_run);
        let context = |run_id| crate::MemoryHookContext {
            run_id,
            turn_id: None,
            namespace: "benchmark/text".to_owned(),
            current_revision: Revision::INITIAL,
            memory_revision: crate::MemoryRevision::INITIAL,
            policy_revision: crate::PolicyRevision::INITIAL,
            now_ms: 1_000,
            max_tokens: 128,
            max_candidates: 4,
            max_projection_bytes: 4_096,
        };
        let decision = InjectionDecision::Skip {
            proposal_id: uuid::Uuid::from_u128(7),
            reason: crate::InjectionSkipReason::NoEligibleCandidate,
        };

        first
            .record_memory_decision(&context(first_run), &decision, 1_001)
            .expect("first run decision should be durable");
        let (ledger, policy, _, _) = first.into_parts();
        let mut second = RuntimeCoordinator::new(second_run, policy, ledger);

        second
            .record_memory_decision(&context(second_run), &decision, 1_002)
            .expect("the same provider proposal id may be reused by another run");

        let events = second
            .ledger()
            .scan_after(EventSequence::INITIAL)
            .expect("both decisions should be readable");
        assert_eq!(events.len(), 2);
        assert_ne!(
            events[0].draft.idempotency_key,
            events[1].draft.idempotency_key
        );
    }

    #[test]
    fn memory_decision_rejects_a_stale_runtime_context() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let mut context = crate::MemoryHookContext {
            run_id,
            turn_id: None,
            namespace: "benchmark/text".to_owned(),
            current_revision: Revision::new(1),
            memory_revision: crate::MemoryRevision::INITIAL,
            policy_revision: crate::PolicyRevision::INITIAL,
            now_ms: 1_000,
            max_tokens: 128,
            max_candidates: 4,
            max_projection_bytes: 4_096,
        };
        let decision = InjectionDecision::Skip {
            proposal_id: uuid::Uuid::from_u128(7),
            reason: crate::InjectionSkipReason::RuntimeRevisionMismatch,
        };
        let error = coordinator
            .record_memory_decision(&context, &decision, 1_001)
            .expect_err("stale memory context should not be recorded");
        assert!(matches!(
            error,
            CoordinatorError::RevisionConflict { expected, actual }
                if expected == Revision::INITIAL && actual == Revision::new(1)
        ));
        context.current_revision = Revision::INITIAL;
        coordinator
            .record_memory_decision(&context, &decision, 1_001)
            .expect("current memory context should be recordable");
    }

    #[test]
    fn evaluate_memory_proposal_audits_a_runtime_skip() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let context = crate::MemoryHookContext {
            run_id,
            turn_id: None,
            namespace: "benchmark/text".to_owned(),
            current_revision: Revision::INITIAL,
            memory_revision: crate::MemoryRevision::INITIAL,
            policy_revision: crate::PolicyRevision::INITIAL,
            now_ms: 1_000,
            max_tokens: 128,
            max_candidates: 4,
            max_projection_bytes: 4_096,
        };
        let proposal = crate::MemoryProposal {
            proposal_id: uuid::Uuid::from_u128(11),
            run_id,
            namespace: context.namespace.clone(),
            base_revision: Revision::new(1),
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: Vec::new(),
            created_at_ms: 900,
            expires_at_ms: Some(2_000),
            retrieval_summary: Some(crate::MemoryRetrievalSummary {
                hit_count: 0,
                truncated: true,
                provider_failures: vec![crate::MemoryProviderFailure {
                    method_id: "graph".to_owned(),
                    code: "unavailable".to_owned(),
                    message: "graph index is offline".to_owned(),
                    retryable: false,
                }],
                ..crate::MemoryRetrievalSummary::default()
            }),
        };

        let receipt = coordinator
            .evaluate_memory_proposal_with_receipt(&proposal, &context, 1_001)
            .expect("a stale proposal should become an auditable skip");
        assert!(matches!(
            receipt.decision,
            InjectionDecision::Skip {
                reason: crate::InjectionSkipReason::RuntimeRevisionMismatch,
                ..
            }
        ));
        assert_eq!(receipt.event_sequence, EventSequence::new(1));
        let event = coordinator
            .ledger()
            .scan_after(EventSequence::INITIAL)
            .expect("memory skip should be readable")
            .pop()
            .expect("one memory event should exist");
        assert_eq!(event.draft.kind.as_str(), MEMORY_SKIPPED_EVENT);
        assert_eq!(
            event.draft.payload["details"]["retrieval_summary"]["provider_failures"][0]
                ["method_id"],
            "graph"
        );
    }

    #[test]
    fn recovery_blocks_new_work_until_an_unknown_action_is_reconciled() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        coordinator
            .admit(request(run_id), 1_000)
            .expect("admission should be durable before the simulated crash");
        let (ledger, policy, _, _) = coordinator.into_parts();

        let mut recovered = RuntimeCoordinator::recover(run_id, policy, ledger)
            .expect("recovery should rebuild the pending grant");
        assert_eq!(recovered.pending_actions().len(), 1);
        let error = recovered
            .admit(request(run_id), 1_001)
            .expect_err("new work must wait for reconciliation");
        assert!(matches!(error, CoordinatorError::PendingAction { .. }));

        let grant = recovered.pending_actions()[0].clone();
        recovered
            .mark_unknown(grant.clone(), "process handoff", 1_001)
            .expect("unknown outcome should remain durable");
        assert_eq!(recovered.pending_actions().len(), 1);
        recovered
            .reconcile(
                grant,
                ReconciliationOutcome::Failed {
                    error: "provider confirmed no operation".to_owned(),
                    evidence: "provider status query: operation absent".to_owned(),
                },
                1_002,
            )
            .expect("reconciliation should close the action");
        assert!(recovered.pending_actions().is_empty());
        assert_eq!(recovered.current_revision(), Revision::new(2));
    }

    #[test]
    fn jsonl_recovery_preserves_unknown_action_until_reconciliation() {
        let run_id = RunId::new();
        let path = std::env::temp_dir().join(format!(
            "pi-agent-runtime-coordinator-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let policy = AdmissionPolicy::deny_all()
            .allow_tool("memory.search")
            .allow_resources(ResourceScope::new(["memory:provider:text"]))
            .with_budget_limit(Budget {
                wall_time_ms: Some(500),
                tokens: Some(512),
                tool_calls: Some(2),
            });

        let ledger = JsonlLedger::open(&path).expect("JSONL ledger should open");
        let mut coordinator = RuntimeCoordinator::new(run_id, policy.clone(), ledger);
        let grant = coordinator
            .admit(request(run_id), 1_000)
            .expect("admission should be durable");
        coordinator
            .mark_unknown(grant.clone(), "process handoff", 1_001)
            .expect("unknown outcome should be durable");
        coordinator.close().expect("first handle should close");

        let ledger = JsonlLedger::open(&path).expect("JSONL ledger should reopen");
        let mut recovered = RuntimeCoordinator::recover_latest(policy.clone(), ledger)
            .expect("JSONL history should be replayable")
            .expect("the unresolved run should be selected");
        assert_eq!(recovered.run_id(), run_id);
        assert_eq!(recovered.current_revision(), Revision::new(1));
        assert_eq!(recovered.pending_actions(), std::slice::from_ref(&grant));

        recovered
            .reconcile(
                grant,
                ReconciliationOutcome::Failed {
                    error: "provider confirmed no operation".to_owned(),
                    evidence: "provider status query: operation absent".to_owned(),
                },
                1_002,
            )
            .expect("reconciliation should append a terminal event");
        recovered.close().expect("second handle should close");

        let ledger = JsonlLedger::open(&path).expect("JSONL ledger should reopen again");
        let mut completed = RuntimeCoordinator::recover_latest(policy, ledger)
            .expect("completed JSONL history should be replayable")
            .expect("the completed run should remain discoverable");
        assert_eq!(completed.run_id(), run_id);
        assert_eq!(completed.current_revision(), Revision::new(2));
        assert!(completed.pending_actions().is_empty());
        completed.close().expect("final handle should close");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}.lock", path.display()));
    }

    #[test]
    fn jsonl_recovery_restores_preparation_lifecycle() {
        let run_id = RunId::new();
        let path = std::env::temp_dir().join(format!(
            "pi-agent-runtime-preparation-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let policy = AdmissionPolicy::deny_all()
            .allow_tool("memory.search")
            .allow_resources(ResourceScope::new(["memory:provider:text"]))
            .with_budget_limit(Budget {
                wall_time_ms: Some(500),
                tokens: Some(512),
                tool_calls: Some(2),
            });
        let request = request(run_id);
        let proposal = request.proposal.clone();

        let ledger = JsonlLedger::open(&path).expect("JSONL ledger should open");
        let mut coordinator = RuntimeCoordinator::new(run_id, policy.clone(), ledger);
        let plan_id = coordinator
            .submit_preparation(preparation_plan(&request), 1_000)
            .expect("preparation should be durable");
        coordinator
            .start_preparation(plan_id, 1_001)
            .expect("starting state should be durable");
        coordinator
            .mark_preparation_ready(plan_id, 1_002)
            .expect("ready state should be durable");
        coordinator
            .close()
            .expect("first ledger handle should close");

        let ledger = JsonlLedger::open(&path).expect("JSONL ledger should reopen");
        let mut recovered = RuntimeCoordinator::recover_latest(policy.clone(), ledger)
            .expect("JSONL preparation history should be replayable")
            .expect("the preparation run should be discoverable");
        assert_eq!(
            recovered.preparations().get(plan_id).unwrap().status,
            crate::PreparationStatus::Ready
        );
        recovered
            .consume_preparation(plan_id, &proposal, 1_003)
            .expect("the restored preparation should still be consumable");
        recovered
            .close()
            .expect("second ledger handle should close");

        let ledger = JsonlLedger::open(&path).expect("JSONL ledger should reopen again");
        let mut recovered = RuntimeCoordinator::recover_latest(policy, ledger)
            .expect("consumed preparation history should be replayable")
            .expect("the consumed run should remain discoverable");
        assert_eq!(
            recovered.preparations().get(plan_id).unwrap().status,
            crate::PreparationStatus::Consumed
        );
        recovered.close().expect("final ledger handle should close");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}.lock", path.display()));
    }

    #[test]
    fn jsonl_recovery_restores_precise_preparation_evidence() {
        let run_id = RunId::new();
        let path = std::env::temp_dir().join(format!(
            "pi-agent-runtime-preparation-evidence-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let policy = AdmissionPolicy::deny_all()
            .allow_tool("memory.search")
            .allow_resources(ResourceScope::new(["memory:provider:text"]))
            .with_budget_limit(Budget {
                wall_time_ms: Some(500),
                tokens: Some(512),
                tool_calls: Some(2),
            });
        let request = request(run_id);
        let artifact = crate::ArtifactRef {
            artifact_id: Uuid::from_u128(71),
            uri: "cas://runtime/run-2/sha256:video".to_owned(),
            media_type: "video/mp4".to_owned(),
            sha256: "sha256:video-2".to_owned(),
            bytes: 8_192,
            codec: Some("h264".to_owned()),
            schema: Some("episode-video-v1".to_owned()),
            stream_key: Some("camera/rear".to_owned()),
            storage_revision: Some("fs-v1".to_owned()),
        };
        let evidence = crate::EvidenceLocator {
            artifact: artifact.clone(),
            entity_path: Some("robot/base/object-7".to_owned()),
            time_range: Some(pi_agent_memory::TimeRange {
                clock_domain: "episode_ms".to_owned(),
                start_ms: 2_200,
                end_ms: Some(2_400),
            }),
            frame_start: Some(66),
            frame_end: Some(72),
            step_start: Some(22),
            step_end: Some(24),
            byte_start: Some(1_024),
            byte_end: Some(4_096),
        };

        let ledger = JsonlLedger::open(&path).expect("JSONL ledger should open");
        let mut coordinator = RuntimeCoordinator::new(run_id, policy.clone(), ledger);
        let plan_id = coordinator
            .submit_preparation(preparation_plan(&request), 1_000)
            .expect("preparation should be durable");
        coordinator
            .execute_preparation(plan_id, 1_001, 1_200, |_plan| {
                Ok::<_, &str>(PreparationWorkResult::Ready {
                    artifacts: vec![artifact.clone()],
                    evidence: vec![evidence.clone()],
                })
            })
            .expect("precise evidence should be recorded");
        coordinator
            .close()
            .expect("first ledger handle should close");

        let ledger = JsonlLedger::open(&path).expect("JSONL ledger should reopen");
        let mut recovered = RuntimeCoordinator::recover_latest(policy, ledger)
            .expect("JSONL evidence history should be replayable")
            .expect("the preparation run should be discoverable");
        let restored = recovered
            .preparations()
            .get(plan_id)
            .expect("preparation should survive recovery");
        assert_eq!(restored.artifacts, vec![artifact]);
        assert_eq!(restored.evidence, vec![evidence]);
        recovered
            .close()
            .expect("second ledger handle should close");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}.lock", path.display()));
    }

    #[test]
    fn recover_latest_finds_the_only_unresolved_run() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        coordinator
            .admit(request(run_id), 1_000)
            .expect("admission should be durable before restart");
        let (ledger, policy, _, _) = coordinator.into_parts();

        let recovered = RuntimeCoordinator::recover_latest(policy, ledger)
            .expect("latest run should be recoverable")
            .expect("a non-empty ledger should yield a run");
        assert_eq!(recovered.run_id(), run_id);
        assert_eq!(recovered.current_revision(), Revision::INITIAL);
        assert_eq!(recovered.pending_actions().len(), 1);
    }

    #[test]
    fn recover_latest_rejects_multiple_unresolved_runs() {
        let first_run = RunId::new();
        let mut first = coordinator(first_run);
        first
            .admit(request(first_run), 1_000)
            .expect("first admission should be durable");
        let (ledger, policy, _, _) = first.into_parts();

        let second_run = RunId::new();
        let mut second = RuntimeCoordinator::new(second_run, policy.clone(), ledger);
        let mut second_request = request(second_run);
        second_request.proposal.idempotency_key = format!("{}:action/1", second_run.as_uuid());
        second
            .admit(second_request, 1_001)
            .expect("second admission should be durable");
        let (ledger, _, _, _) = second.into_parts();

        let error = match RuntimeCoordinator::recover_latest(policy, ledger) {
            Ok(_) => panic!("ambiguous unresolved runs must fail closed"),
            Err(error) => error,
        };
        assert!(matches!(error, CoordinatorError::AmbiguousRecovery { runs } if runs.len() == 2));
    }

    #[test]
    fn explicit_recovery_only_replays_the_requested_run() {
        let first_run = RunId::new();
        let mut first = coordinator(first_run);
        let grant = first
            .admit(request(first_run), 1_000)
            .expect("first run admission should be durable");
        let (ledger, policy, _, _) = first.into_parts();

        let second_run = RunId::new();
        let mut second = RuntimeCoordinator::new(second_run, policy.clone(), ledger);
        let mut second_request = request(second_run);
        second_request.proposal.idempotency_key = "second-run/action/1".to_owned();
        second
            .execute(second_request, 1_001, |_, _| Ok::<_, &str>(json!("done")))
            .expect("second run should complete");
        let (ledger, _, _, _) = second.into_parts();

        let first = RuntimeCoordinator::recover(first_run, policy.clone(), ledger)
            .expect("another run must not prevent explicit recovery");
        assert_eq!(first.current_revision(), Revision::INITIAL);
        assert_eq!(first.pending_actions(), std::slice::from_ref(&grant));
        let (ledger, _, _, _) = first.into_parts();
        let second = RuntimeCoordinator::recover(second_run, policy, ledger)
            .expect("second run should recover");
        assert_eq!(second.current_revision(), Revision::new(1));
        assert!(second.pending_actions().is_empty());
    }

    #[test]
    fn recover_latest_returns_none_for_an_empty_ledger() {
        let recovered = RuntimeCoordinator::recover_latest(
            AdmissionPolicy::deny_all(),
            InMemoryLedger::default(),
        )
        .expect("empty ledger should be a valid state");
        assert!(recovered.is_none());
    }

    #[test]
    fn rejected_proposal_never_reaches_executor_or_ledger() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let mut request = request(run_id);
        request.proposal.base_revision = Revision::new(4);
        let mut executed = false;

        let error = coordinator
            .execute(request, 1_000, |_grant, _proposal| {
                executed = true;
                Ok::<_, &'static str>(Value::Null)
            })
            .expect_err("stale proposal should be rejected");

        assert!(matches!(
            error,
            CoordinatorError::Admission(AdmissionError::RevisionConflict { .. })
        ));
        assert!(!executed);
        assert_eq!(coordinator.current_revision(), Revision::INITIAL);
        assert!(coordinator
            .ledger()
            .scan_after(EventSequence::INITIAL)
            .expect("ledger should be readable")
            .is_empty());
    }

    #[test]
    fn proposal_from_another_run_is_rejected_before_policy_evaluation() {
        let coordinator_run = RunId::new();
        let mut coordinator = coordinator(coordinator_run);
        let error = coordinator
            .admit(request(RunId::new()), 1_000)
            .expect_err("run mismatch should be rejected");

        assert!(matches!(error, CoordinatorError::RunMismatch { .. }));
    }

    #[test]
    fn recovery_rebuilds_revision_and_duplicate_idempotency_is_rejected() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        coordinator
            .execute(request(run_id), 1_000, |_grant, _proposal| {
                Ok::<_, &'static str>(json!({"hits": 1}))
            })
            .expect("initial execution should complete");

        let (ledger, policy, recovered_run_id, revision) = coordinator.into_parts();
        assert_eq!(recovered_run_id, run_id);
        assert_eq!(revision, Revision::new(1));
        let recovered = RuntimeCoordinator::recover(run_id, policy.clone(), ledger)
            .expect("persisted revision should be recoverable");
        assert_eq!(recovered.current_revision(), Revision::new(1));

        let mut fresh = RuntimeCoordinator::new(run_id, policy, InMemoryLedger::default());
        let initial_request = request(run_id);
        fresh
            .execute(initial_request, 1_000, |_grant, _proposal| {
                Ok::<_, &'static str>(json!({"hits": 1}))
            })
            .expect("first execution should succeed");
        let mut duplicate_request = request(run_id);
        duplicate_request.proposal.base_revision = Revision::new(1);
        let duplicate = fresh
            .admit(duplicate_request, 1_000)
            .expect_err("same idempotency key must not be admitted twice");
        assert!(matches!(
            duplicate,
            CoordinatorError::Ledger(LedgerError::DuplicateIdempotencyKey(_))
        ));
    }
}
