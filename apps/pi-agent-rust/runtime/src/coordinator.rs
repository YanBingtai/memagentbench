//! Small executable control loop for side-effecting actions.
//!
//! [`RuntimeCoordinator`] is the first runtime vertical slice. It admits an
//! action, durably records the grant before execution, invokes a caller-owned
//! executor, and records the outcome before advancing the runtime revision.
//! The executor is intentionally a callback: the runtime controls authority
//! and recovery boundaries without depending on a particular tool, model, or
//! memory provider.

use crate::{
    ActionId, ActionProposal, AdmissionContext, AdmissionError, AdmissionPolicy, AdmissionRequest,
    CommitGrant, EventDraft, EventDraftError, EventKind, EventLedger, EventSequence,
    InjectionDecision, LedgerError, Revision, RunId,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashMap, fmt, future::Future};

/// Event emitted after an action receives a durable commit grant.
pub const ACTION_ADMITTED_EVENT: &str = "action_admitted";
/// Event emitted after the caller-owned executor returns successfully.
pub const ACTION_SUCCEEDED_EVENT: &str = "action_succeeded";
/// Event emitted after the caller-owned executor reports a failure.
pub const ACTION_FAILED_EVENT: &str = "action_failed";
/// Event emitted when execution stops before success or failure is known.
pub const ACTION_UNKNOWN_EVENT: &str = "action_unknown";
/// Event emitted when a validated memory proposal is injected into a request.
pub const MEMORY_INJECTED_EVENT: &str = "memory_injected";
/// Event emitted when a memory proposal is evaluated but not injected.
pub const MEMORY_SKIPPED_EVENT: &str = "memory_skipped";

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

    /// Creates a coordinator from a caller-verified checkpoint with no pending actions.
    /// Use [`Self::recover`] to rebuild pending grants from an existing ledger.
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
}

impl<L: EventLedger> RuntimeCoordinator<L> {
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
            .map_err(CoordinatorError::Ledger)?;
        let state = replay_run(run_id, events)?;
        let mut coordinator = Self::from_revision(run_id, policy, state.revision, ledger);
        coordinator.pending_actions = state.pending_actions;
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
            None,
        )?;
        // Keep the grant pending until a caller reconciles the external
        // outcome. Even a declared read can hide provider-specific effects.
        self.current_revision = next_revision;

        Ok(ExecutionResult::Unknown {
            grant,
            reason,
            outcome_sequence,
        })
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

        let (kind, status, details) = match decision {
            InjectionDecision::Inject {
                proposal_id,
                injections,
                total_token_cost,
            } => (
                MEMORY_INJECTED_EVENT,
                "injected",
                json!({
                    "proposal_id": proposal_id,
                    "candidate_ids": injections
                        .iter()
                        .map(|injection| injection.candidate_id)
                        .collect::<Vec<_>>(),
                    "total_token_cost": total_token_cost,
                }),
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
            ),
        };
        let payload = json!({
            "status": status,
            "run_id": context.run_id,
            "namespace": context.namespace,
            "runtime_revision": context.current_revision,
            "memory_revision": context.memory_revision,
            "policy_revision": context.policy_revision,
            "details": details,
        });
        self.append_and_flush(kind, self.current_revision, now_ms, payload, None)
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
        let decision = proposal
            .evaluate(context)
            .map_err(CoordinatorError::MemoryProposal)?;
        self.record_memory_decision(context, &decision, now_ms)?;
        Ok(decision)
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
        self.validate_pending(&grant)?;
        match execution {
            Ok(output) => {
                let payload = json!({
                    "grant_id": grant.grant_id,
                    "action_id": grant.action_id,
                    "tool_name": grant.tool_name,
                    "status": "succeeded",
                    "output": output.clone(),
                });
                let outcome_sequence = self.append_and_flush(
                    ACTION_SUCCEEDED_EVENT,
                    next_revision,
                    now_ms,
                    payload,
                    None,
                )?;
                self.remove_pending(grant.action_id);
                self.current_revision = next_revision;

                Ok(ExecutionResult::Succeeded {
                    grant,
                    output,
                    outcome_sequence,
                })
            }
            Err(error) => {
                let outcome_sequence = self.append_and_flush(
                    ACTION_FAILED_EVENT,
                    next_revision,
                    now_ms,
                    json!({
                        "grant_id": grant.grant_id,
                        "action_id": grant.action_id,
                        "tool_name": grant.tool_name,
                        "status": "failed",
                        "error": error,
                    }),
                    None,
                )?;
                self.remove_pending(grant.action_id);
                self.current_revision = next_revision;

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
            ACTION_SUCCEEDED_EVENT | ACTION_FAILED_EVENT | ACTION_UNKNOWN_EVENT => {
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
            _ => {}
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
    MemoryProposal(crate::MemoryProposalError),
    Admission(AdmissionError),
    Draft(EventDraftError),
    Ledger(LedgerError),
    Serialize(serde_json::Error),
    Deserialize {
        kind: String,
        source: serde_json::Error,
    },
    RevisionExhausted,
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
            Self::MemoryProposal(error) => error.fmt(formatter),
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
            Self::RunMismatch { .. }
            | Self::PendingAction { .. }
            | Self::AmbiguousRecovery { .. }
            | Self::UnrecognizedGrant(_)
            | Self::RevisionConflict { .. }
            | Self::RevisionExhausted => None,
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
    use crate::{Budget, InMemoryLedger, ResourceScope, RiskClass};

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
        assert!(matches!(
            &result,
            ExecutionResult::Succeeded { output, outcome_sequence, .. }
                if output == &json!({"hits": 2}) && *outcome_sequence == EventSequence::new(2)
        ));
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
        assert_eq!(coordinator.pending_actions().len(), 1);
    }

    #[test]
    fn memory_decision_is_audited_without_advancing_action_revision() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let context = crate::MemoryHookContext {
            run_id,
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
        let sequence = coordinator
            .record_memory_decision(
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
                            provenance: "text-memory:record-1".to_owned(),
                            token_cost: 2,
                        },
                    }],
                    total_token_cost: 2,
                },
                1_001,
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
            event.draft.payload["details"]["candidate_ids"][0],
            serde_json::json!(candidate_id)
        );
    }

    #[test]
    fn memory_decision_rejects_a_stale_runtime_context() {
        let run_id = RunId::new();
        let mut coordinator = coordinator(run_id);
        let mut context = crate::MemoryHookContext {
            run_id,
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
        };

        let decision = coordinator
            .evaluate_memory_proposal(&proposal, &context, 1_001)
            .expect("a stale proposal should become an auditable skip");
        assert!(matches!(
            decision,
            InjectionDecision::Skip {
                reason: crate::InjectionSkipReason::RuntimeRevisionMismatch,
                ..
            }
        ));
        let event = coordinator
            .ledger()
            .scan_after(EventSequence::INITIAL)
            .expect("memory skip should be readable")
            .pop()
            .expect("one memory event should exist");
        assert_eq!(event.draft.kind.as_str(), MEMORY_SKIPPED_EVENT);
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
            .fail(grant, "provider confirmed no operation", 1_002)
            .expect("reconciliation should close the action");
        assert!(recovered.pending_actions().is_empty());
        assert_eq!(recovered.current_revision(), Revision::new(2));
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

        let error = RuntimeCoordinator::recover_latest(policy, ledger)
            .expect_err("ambiguous unresolved runs must fail closed");
        assert!(matches!(error, CoordinatorError::AmbiguousRecovery { runs } if runs.len() == 2));
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
