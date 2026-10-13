//! Small executable control loop for side-effecting actions.
//!
//! [`RuntimeCoordinator`] is the first runtime vertical slice. It admits an
//! action, durably records the grant before execution, invokes a caller-owned
//! executor, and records the outcome before advancing the runtime revision.
//! The executor is intentionally a callback: the runtime controls authority
//! and recovery boundaries without depending on a particular tool, model, or
//! memory provider.

use crate::{
    ActionProposal, AdmissionContext, AdmissionError, AdmissionPolicy, AdmissionRequest,
    CommitGrant, EventDraft, EventDraftError, EventKind, EventLedger, EventSequence, LedgerError,
    Revision, RunId,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fmt, future::Future};

/// Event emitted after an action receives a durable commit grant.
pub const ACTION_ADMITTED_EVENT: &str = "action_admitted";
/// Event emitted after the caller-owned executor returns successfully.
pub const ACTION_SUCCEEDED_EVENT: &str = "action_succeeded";
/// Event emitted after the caller-owned executor reports a failure.
pub const ACTION_FAILED_EVENT: &str = "action_failed";
/// Event emitted when execution stops before success or failure is known.
pub const ACTION_UNKNOWN_EVENT: &str = "action_unknown";

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
}

impl<L> RuntimeCoordinator<L> {
    /// Creates a coordinator at the initial runtime revision.
    pub fn new(run_id: RunId, policy: AdmissionPolicy, ledger: L) -> Self {
        Self {
            ledger,
            policy,
            run_id,
            current_revision: Revision::INITIAL,
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

    /// Recovers a coordinator from a known run revision and ledger.
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
        }
    }

    /// Consumes the coordinator and returns its owned parts.
    pub fn into_parts(self) -> (L, AdmissionPolicy, RunId, Revision) {
        (self.ledger, self.policy, self.run_id, self.current_revision)
    }
}

impl<L: EventLedger> RuntimeCoordinator<L> {
    /// Rebuilds the coordinator revision from the events belonging to a run.
    ///
    /// An admission event keeps the current revision, while a terminal
    /// outcome records the next revision. Taking the greatest persisted
    /// revision therefore recovers the state needed for the next proposal.
    pub fn recover(
        run_id: RunId,
        policy: AdmissionPolicy,
        ledger: L,
    ) -> Result<Self, CoordinatorError> {
        let current_revision = ledger
            .scan_after(EventSequence::INITIAL)
            .map_err(CoordinatorError::Ledger)?
            .into_iter()
            .filter(|event| event.draft.run_id == run_id)
            .map(|event| event.draft.revision)
            .max()
            .unwrap_or(Revision::INITIAL);

        Ok(Self::from_revision(
            run_id,
            policy,
            current_revision,
            ledger,
        ))
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
        self.current_revision = next_revision;

        Ok(ExecutionResult::Unknown {
            grant,
            reason,
            outcome_sequence,
        })
    }

    fn next_revision(&self) -> Result<Revision, CoordinatorError> {
        self.current_revision
            .checked_next()
            .ok_or(CoordinatorError::RevisionExhausted)
    }

    fn finish_execution(
        &mut self,
        grant: CommitGrant,
        next_revision: Revision,
        now_ms: u64,
        execution: Result<Value, String>,
    ) -> Result<ExecutionResult, CoordinatorError> {
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

/// Errors raised by the coordinator before an execution result is available.
#[derive(Debug)]
pub enum CoordinatorError {
    RunMismatch { expected: RunId, actual: RunId },
    Admission(AdmissionError),
    Draft(EventDraftError),
    Ledger(LedgerError),
    Serialize(serde_json::Error),
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
            Self::Admission(error) => error.fmt(formatter),
            Self::Draft(error) => error.fmt(formatter),
            Self::Ledger(error) => error.fmt(formatter),
            Self::Serialize(error) => {
                write!(formatter, "failed to serialize runtime event: {error}")
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
            Self::RunMismatch { .. } | Self::RevisionExhausted => None,
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
    fn recovery_rebuilds_revision_and_duplicate_admission_is_idempotent() {
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
        let request = request(run_id);
        fresh
            .admit(request.clone(), 1_000)
            .expect("first admission should succeed");
        let duplicate = fresh
            .admit(request, 1_000)
            .expect_err("same idempotency key must not be admitted twice");
        assert!(matches!(
            duplicate,
            CoordinatorError::Ledger(LedgerError::DuplicateIdempotencyKey(_))
        ));
    }
}
