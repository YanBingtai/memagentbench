//! Typed proposals for actions that may eventually reach a tool gateway.
//!
//! An action proposal describes intent and the state it was based on. It does
//! not authorize execution. The runtime must re-check the revision, digest,
//! policy, and risk before sending anything to a tool.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Identifies one agent run across proposals, events, and checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(Uuid);

impl RunId {
    /// Creates a new run identifier.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Wraps an existing UUID, useful when recovering a persisted run.
    pub const fn from_uuid(value: Uuid) -> Self {
        Self(value)
    }

    /// Returns the underlying UUID for storage or integration boundaries.
    pub const fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

/// Identifies one proposed action within a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ActionId(Uuid);

impl ActionId {
    /// Creates a new action identifier.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Wraps an existing UUID, useful when loading an event ledger.
    pub const fn from_uuid(value: Uuid) -> Self {
        Self(value)
    }

    /// Returns the underlying UUID for storage or integration boundaries.
    pub const fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for ActionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Monotonically increasing version of the runtime state for one run.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Revision(u64);

impl Revision {
    /// The revision before the first state-changing event.
    pub const INITIAL: Self = Self(0);

    /// Creates a revision from a persisted numeric value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric representation used by ledgers and logs.
    pub const fn value(self) -> u64 {
        self.0
    }

    /// Advances the revision, returning `None` only if the counter is exhausted.
    pub const fn checked_next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

/// Risk class used by the runtime before dispatching a proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskClass {
    /// Reading state without changing an external system.
    ReadOnly,
    /// A change that the runtime can undo locally or through a defined inverse.
    Reversible,
    /// A change that needs an explicit compensating action if it fails.
    Compensatable,
    /// A change that cannot be reliably undone once sent.
    Irreversible,
    /// The runtime cannot establish the action's side-effect behavior.
    Unknown,
}

impl RiskClass {
    /// Returns whether dispatch must pass through the strong commit barrier.
    pub const fn requires_barrier(self) -> bool {
        matches!(self, Self::Irreversible | Self::Unknown)
    }
}

/// A proposed tool call tied to a specific runtime state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionProposal {
    pub run_id: RunId,
    pub action_id: ActionId,
    pub base_revision: Revision,
    pub tool_name: String,
    pub arguments: Value,
    /// Digest of canonicalized `arguments`, checked again before dispatch.
    pub arguments_digest: String,
    pub risk: RiskClass,
    /// Stable key used to prevent duplicate external operations on retry.
    pub idempotency_key: String,
    /// Timestamp supplied by the runtime clock, in milliseconds.
    pub created_at_ms: u64,
    /// Optional expiry after which this proposal must be recomputed.
    pub expires_at_ms: Option<u64>,
}

impl ActionProposal {
    /// Returns whether the proposal is expired at the supplied runtime time.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.expires_at_ms
            .is_some_and(|expires_at| now_ms >= expires_at)
    }

    /// Returns whether dispatch requires the strong commit barrier.
    pub const fn requires_barrier(&self) -> bool {
        self.risk.requires_barrier()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revision_is_ordered_and_checked() {
        let revision = Revision::new(41);

        assert_eq!(Revision::INITIAL.value(), 0);
        assert!(revision > Revision::INITIAL);
        assert_eq!(revision.checked_next(), Some(Revision::new(42)));
        assert_eq!(Revision::new(u64::MAX).checked_next(), None);
    }

    #[test]
    fn risk_class_marks_only_uncertain_or_irreversible_actions_for_barrier() {
        assert!(!RiskClass::ReadOnly.requires_barrier());
        assert!(!RiskClass::Reversible.requires_barrier());
        assert!(!RiskClass::Compensatable.requires_barrier());
        assert!(RiskClass::Irreversible.requires_barrier());
        assert!(RiskClass::Unknown.requires_barrier());
    }

    #[test]
    fn action_proposal_exposes_revision_digest_and_expiry() {
        let proposal = ActionProposal {
            run_id: RunId::new(),
            action_id: ActionId::new(),
            base_revision: Revision::new(17),
            tool_name: "publish_report".to_string(),
            arguments: serde_json::json!({"format": "pdf"}),
            arguments_digest: "sha256:example".to_string(),
            risk: RiskClass::Irreversible,
            idempotency_key: "run-1/action-1".to_string(),
            created_at_ms: 1_000,
            expires_at_ms: Some(1_500),
        };

        assert_eq!(proposal.base_revision, Revision::new(17));
        assert!(proposal.requires_barrier());
        assert!(!proposal.is_expired(1_499));
        assert!(proposal.is_expired(1_500));

        let json = serde_json::to_value(proposal).expect("proposal should serialize");
        assert_eq!(json["risk"], "irreversible");
        assert_eq!(json["base_revision"], 17);
        assert_eq!(json["arguments_digest"], "sha256:example");
    }
}
