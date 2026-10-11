//! Side-effect-aware execution control for agent runtimes.
//!
//! This crate is intentionally independent from both the main agent
//! implementation and concrete memory backends. It will own the typed
//! contracts for proposed actions, revisions, barriers, and the event ledger.
//! A higher-level coordinator can compose this crate with `pi-agent-memory`
//! without making either crate depend on the other.

pub mod action;
pub mod ledger;

pub use action::{ActionId, ActionProposal, Revision, RiskClass, RunId};
pub use ledger::{
    AppendReceipt, DurabilityClass, EventDraft, EventDraftError, EventKind, EventLedger,
    EventSequence, InMemoryLedger, LedgerError, LedgerEvent,
};

/// Version of the serialized runtime control-plane contract.
///
/// This changes when the wire representation of runtime actions or events
/// becomes incompatible. It is separate from the crate version so persisted
/// ledgers can reject unknown schemas explicitly.
pub const RUNTIME_SCHEMA_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::RUNTIME_SCHEMA_VERSION;

    #[test]
    fn runtime_schema_version_is_nonzero() {
        assert_ne!(RUNTIME_SCHEMA_VERSION, 0);
    }
}
