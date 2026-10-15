//! Side-effect-aware execution control for agent runtimes.
//!
//! This crate is intentionally independent from both the main agent
//! implementation and concrete memory backends. It owns the typed contracts
//! for proposed actions, revisions, barriers, preparation, and the event
//! ledger, while reusing shared artifact-reference types from
//! `pi-agent-memory`.

pub mod action;
pub mod admission;
pub mod coordinator;
pub mod digest;
pub mod gateway;
pub mod jsonl_ledger;
pub mod ledger;
pub mod memory_hook;
pub mod policy;
pub mod preparation;
pub mod sandbox;

pub use action::{ActionId, ActionProposal, Revision, RiskClass, RunId};
pub use admission::{
    AdmissionContext, AdmissionError, AdmissionPolicy, AdmissionRequest, Budget, BudgetDimension,
    CommitGrant, ResourceScope,
};
pub use coordinator::{
    CoordinatorError, ExecutionResult, MemoryDecisionReceipt, ReconciliationOutcome,
    RuntimeCoordinator, ACTION_ADMITTED_EVENT, ACTION_FAILED_EVENT, ACTION_RECONCILED_EVENT,
    ACTION_SUCCEEDED_EVENT, ACTION_UNKNOWN_EVENT, MEMORY_INJECTED_EVENT, MEMORY_SKIPPED_EVENT,
    PREPARATION_UPDATED_EVENT,
};
pub use digest::{bytes_digest, canonical_json_digest};
pub use gateway::{
    CommandGateway, CommandGatewayError, CommandOutput, CommandSpec, CommandSpecError, OutputStream,
};
pub use jsonl_ledger::{JsonlLedger, JsonlLedgerError};
pub use ledger::{
    AppendReceipt, DurabilityClass, EventDraft, EventDraftError, EventKind, EventLedger,
    EventSequence, InMemoryLedger, LedgerError, LedgerEvent,
};
pub use memory_hook::{
    InjectionDecision, InjectionSkipReason, MemoryCandidate, MemoryHookContext, MemoryInjection,
    MemoryProjection, MemoryProposal, MemoryProposalError, MemoryProviderFailure,
    MemoryRetrievalHit, MemoryRetrievalSummary, MemoryRevision, PolicyRevision,
};
pub use pi_agent_memory::{ArtifactRef, EvidenceLocator};
pub use policy::{
    FilesystemPolicy, NamespacePolicy, NetworkMode, PolicyError, ResourceLimits, SandboxMode,
    SandboxPolicy,
};
pub use preparation::{
    PreparationEffectClass, PreparationError, PreparationManager, PreparationOutcome,
    PreparationPlan, PreparationStatus, PreparationWorkResult,
};
pub use sandbox::{SandboxCommand, SandboxError, SandboxExecutor};

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
