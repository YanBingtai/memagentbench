use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Durable provider-neutral retrieval evidence carried by a memory decision.
///
/// Runtime ledger events remain the source of truth. The alias keeps the
/// event-facing API discoverable without making the core agent depend on a
/// concrete memory backend.
pub type MemoryTrace = pi_agent_runtime::MemoryRetrievalSummary;

/// Identifies one logical execution and lets consumers correlate its events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RunContext {
    /// The durable conversation, when this run is attached to one.
    pub session_id: Option<Uuid>,
    /// The process-level execution of one or more turns.
    pub run_id: Uuid,
    /// The user turn currently being processed.
    pub turn_id: Uuid,
}

impl RunContext {
    /// Create a fresh run and turn context for an optional session.
    pub fn new(session_id: Option<Uuid>) -> Self {
        Self {
            session_id,
            run_id: Uuid::new_v4(),
            turn_id: Uuid::new_v4(),
        }
    }
}

/// Terminal state of one agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    Completed,
    Failed,
    Cancelled,
    Unknown,
}
