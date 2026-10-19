use std::path::PathBuf;

use crate::session::SessionStore;

use super::super::super::AgentLoop;

impl<M> AgentLoop<M> {
    /// Attach durable transcript storage to this agent.
    pub fn with_session(mut self, session: SessionStore) -> Self {
        self.session = Some(session);
        self
    }

    /// Persist runtime action events in the supplied JSONL ledger.
    pub fn with_runtime_ledger_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.runtime_ledger_file = Some(path.into());
        self
    }

    /// Allow tools classified as irreversible to pass runtime admission.
    ///
    /// This is intentionally opt-in. A shell can mutate files, start
    /// processes, or affect external systems, so callers should enable this
    /// only when the surrounding environment supplies the required approval
    /// or sandbox boundary.
    pub fn with_allow_irreversible_tools(mut self, allow: bool) -> Self {
        self.allow_irreversible_tools = allow;
        self
    }

    /// Resume the latest runtime run represented by the configured ledger.
    ///
    /// The default starts a fresh runtime run for each invocation. Recovery is
    /// opt-in because an unresolved external action must be reconciled before
    /// a new tool call is allowed to proceed.
    pub fn with_runtime_recovery(mut self) -> Self {
        self.runtime_recovery = true;
        self
    }
}
