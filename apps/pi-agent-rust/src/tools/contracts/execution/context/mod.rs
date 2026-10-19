//! Cooperative cancellation and progress reporting for one tool call.

mod identity;
mod progress;

use crate::agent_loop::CancellationToken;

use super::super::super::limits::DEFAULT_TOOL_UPDATE_BYTES;
use super::updates::ToolUpdateSink;

pub(crate) use progress::{UpdateGate, UpdateLease};

/// Cooperative cancellation and progress reporting for one tool call.
#[derive(Clone)]
pub struct ToolExecutionContext {
    cancellation: CancellationToken,
    update_sink: Option<ToolUpdateSink>,
    max_update_bytes: usize,
    update_gate: std::sync::Arc<UpdateGate>,
    parent_run_id: Option<uuid::Uuid>,
    parent_turn_id: Option<uuid::Uuid>,
    parent_step: Option<usize>,
    parent_tool_call_id: Option<String>,
    delegation_depth: u16,
}

impl ToolExecutionContext {
    pub fn new(cancellation: CancellationToken) -> Self {
        Self {
            cancellation,
            update_sink: None,
            max_update_bytes: DEFAULT_TOOL_UPDATE_BYTES,
            update_gate: std::sync::Arc::new(UpdateGate::new()),
            parent_run_id: None,
            parent_turn_id: None,
            parent_step: None,
            parent_tool_call_id: None,
            delegation_depth: 0,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Clone the run-wide cancellation signal for a delegated operation.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }
}
