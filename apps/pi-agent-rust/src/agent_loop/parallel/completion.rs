//! Batch state and terminal result exposed to the turn coordinator.

use crate::{message::Message, tools::ToolResult};

use super::super::AgentError;

/// Result of one parallel tool batch after all terminal writes are complete.
///
/// The turn coordinator uses this to preserve the existing lifecycle: it
/// emits `TurnFinished`, then either returns a terminal result or asks for the
/// next model turn.
pub(in crate::agent_loop) struct ParallelBatchOutcome {
    pub(in crate::agent_loop) all_tools_terminate: bool,
    pub(in crate::agent_loop) last_tool_message: Option<Message>,
    pub(in crate::agent_loop) terminal_error: Option<AgentError>,
}

/// Mutable state accumulated while chunks are drained in tool-call order.
pub(super) struct ParallelBatchState {
    all_tools_terminate: bool,
    last_tool_message: Option<Message>,
    terminal_error: Option<AgentError>,
}

impl ParallelBatchState {
    pub(super) fn new() -> Self {
        Self {
            all_tools_terminate: true,
            last_tool_message: None,
            terminal_error: None,
        }
    }

    pub(super) fn record_result(
        &mut self,
        result: &ToolResult,
        message: Message,
        stop_error: Option<AgentError>,
    ) {
        if !result.terminate {
            self.all_tools_terminate = false;
        }
        self.last_tool_message = Some(message);
        self.record_error(stop_error);
    }

    pub(super) fn record_cancelled_message(&mut self, message: Option<Message>) {
        self.all_tools_terminate = false;
        if let Some(message) = message {
            self.last_tool_message = Some(message);
        }
    }

    pub(super) fn record_error(&mut self, error: Option<AgentError>) {
        if self.terminal_error.is_none() {
            self.terminal_error = error;
        }
    }

    pub(super) fn has_terminal_error(&self) -> bool {
        self.terminal_error.is_some()
    }

    pub(super) fn into_outcome(self) -> ParallelBatchOutcome {
        ParallelBatchOutcome {
            all_tools_terminate: self.all_tools_terminate,
            last_tool_message: self.last_tool_message,
            terminal_error: self.terminal_error,
        }
    }
}
