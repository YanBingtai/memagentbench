//! State accumulated while a sequential batch runs.

use crate::message::Message;

use super::call::ToolStepOutcome;

/// Mutable state accumulated in call order while a sequential batch runs.
pub(in crate::agent_loop::sequential) struct BatchState {
    all_tools_terminate: bool,
    last_tool_message: Option<Message>,
}

impl BatchState {
    pub(in crate::agent_loop::sequential) fn new() -> Self {
        Self {
            all_tools_terminate: true,
            last_tool_message: None,
        }
    }

    pub(in crate::agent_loop::sequential) fn record(&mut self, outcome: ToolStepOutcome) {
        if !outcome.terminate {
            self.all_tools_terminate = false;
        }
        self.last_tool_message = Some(outcome.message);
    }

    pub(in crate::agent_loop::sequential) fn all_tools_terminate(&self) -> bool {
        self.all_tools_terminate
    }

    pub(in crate::agent_loop::sequential) fn take_last_tool_message(self) -> Message {
        self.last_tool_message
            .expect("a non-empty tool batch produces a tool message")
    }
}
