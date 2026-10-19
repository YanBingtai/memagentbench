//! Data passed through one sequential tool-call lifecycle.

use std::time::Duration;

use pi_agent_runtime::{EventLedger, RunId, RuntimeCoordinator};
use serde_json::Value;

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    tools::PreparedToolCall,
};

use super::super::super::{AgentError, CancellationToken, RunStats};

/// Parsed arguments and the registry preparation result for one model call.
pub(in crate::agent_loop::sequential) type PreflightCall =
    (Value, Option<PreparedToolCall>, Option<String>);

pub(in crate::agent_loop::sequential) struct ToolStepOutcome {
    pub(in crate::agent_loop::sequential) terminate: bool,
    pub(in crate::agent_loop::sequential) message: Message,
    pub(in crate::agent_loop::sequential) stop: Option<(AgentError, String)>,
}

pub(in crate::agent_loop::sequential) struct ToolStepRequest<'a> {
    pub(in crate::agent_loop::sequential) context: RunContext,
    pub(in crate::agent_loop::sequential) step: usize,
    pub(in crate::agent_loop::sequential) tool_calls: &'a [ToolCall],
    pub(in crate::agent_loop::sequential) tool_index: usize,
    pub(in crate::agent_loop::sequential) tool_call: &'a ToolCall,
    pub(in crate::agent_loop::sequential) preflight: PreflightCall,
    pub(in crate::agent_loop::sequential) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(in crate::agent_loop::sequential) cancellation: &'a CancellationToken,
    pub(in crate::agent_loop::sequential) runtime_run_id: RunId,
    pub(in crate::agent_loop::sequential) messages: &'a mut Vec<Message>,
    pub(in crate::agent_loop::sequential) transcript: &'a mut Vec<Message>,
    pub(in crate::agent_loop::sequential) turn_tool_results: &'a mut Vec<Message>,
    pub(in crate::agent_loop::sequential) stats: &'a mut RunStats,
    pub(in crate::agent_loop::sequential) runtime:
        &'a mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
}
