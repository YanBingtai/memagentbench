use std::time::Duration;

use serde_json::Value;

use crate::{events::RunContext, message::ToolCall, tools::ToolResult};

use super::super::super::{AgentError, CancellationToken};

pub(in crate::agent_loop::sequential) struct AfterToolHookRequest<'a> {
    pub(in crate::agent_loop::sequential) context: RunContext,
    pub(in crate::agent_loop::sequential) step: usize,
    pub(in crate::agent_loop::sequential) tool_call: &'a ToolCall,
    pub(in crate::agent_loop::sequential) arguments: &'a Value,
    pub(in crate::agent_loop::sequential) tool_result: &'a mut ToolResult,
    pub(in crate::agent_loop::sequential) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(in crate::agent_loop::sequential) cancellation: &'a CancellationToken,
    pub(in crate::agent_loop::sequential) skip: bool,
}

pub(in crate::agent_loop::sequential) struct AfterToolHookOutcome {
    pub(in crate::agent_loop::sequential) content: String,
    pub(in crate::agent_loop::sequential) is_error: bool,
    pub(in crate::agent_loop::sequential) stop: Option<(AgentError, String)>,
}
