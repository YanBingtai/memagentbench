//! Registry preparation for model-provided tool calls.

use crate::{message::ToolCall, tools::ToolGateway};

use super::super::super::super::preflight_model_tool_call;
use super::request::PreflightCall;

pub(in crate::agent_loop::sequential) fn preflight_calls(
    tools: &ToolGateway,
    tool_calls: &[ToolCall],
) -> Vec<PreflightCall> {
    tool_calls
        .iter()
        .map(|tool_call| preflight_model_tool_call(tools, tool_call))
        .collect()
}
