//! Protocol validation for one assistant response.

use std::collections::BTreeSet;

use crate::message::ToolCall;

use super::AgentError;

pub(crate) fn validate_tool_call_batch(
    step: usize,
    tool_calls: &[ToolCall],
) -> Result<(), AgentError> {
    let mut call_ids = BTreeSet::new();
    for (index, call) in tool_calls.iter().enumerate() {
        if call.id.trim().is_empty() {
            return Err(AgentError::InvalidToolCall {
                step,
                index,
                reason: "call id must not be empty".to_string(),
            });
        }
        if !call_ids.insert(call.id.as_str()) {
            return Err(AgentError::InvalidToolCall {
                step,
                index,
                reason: format!("duplicate call id {:?}", call.id),
            });
        }
        if call.call_type != "function" {
            return Err(AgentError::InvalidToolCall {
                step,
                index,
                reason: format!("unsupported call type {:?}", call.call_type),
            });
        }
        if call.function.name.trim().is_empty() {
            return Err(AgentError::InvalidToolCall {
                step,
                index,
                reason: "function name must not be empty".to_string(),
            });
        }
    }
    Ok(())
}

/// Validate the optional tool-call batch returned by the model.
///
/// A response without tool calls is valid at this stage. When calls are
/// present, the per-step limit is checked before the structural call checks so
/// the model receives the same error precedence as the original turn loop.
pub(super) fn validate_assistant_tool_calls(
    step: usize,
    tool_calls: Option<&[ToolCall]>,
    limit: usize,
) -> Result<(), AgentError> {
    let Some(tool_calls) = tool_calls else {
        return Ok(());
    };

    if tool_calls.len() > limit {
        return Err(AgentError::TooManyToolCalls {
            step,
            count: tool_calls.len(),
            limit,
        });
    }

    validate_tool_call_batch(step, tool_calls)
}
