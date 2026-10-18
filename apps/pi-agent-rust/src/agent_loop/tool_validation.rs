//! Validation of assistant tool-call batches before admission.

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
