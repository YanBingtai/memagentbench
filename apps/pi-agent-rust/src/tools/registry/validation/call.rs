//! Validation and parsing of model tool-call envelopes.

use serde_json::Value;

use super::super::super::limits::MAX_TOOL_ARGUMENT_BYTES;
use super::super::{PreparedToolCall, ToolError, ToolRegistry};
use crate::protocol::ToolCall;

impl ToolRegistry {
    pub(in crate::tools::registry) fn prepare_call(
        &self,
        call: &ToolCall,
    ) -> Result<PreparedToolCall, ToolError> {
        validate_tool_call(call)?;
        if call.function.arguments.len() > MAX_TOOL_ARGUMENT_BYTES {
            return Err(ToolError::ArgumentsTooLarge {
                name: call.function.name.clone(),
                size: call.function.arguments.len(),
                limit: MAX_TOOL_ARGUMENT_BYTES,
            });
        }
        let arguments: Value =
            serde_json::from_str(&call.function.arguments).map_err(|source| {
                ToolError::InvalidArguments {
                    name: call.function.name.clone(),
                    source,
                }
            })?;
        self.prepare_parsed_call(call, arguments)
    }
}

pub(super) fn validate_tool_call(call: &ToolCall) -> Result<(), ToolError> {
    if call.id.trim().is_empty() {
        return Err(ToolError::InvalidCall {
            call_id: call.id.clone(),
            reason: "call id must not be empty".to_string(),
        });
    }
    if call.call_type != "function" {
        return Err(ToolError::InvalidCall {
            call_id: call.id.clone(),
            reason: format!("unsupported call type {:?}", call.call_type),
        });
    }
    if call.function.name.trim().is_empty() {
        return Err(ToolError::InvalidCall {
            call_id: call.id.clone(),
            reason: "function name must not be empty".to_string(),
        });
    }
    Ok(())
}

pub(in crate::tools::registry) fn validate_tool_name(name: &str) -> Result<(), String> {
    if !(1..=64).contains(&name.len())
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        return Err(
            "function name must be 1-64 ASCII characters using only letters, numbers, '_' or '-'"
                .to_string(),
        );
    }
    Ok(())
}
