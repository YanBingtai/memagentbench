//! Tool-call and argument validation before admission or execution.

use super::super::{
    limits::*,
    schema::{argument_depth, validate_schema_arguments},
};
use super::*;

impl ToolRegistry {
    /// Validate one already-parsed call before hooks or runtime admission.
    pub fn validate_arguments(&self, call: &ToolCall, arguments: &Value) -> Result<(), ToolError> {
        self.prepare_parsed_call(call, arguments.clone())
            .map(|_| ())
    }

    /// Prepare a call after its JSON arguments have already been parsed.
    pub fn prepare_parsed_call(
        &self,
        call: &ToolCall,
        arguments: Value,
    ) -> Result<PreparedToolCall, ToolError> {
        validate_tool_call(call)?;
        if !arguments.is_object() {
            return Err(ToolError::InvalidCall {
                call_id: call.id.clone(),
                reason: "tool arguments must be a JSON object".to_string(),
            });
        }
        let registered =
            self.tools
                .get(&call.function.name)
                .ok_or_else(|| ToolError::UnknownTool {
                    name: call.function.name.clone(),
                })?;
        let tool = Arc::clone(&registered.tool);
        let argument_bytes = serde_json::to_vec(&arguments)
            .map_err(|source| ToolError::InvalidArguments {
                name: call.function.name.clone(),
                source,
            })?
            .len();
        if argument_bytes > MAX_TOOL_ARGUMENT_BYTES {
            return Err(ToolError::ArgumentsTooLarge {
                name: call.function.name.clone(),
                size: argument_bytes,
                limit: MAX_TOOL_ARGUMENT_BYTES,
            });
        }
        if argument_depth(&arguments, 0) > MAX_TOOL_ARGUMENT_DEPTH {
            return Err(ToolError::ArgumentsTooDeep {
                name: call.function.name.clone(),
                limit: MAX_TOOL_ARGUMENT_DEPTH,
            });
        }
        tool.validate_arguments(&arguments)?;
        validate_schema_arguments(&registered.definition, &arguments)?;
        Ok(PreparedToolCall { tool, arguments })
    }

    /// Prepare a call originating from the model.
    ///
    /// Trusted in-process callers may still use [`Self::prepare_parsed_call`]
    /// for hidden capabilities, but model output is restricted to the
    /// advertised tool surface.
    pub fn prepare_model_call(
        &self,
        call: &ToolCall,
        arguments: Value,
    ) -> Result<PreparedToolCall, ToolError> {
        if self.exposure(&call.function.name) == Some(ToolExposure::Hidden) {
            return Err(ToolError::ToolNotModelVisible {
                name: call.function.name.clone(),
            });
        }
        self.prepare_parsed_call(call, arguments)
    }

    pub(super) fn prepare_call(&self, call: &ToolCall) -> Result<PreparedToolCall, ToolError> {
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

fn validate_tool_call(call: &ToolCall) -> Result<(), ToolError> {
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

pub(super) fn validate_tool_name(name: &str) -> Result<(), String> {
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
