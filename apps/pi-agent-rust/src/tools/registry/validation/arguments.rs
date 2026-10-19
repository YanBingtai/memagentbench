//! Registered-tool lookup and argument validation.

use std::sync::Arc;

use serde_json::Value;

use super::super::super::{
    limits::{MAX_TOOL_ARGUMENT_BYTES, MAX_TOOL_ARGUMENT_DEPTH},
    schema::{argument_depth, validate_schema_arguments},
};
use super::super::{PreparedToolCall, ToolError, ToolExposure, ToolRegistry};
use crate::protocol::ToolCall;

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
        super::call::validate_tool_call(call)?;
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
}
