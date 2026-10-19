use std::collections::BTreeMap;

use crate::{
    model::error::ModelError,
    protocol::{FunctionCall, ToolCall},
};

use super::StreamToolAccumulator;

/// Convert accumulated fragments into canonical protocol tool calls.
pub(crate) fn finish(
    tools: BTreeMap<usize, StreamToolAccumulator>,
    finish_reason: Option<&str>,
) -> Result<Option<Vec<ToolCall>>, ModelError> {
    let incomplete_tool_metadata = tools.values().any(|tool| {
        tool.id.is_empty()
            || tool.call_type.is_empty()
            || tool.call_type != "function"
            || tool.name.is_empty()
    });
    let discard_incomplete_tools =
        incomplete_tool_metadata && matches!(finish_reason, Some("length" | "content_filter"));
    if tools.is_empty() || discard_incomplete_tools {
        return Ok(None);
    }

    let mut calls = Vec::with_capacity(tools.len());
    for (_, tool) in tools {
        if tool.id.is_empty() || tool.call_type.is_empty() || tool.name.is_empty() {
            return Err(ModelError::StreamProtocol(
                "tool call is missing id, type, or function name".to_string(),
            ));
        }
        if tool.call_type != "function" {
            return Err(ModelError::StreamProtocol(format!(
                "unsupported tool call type {:?}",
                tool.call_type
            )));
        }
        calls.push(ToolCall {
            id: tool.id,
            call_type: tool.call_type,
            function: FunctionCall {
                name: tool.name,
                arguments: tool.arguments,
            },
        });
    }
    Ok(Some(calls))
}
