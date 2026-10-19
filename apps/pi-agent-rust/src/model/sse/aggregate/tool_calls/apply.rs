use std::collections::BTreeMap;

use crate::model::{
    api::{ToolCallDelta, ToolCallDeltaCallback},
    error::ModelError,
    limits::{MAX_STREAM_TOOL_BYTES, MAX_STREAM_TOOL_COUNT, MAX_STREAM_TOOL_INDEX},
};

use super::super::protocol::StreamToolCallDelta;
use super::{
    merge::{append_fragment, merge_metadata},
    StreamToolAccumulator,
};

/// Apply one provider fragment and notify the ephemeral tool-call consumer.
pub(crate) fn apply_delta(
    tools: &mut BTreeMap<usize, StreamToolAccumulator>,
    tool_bytes: &mut usize,
    delta: StreamToolCallDelta,
    on_tool_call: ToolCallDeltaCallback<'_>,
) -> Result<(), ModelError> {
    let callback_delta = ToolCallDelta {
        index: delta.index,
        id: delta.id.clone(),
        call_type: delta.call_type.clone(),
        name: delta
            .function
            .as_ref()
            .and_then(|function| function.name.clone()),
        arguments: delta
            .function
            .as_ref()
            .and_then(|function| function.arguments.clone()),
    };
    on_tool_call(&callback_delta);

    if delta.index > MAX_STREAM_TOOL_INDEX {
        return Err(ModelError::StreamLimit {
            limit: "tool call index",
        });
    }
    if !tools.contains_key(&delta.index) && tools.len() >= MAX_STREAM_TOOL_COUNT {
        return Err(ModelError::StreamLimit {
            limit: "tool call count",
        });
    }

    let tool = tools.entry(delta.index).or_default();
    if let Some(value) = delta.id {
        merge_metadata(&mut tool.id, &value, "tool call id")?;
    }
    if let Some(value) = delta.call_type {
        merge_metadata(&mut tool.call_type, &value, "tool call type")?;
    }
    if let Some(function) = delta.function {
        if let Some(value) = function.name {
            append_fragment(
                &mut tool.name,
                &value,
                tool_bytes,
                MAX_STREAM_TOOL_BYTES,
                "tool arguments",
            )?;
        }
        if let Some(value) = function.arguments {
            super::append_bounded(
                &mut tool.arguments,
                &value,
                tool_bytes,
                MAX_STREAM_TOOL_BYTES,
                "tool arguments",
            )?;
        }
    }
    Ok(())
}
