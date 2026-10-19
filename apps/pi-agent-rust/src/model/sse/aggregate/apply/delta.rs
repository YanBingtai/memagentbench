//! Apply content, reasoning, and tool-call deltas.

use super::super::{protocol::StreamToolCallDelta, tool_calls, StreamAggregate, StreamDelta};
use crate::model::{
    api::{DeltaCallback, ReasoningDeltaCallback, ToolCallDeltaCallback},
    error::ModelError,
    limits::MAX_STREAM_CONTENT_BYTES,
};

pub(super) fn apply_tool_delta(
    aggregate: &mut StreamAggregate,
    delta: StreamToolCallDelta,
    on_tool_call: ToolCallDeltaCallback<'_>,
) -> Result<(), ModelError> {
    tool_calls::apply_delta(
        &mut aggregate.tools,
        &mut aggregate.tool_bytes,
        delta,
        on_tool_call,
    )
}

pub(super) fn apply_delta(
    aggregate: &mut StreamAggregate,
    delta: StreamDelta,
    on_delta: DeltaCallback<'_>,
    on_reasoning: ReasoningDeltaCallback<'_>,
    on_tool_call: ToolCallDeltaCallback<'_>,
) -> Result<(), ModelError> {
    if let Some(reasoning) = delta.reasoning_content {
        tool_calls::append_bounded(
            &mut aggregate.reasoning_content,
            &reasoning,
            &mut aggregate.reasoning_bytes,
            MAX_STREAM_CONTENT_BYTES,
            "stream reasoning content",
        )?;
        on_reasoning(&reasoning);
    }
    if let Some(content) = delta.content {
        tool_calls::append_bounded(
            &mut aggregate.content,
            &content,
            &mut aggregate.content_bytes,
            MAX_STREAM_CONTENT_BYTES,
            "stream content",
        )?;
        on_delta(&content);
    }
    if let Some(tool_calls) = delta.tool_calls {
        for tool_call in tool_calls {
            aggregate.apply_tool_delta(tool_call, on_tool_call)?;
        }
    }
    Ok(())
}
