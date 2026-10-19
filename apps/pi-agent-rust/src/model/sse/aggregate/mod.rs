//! State machine for one bounded streaming model response.

use std::collections::BTreeMap;

mod apply;
mod finish;
mod protocol;
mod tool_calls;

#[allow(unused_imports)]
pub(crate) use protocol::{
    StreamChoice, StreamDelta, StreamEnvelope, StreamErrorBody, StreamFunctionDelta,
    StreamToolCallDelta,
};
pub(crate) use tool_calls::StreamToolAccumulator;

use crate::model::api::ModelUsage;

#[derive(Debug, Default)]
pub(crate) struct StreamAggregate {
    content: String,
    content_bytes: usize,
    reasoning_content: String,
    reasoning_bytes: usize,
    tools: BTreeMap<usize, StreamToolAccumulator>,
    tool_bytes: usize,
    finish_reason: Option<String>,
    usage: Option<ModelUsage>,
    /// Set only after the provider sends the SSE `[DONE]` marker.
    /// `finish_reason` is metadata for the assistant message and does not
    /// prove that the transport reached a clean terminal boundary.
    pub(crate) done_marker: bool,
}
