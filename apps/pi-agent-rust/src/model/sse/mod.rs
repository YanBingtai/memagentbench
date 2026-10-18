//! Compatibility facade for streaming and bounded model responses.

mod aggregate;
mod decoder;
mod response;

#[allow(unused_imports)]
pub(crate) use aggregate::{
    StreamAggregate, StreamChoice, StreamDelta, StreamEnvelope, StreamErrorBody,
    StreamFunctionDelta, StreamToolAccumulator, StreamToolCallDelta,
};
pub(crate) use decoder::SseDecoder;
pub(crate) use response::{
    api_error_message, normalize_inline_thinking, read_bounded_body,
    validate_non_streaming_message, ChatCompletionResponse, Choice,
};
