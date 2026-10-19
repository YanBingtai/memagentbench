//! Execute a streaming request and aggregate its SSE response.

use crate::model::{
    api::{DeltaCallback, ReasoningDeltaCallback, ToolCallDeltaCallback},
    error::ModelError,
    limits::MAX_ERROR_BODY_BYTES,
    sse::{
        api_error_message, normalize_inline_thinking, read_bounded_body, SseDecoder,
        StreamAggregate,
    },
};

use crate::protocol::Message;

use super::super::ModelClient;

/// Send the request, decode SSE events, and produce the canonical assistant message.
pub(super) async fn execute<'a>(
    client: &'a ModelClient,
    builder: reqwest::RequestBuilder,
    enable_thinking: Option<bool>,
    on_delta: DeltaCallback<'a>,
    on_reasoning: ReasoningDeltaCallback<'a>,
    on_tool_call: ToolCallDeltaCallback<'a>,
) -> Result<Message, ModelError> {
    let mut response = builder.send().await.map_err(ModelError::Http)?;
    let status = response.status();
    if !status.is_success() {
        let body = read_bounded_body(response, MAX_ERROR_BODY_BYTES).await?;
        return Err(ModelError::Api {
            status,
            message: api_error_message(&body),
        });
    }

    let mut decoder = SseDecoder::default();
    let mut aggregate = StreamAggregate::default();
    while let Some(chunk) = response.chunk().await.map_err(ModelError::Http)? {
        for payload in decoder.push(&chunk)? {
            if payload == "[DONE]" {
                aggregate.done_marker = true;
                break;
            }
            aggregate.apply_json_with_reasoning_and_tools(
                &payload,
                on_delta,
                on_reasoning,
                on_tool_call,
            )?;
        }
        if aggregate.done_marker {
            break;
        }
    }
    decoder.finish()?;

    if !aggregate.done_marker {
        return Err(ModelError::StreamIncomplete);
    }
    let (message, usage) = aggregate.finish_with_usage()?;
    client.store_last_usage(usage);
    Ok(if enable_thinking == Some(true) {
        normalize_inline_thinking(message)
    } else {
        message
    })
}
