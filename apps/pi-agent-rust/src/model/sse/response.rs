use serde::Deserialize;

use crate::protocol::Message;

use super::super::{api::ModelUsage, error::ModelError, limits::*};

/// Normalize Qwen-compatible providers that put a complete thinking block in
/// `content` instead of returning the OpenAI-compatible `reasoning_content`
/// field. Some local servers omit the opening `<think>` tag but keep the
/// closing tag, so the closing marker is the reliable boundary here.
pub(crate) fn normalize_inline_thinking(mut message: Message) -> Message {
    if message.role != "assistant" {
        return message;
    }
    let Some(content) = message.content.take() else {
        return message;
    };
    let Some(end) = content.find("</think>") else {
        message.content = Some(content);
        return message;
    };

    let mut reasoning = content[..end].trim();
    if let Some(stripped) = reasoning.strip_prefix("<think>") {
        reasoning = stripped.trim();
    }
    if message.reasoning_content.is_none() && !reasoning.is_empty() {
        message.reasoning_content = Some(reasoning.to_owned());
    }

    let answer = content[end + "</think>".len()..].trim_start();
    message.content = (!answer.is_empty()).then(|| answer.to_owned());
    message
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionResponse {
    pub(crate) choices: Vec<Choice>,
    #[serde(default)]
    pub(crate) usage: Option<ModelUsage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Choice {
    pub(crate) message: Message,
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
}

pub(crate) async fn read_bounded_body(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<String, ModelError> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(ModelError::ResponseLimit { limit });
    }

    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(ModelError::Http)? {
        if body.len().saturating_add(chunk.len()) > limit {
            return Err(ModelError::ResponseLimit { limit });
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).map_err(|_| ModelError::ResponseEncoding)
}

pub(crate) fn validate_non_streaming_message(message: &Message) -> Result<(), ModelError> {
    let Some(tool_calls) = &message.tool_calls else {
        return Ok(());
    };
    if tool_calls.len() > MAX_STREAM_TOOL_COUNT {
        return Err(ModelError::ToolCallCountLimit {
            limit: MAX_STREAM_TOOL_COUNT,
        });
    }
    for call in tool_calls {
        if call.function.arguments.len() > MAX_STREAM_TOOL_BYTES {
            return Err(ModelError::ResponseLimit {
                limit: MAX_STREAM_TOOL_BYTES,
            });
        }
    }
    Ok(())
}

/// Extract a provider error message and bound an otherwise unstructured body.
pub(crate) fn api_error_message(body: &str) -> String {
    #[derive(Deserialize)]
    struct ErrorEnvelope {
        error: Option<ApiErrorBody>,
    }

    #[derive(Deserialize)]
    struct ApiErrorBody {
        message: Option<String>,
    }

    let parsed_message = serde_json::from_str::<ErrorEnvelope>(body)
        .ok()
        .and_then(|envelope| envelope.error)
        .and_then(|error| error.message);

    parsed_message.unwrap_or_else(|| truncate_for_error(body))
}

pub(crate) fn truncate_for_error(body: &str) -> String {
    let mut end = body.len().min(MAX_ERROR_BODY_BYTES);
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }

    let truncated = &body[..end];
    if end < body.len() {
        format!("{truncated}…")
    } else {
        truncated.to_string()
    }
}
