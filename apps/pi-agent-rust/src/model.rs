use std::{collections::BTreeMap, future::Future, pin::Pin, time::Duration};

use reqwest::{header, StatusCode};
use serde::{Deserialize, Serialize, Serializer};
use thiserror::Error;

use crate::message::{Message, ToolCall, ToolDefinition};

const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
const MAX_ERROR_BODY_BYTES: usize = 8 * 1024;
const MAX_RESPONSE_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_STREAM_EVENT_BYTES: usize = 64 * 1024;
const MAX_STREAM_CONTENT_BYTES: usize = 4 * 1024 * 1024;
const MAX_STREAM_TOOL_BYTES: usize = 4 * 1024 * 1024;
const MAX_STREAM_TOOL_COUNT: usize = 128;
const MAX_STREAM_TOOL_INDEX: usize = 1024;

/// Errors returned while constructing a model client or calling a provider.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("invalid model endpoint: {0}")]
    InvalidEndpoint(String),

    #[error("model request timeout must be greater than zero")]
    InvalidTimeout,

    #[error("temperature must be finite and between 0.0 and 2.0")]
    InvalidTemperature,

    #[error("failed to build HTTP client: {0}")]
    BuildClient(#[source] reqwest::Error),

    #[error("model request failed: {0}")]
    Http(#[source] reqwest::Error),

    #[error("model returned HTTP {status}: {message}")]
    Api { status: StatusCode, message: String },

    #[error("model returned no choices")]
    EmptyResponse,

    #[error("failed to decode model response: {0}")]
    Decode(#[source] serde_json::Error),

    #[error("model response is not valid UTF-8")]
    ResponseEncoding,

    #[error("model response exceeded the {limit}-byte limit")]
    ResponseLimit { limit: usize },

    #[error("model response contained more than {limit} tool calls")]
    ToolCallCountLimit { limit: usize },

    #[error("model stream ended before a completion signal")]
    StreamIncomplete,

    #[error("invalid model stream: {0}")]
    StreamProtocol(String),

    #[error("model stream exceeded the {limit} limit")]
    StreamLimit { limit: &'static str },
}

impl ModelError {
    /// Whether retrying the same read-only model request may succeed later.
    ///
    /// Model requests are safe to retry because the provider call does not
    /// execute an agent tool. Decode and protocol errors remain terminal so a
    /// malformed response is not hidden by repeated requests.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Http(error) => error.is_connect() || error.is_timeout(),
            Self::Api { status, .. } => {
                status.is_server_error()
                    || matches!(
                        *status,
                        StatusCode::REQUEST_TIMEOUT
                            | StatusCode::TOO_EARLY
                            | StatusCode::TOO_MANY_REQUESTS
                    )
            }
            Self::StreamIncomplete => true,
            Self::InvalidEndpoint(_)
            | Self::InvalidTimeout
            | Self::InvalidTemperature
            | Self::BuildClient(_)
            | Self::EmptyResponse
            | Self::Decode(_)
            | Self::ResponseEncoding
            | Self::ResponseLimit { .. }
            | Self::ToolCallCountLimit { .. }
            | Self::StreamProtocol(_)
            | Self::StreamLimit { .. } => false,
        }
    }
}

/// The provider-independent input for one chat completion.
#[derive(Debug, Clone, Copy)]
pub struct ChatRequest<'a> {
    pub model: &'a str,
    pub api_key: Option<&'a str>,
    pub messages: &'a [Message],
    pub tools: &'a [ToolDefinition],
    pub options: ChatOptions,
}

/// Boxed asynchronous result used by [`ChatModel`] without an async-trait
/// dependency. The lifetime ties the request and future to the model client.
pub type ChatFuture<'a> = Pin<Box<dyn Future<Output = Result<Message, ModelError>> + Send + 'a>>;

/// Receives ephemeral text fragments from a streaming model response.
///
/// Fragments are notifications only. The returned [`Message`] remains the
/// canonical assistant message and is the only value suitable for persistence.
pub type DeltaCallback<'a> = &'a mut (dyn FnMut(&str) + Send + 'a);

/// Receives reasoning fragments from a thinking-capable provider.
pub type ReasoningDeltaCallback<'a> = &'a mut (dyn FnMut(&str) + Send + 'a);

/// One provider-emitted fragment of a streamed tool call.
///
/// `arguments` and `name` are fragments, not complete values. `id` and
/// `call_type` are metadata and may be repeated by a provider. The model
/// client still assembles the canonical [`ToolCall`] before returning; this
/// callback is only for ephemeral consumers such as a CLI or TUI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub call_type: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Receives an ephemeral streamed tool-call fragment.
pub type ToolCallDeltaCallback<'a> = &'a mut (dyn FnMut(&ToolCallDelta) + Send + 'a);

/// Provider-independent model interface used by the agent runtime.
pub trait ChatModel: Send + Sync {
    fn complete<'a>(&'a self, request: ChatRequest<'a>) -> ChatFuture<'a>;

    /// Stream text when the provider supports it, while preserving the
    /// complete-message API for simple test doubles and other providers.
    fn stream<'a>(
        &'a self,
        request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
    ) -> ChatFuture<'a> {
        Box::pin(async move {
            let message = self.complete(request).await?;
            if let Some(content) = message.content.as_deref() {
                on_delta(content);
            }
            Ok(message)
        })
    }

    /// Stream answer and reasoning fragments separately when supported.
    /// Providers without a reasoning channel retain the regular stream behavior.
    fn stream_with_reasoning<'a>(
        &'a self,
        request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
        _on_reasoning: ReasoningDeltaCallback<'a>,
    ) -> ChatFuture<'a> {
        self.stream(request, on_delta)
    }

    /// Stream answer, reasoning, and tool-call fragments separately.
    ///
    /// Providers that do not expose tool-call fragments retain the existing
    /// behavior. The final [`Message`] returned by this method remains the
    /// only canonical value suitable for persistence.
    fn stream_with_reasoning_and_tools<'a>(
        &'a self,
        request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
        on_reasoning: ReasoningDeltaCallback<'a>,
        _on_tool_call: ToolCallDeltaCallback<'a>,
    ) -> ChatFuture<'a> {
        self.stream_with_reasoning(request, on_delta, on_reasoning)
    }
}

/// A small OpenAI-compatible chat-completions client.
///
/// The client owns transport configuration, while the agent owns conversation
/// state and decides when a request should be made. Keeping these concerns
/// separate lets us add other providers without changing the agent loop.
#[derive(Clone, Debug)]
pub struct ModelClient {
    http: reqwest::Client,
    endpoint: String,
    tool_choice: ToolChoice,
}

impl ModelClient {
    /// Build a client from a provider base URL such as `http://localhost:8000/v1`.
    pub fn new(base_url: &str, timeout: Duration) -> Result<Self, ModelError> {
        let base_url = base_url.trim_end_matches('/');
        if base_url.is_empty() || !base_url.contains("://") {
            return Err(ModelError::InvalidEndpoint(base_url.to_string()));
        }
        if timeout.is_zero() {
            return Err(ModelError::InvalidTimeout);
        }

        let mut default_headers = header::HeaderMap::new();
        default_headers.insert(
            header::ACCEPT,
            header::HeaderValue::from_static("application/json"),
        );

        let http = reqwest::Client::builder()
            .default_headers(default_headers)
            .timeout(timeout)
            .build()
            .map_err(ModelError::BuildClient)?;

        Ok(Self {
            http,
            endpoint: format!("{base_url}{CHAT_COMPLETIONS_PATH}"),
            tool_choice: ToolChoice::Auto,
        })
    }

    /// Select how the provider should handle advertised tools.
    ///
    /// `Auto` preserves agent behavior. `None` is useful for providers that
    /// accept tool definitions but do not enable automatic tool selection.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Send the current conversation and return the assistant message.
    pub async fn complete(
        &self,
        model: &str,
        api_key: Option<&str>,
        messages: &[Message],
        tools: &[ToolDefinition],
        temperature: f32,
    ) -> Result<Message, ModelError> {
        self.complete_with_options(
            model,
            api_key,
            messages,
            tools,
            ChatOptions {
                temperature,
                enable_thinking: None,
            },
        )
        .await
    }

    /// Send a chat request with provider-specific generation options.
    pub async fn complete_with_options(
        &self,
        model: &str,
        api_key: Option<&str>,
        messages: &[Message],
        tools: &[ToolDefinition],
        options: ChatOptions,
    ) -> Result<Message, ModelError> {
        validate_chat_options(options)?;
        let provider_messages = provider_messages(messages, options.enable_thinking == Some(true));
        let request = ChatCompletionRequest {
            model,
            messages: &provider_messages,
            tools: (!tools.is_empty()).then_some(tools),
            tool_choice: (!tools.is_empty()).then_some(self.tool_choice.clone()),
            temperature: options.temperature,
            stream: None,
            chat_template_kwargs: options
                .enable_thinking
                .map(|enable_thinking| ChatTemplateKwargs { enable_thinking }),
        };

        let mut builder = self.http.post(&self.endpoint).json(&request);
        if let Some(api_key) = api_key.filter(|key| !key.is_empty()) {
            builder = builder.bearer_auth(api_key);
        }

        let response = builder.send().await.map_err(ModelError::Http)?;
        let status = response.status();
        let body = read_bounded_body(response, MAX_RESPONSE_BODY_BYTES).await?;

        if !status.is_success() {
            return Err(ModelError::Api {
                status,
                message: api_error_message(&body),
            });
        }

        let parsed: ChatCompletionResponse =
            serde_json::from_str(&body).map_err(ModelError::Decode)?;
        let message = parsed
            .choices
            .into_iter()
            .next()
            .map(|choice| {
                let Choice {
                    mut message,
                    finish_reason,
                } = choice;
                if finish_reason.is_some() {
                    message.stop_reason = finish_reason;
                }
                message
            })
            .ok_or(ModelError::EmptyResponse)?;
        validate_non_streaming_message(&message)?;
        Ok(message)
    }

    /// Send an OpenAI-compatible SSE request and aggregate its final message.
    ///
    /// HTTP chunks are not SSE events, and tool arguments are not guaranteed to
    /// arrive in one piece. This method therefore buffers by SSE event boundary
    /// and aggregates each tool call by its provider-supplied index before
    /// returning a message to the agent loop.
    pub async fn complete_streaming<'a>(
        &'a self,
        request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
    ) -> Result<Message, ModelError> {
        fn ignore_reasoning(_: &str) {}
        fn ignore_tool_call(_: &ToolCallDelta) {}
        let mut ignored = ignore_reasoning;
        let mut ignored_tool_call = ignore_tool_call;
        self.complete_streaming_with_reasoning_and_tools(
            request,
            on_delta,
            &mut ignored,
            &mut ignored_tool_call,
        )
        .await
    }

    /// Streaming completion variant that exposes reasoning fragments.
    pub async fn complete_streaming_with_reasoning<'a>(
        &'a self,
        request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
        on_reasoning: ReasoningDeltaCallback<'a>,
    ) -> Result<Message, ModelError> {
        fn ignore_tool_call(_: &ToolCallDelta) {}
        let mut ignored_tool_call = ignore_tool_call;
        self.complete_streaming_with_reasoning_and_tools(
            request,
            on_delta,
            on_reasoning,
            &mut ignored_tool_call,
        )
        .await
    }

    /// Streaming completion variant that also exposes tool-call fragments.
    pub async fn complete_streaming_with_reasoning_and_tools<'a>(
        &'a self,
        request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
        on_reasoning: ReasoningDeltaCallback<'a>,
        on_tool_call: ToolCallDeltaCallback<'a>,
    ) -> Result<Message, ModelError> {
        validate_chat_options(request.options)?;
        let provider_messages = provider_messages(
            request.messages,
            request.options.enable_thinking == Some(true),
        );
        let body = ChatCompletionRequest {
            model: request.model,
            messages: &provider_messages,
            tools: (!request.tools.is_empty()).then_some(request.tools),
            tool_choice: (!request.tools.is_empty()).then_some(self.tool_choice.clone()),
            temperature: request.options.temperature,
            stream: Some(true),
            chat_template_kwargs: request
                .options
                .enable_thinking
                .map(|enable_thinking| ChatTemplateKwargs { enable_thinking }),
        };

        let mut builder = self
            .http
            .post(&self.endpoint)
            .header(header::ACCEPT, "text/event-stream")
            .json(&body);
        if let Some(api_key) = request.api_key.filter(|key| !key.is_empty()) {
            builder = builder.bearer_auth(api_key);
        }

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
        aggregate.finish()
    }
}

impl ChatModel for ModelClient {
    fn complete<'a>(&'a self, request: ChatRequest<'a>) -> ChatFuture<'a> {
        Box::pin(self.complete_with_options(
            request.model,
            request.api_key,
            request.messages,
            request.tools,
            request.options,
        ))
    }

    fn stream<'a>(
        &'a self,
        request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
    ) -> ChatFuture<'a> {
        Box::pin(self.complete_streaming(request, on_delta))
    }

    fn stream_with_reasoning<'a>(
        &'a self,
        request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
        on_reasoning: ReasoningDeltaCallback<'a>,
    ) -> ChatFuture<'a> {
        Box::pin(self.complete_streaming_with_reasoning(request, on_delta, on_reasoning))
    }

    fn stream_with_reasoning_and_tools<'a>(
        &'a self,
        request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
        on_reasoning: ReasoningDeltaCallback<'a>,
        on_tool_call: ToolCallDeltaCallback<'a>,
    ) -> ChatFuture<'a> {
        Box::pin(self.complete_streaming_with_reasoning_and_tools(
            request,
            on_delta,
            on_reasoning,
            on_tool_call,
        ))
    }
}

/// Generation settings understood by OpenAI-compatible local model servers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChatOptions {
    pub temperature: f32,
    pub enable_thinking: Option<bool>,
}

/// OpenAI-compatible policy for selecting tools during a chat completion.
///
/// `Function` uses the object form required by the Chat Completions API:
/// `{ "type": "function", "function": { "name": "..." } }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function { name: String },
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function { name } => serde_json::json!({
                "type": "function",
                "function": { "name": name },
            })
            .serialize(serializer),
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ProviderMessage<'a>],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDefinition]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
}

/// Provider-facing projection of a transcript message.
///
/// `Message` also carries agent-only tool metadata for session recovery and
/// event consumers. That metadata must not leak into an OpenAI-compatible
/// request body, so requests are serialized through this explicit projection.
#[derive(Debug, Serialize)]
struct ProviderMessage<'a> {
    role: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a String>,
    /// Qwen-style thinking providers need the prior reasoning channel to
    /// continue a tool-calling turn. Keep it opt-in so ordinary providers do
    /// not receive agent-internal reasoning content.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<&'a Vec<ToolCall>>,
}

impl<'a> ProviderMessage<'a> {
    fn new(message: &'a Message, include_reasoning: bool) -> Self {
        Self {
            role: &message.role,
            content: message.content.as_ref(),
            reasoning_content: if include_reasoning && message.role == "assistant" {
                message.reasoning_content.as_ref()
            } else {
                None
            },
            name: message.name.as_ref(),
            tool_call_id: message.tool_call_id.as_ref(),
            tool_calls: message.tool_calls.as_ref(),
        }
    }
}

fn provider_messages(messages: &[Message], include_reasoning: bool) -> Vec<ProviderMessage<'_>> {
    messages
        .iter()
        .map(|message| ProviderMessage::new(message, include_reasoning))
        .collect()
}

fn validate_chat_options(options: ChatOptions) -> Result<(), ModelError> {
    if !options.temperature.is_finite() || !(0.0..=2.0).contains(&options.temperature) {
        return Err(ModelError::InvalidTemperature);
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default)]
struct SseDecoder {
    buffer: Vec<u8>,
    data_lines: Vec<String>,
    event_bytes: usize,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, ModelError> {
        self.buffer.extend_from_slice(bytes);

        let mut payloads = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..newline).collect();
            self.buffer.drain(..1);
            if line.last() == Some(&b'\r') {
                line.pop();
            }

            let line = std::str::from_utf8(&line).map_err(|_| {
                ModelError::StreamProtocol("SSE line is not valid UTF-8".to_string())
            })?;
            if line.is_empty() {
                if !self.data_lines.is_empty() {
                    payloads.push(self.data_lines.join("\n"));
                    self.data_lines.clear();
                    self.event_bytes = 0;
                }
                continue;
            }
            if line.starts_with(':') {
                continue;
            }
            if let Some(value) = line.strip_prefix("data:") {
                let value = value.strip_prefix(' ').unwrap_or(value);
                self.event_bytes = self
                    .event_bytes
                    .checked_add(value.len() + 1)
                    .ok_or(ModelError::StreamLimit { limit: "SSE event" })?;
                if self.event_bytes > MAX_STREAM_EVENT_BYTES {
                    return Err(ModelError::StreamLimit { limit: "SSE event" });
                }
                self.data_lines.push(value.to_string());
            }
        }
        if self.buffer.len() > MAX_STREAM_EVENT_BYTES {
            return Err(ModelError::StreamLimit { limit: "SSE event" });
        }
        Ok(payloads)
    }

    fn finish(&self) -> Result<(), ModelError> {
        if self.buffer.is_empty() && self.data_lines.is_empty() {
            Ok(())
        } else {
            Err(ModelError::StreamIncomplete)
        }
    }
}

#[derive(Debug, Default)]
struct StreamToolAccumulator {
    id: String,
    call_type: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Default)]
struct StreamAggregate {
    content: String,
    content_bytes: usize,
    reasoning_content: String,
    reasoning_bytes: usize,
    tools: BTreeMap<usize, StreamToolAccumulator>,
    tool_bytes: usize,
    finish_reason: Option<String>,
    /// Set only after the provider sends the SSE `[DONE]` marker.
    /// `finish_reason` is metadata for the assistant message and does not
    /// prove that the transport reached a clean terminal boundary.
    done_marker: bool,
}

impl StreamAggregate {
    #[cfg(test)]
    fn apply_json(&mut self, payload: &str, on_delta: DeltaCallback<'_>) -> Result<(), ModelError> {
        fn ignore_reasoning(_: &str) {}
        fn ignore_tool_call(_: &ToolCallDelta) {}
        let mut ignored = ignore_reasoning;
        let mut ignored_tool_call = ignore_tool_call;
        self.apply_json_with_reasoning_and_tools(
            payload,
            on_delta,
            &mut ignored,
            &mut ignored_tool_call,
        )
    }

    #[allow(dead_code)]
    fn apply_json_with_reasoning(
        &mut self,
        payload: &str,
        on_delta: DeltaCallback<'_>,
        on_reasoning: ReasoningDeltaCallback<'_>,
    ) -> Result<(), ModelError> {
        fn ignore_tool_call(_: &ToolCallDelta) {}
        let mut ignored_tool_call = ignore_tool_call;
        self.apply_json_with_reasoning_and_tools(
            payload,
            on_delta,
            on_reasoning,
            &mut ignored_tool_call,
        )
    }

    fn apply_json_with_reasoning_and_tools(
        &mut self,
        payload: &str,
        on_delta: DeltaCallback<'_>,
        on_reasoning: ReasoningDeltaCallback<'_>,
        on_tool_call: ToolCallDeltaCallback<'_>,
    ) -> Result<(), ModelError> {
        let envelope: StreamEnvelope = serde_json::from_str(payload).map_err(ModelError::Decode)?;
        if let Some(error) = envelope.error {
            return Err(ModelError::StreamProtocol(
                error
                    .message
                    .unwrap_or_else(|| "provider returned a stream error".to_string()),
            ));
        }

        for choice in envelope.choices {
            if choice.index != 0 {
                continue;
            }
            if let Some(delta) = choice.delta {
                if let Some(reasoning) = delta.reasoning_content {
                    append_bounded(
                        &mut self.reasoning_content,
                        &reasoning,
                        &mut self.reasoning_bytes,
                        MAX_STREAM_CONTENT_BYTES,
                        "stream reasoning content",
                    )?;
                    on_reasoning(&reasoning);
                }
                if let Some(content) = delta.content {
                    append_bounded(
                        &mut self.content,
                        &content,
                        &mut self.content_bytes,
                        MAX_STREAM_CONTENT_BYTES,
                        "stream content",
                    )?;
                    on_delta(&content);
                }
                if let Some(tool_calls) = delta.tool_calls {
                    for tool_call in tool_calls {
                        self.apply_tool_delta(tool_call, on_tool_call)?;
                    }
                }
            }
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }
        }
        Ok(())
    }

    fn apply_tool_delta(
        &mut self,
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
        if !self.tools.contains_key(&delta.index) && self.tools.len() >= MAX_STREAM_TOOL_COUNT {
            return Err(ModelError::StreamLimit {
                limit: "tool call count",
            });
        }
        let tool = self.tools.entry(delta.index).or_default();
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
                    &mut self.tool_bytes,
                    MAX_STREAM_TOOL_BYTES,
                    "tool arguments",
                )?;
            }
            if let Some(value) = function.arguments {
                append_bounded(
                    &mut tool.arguments,
                    &value,
                    &mut self.tool_bytes,
                    MAX_STREAM_TOOL_BYTES,
                    "tool arguments",
                )?;
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<Message, ModelError> {
        let incomplete_tool_metadata = self.tools.values().any(|tool| {
            tool.id.is_empty()
                || tool.call_type.is_empty()
                || tool.call_type != "function"
                || tool.name.is_empty()
        });
        let discard_incomplete_tools = incomplete_tool_metadata
            && matches!(
                self.finish_reason.as_deref(),
                Some("length" | "content_filter")
            );
        let tool_calls = if self.tools.is_empty() || discard_incomplete_tools {
            None
        } else {
            let mut calls = Vec::with_capacity(self.tools.len());
            for (_, tool) in self.tools {
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
                    function: crate::message::FunctionCall {
                        name: tool.name,
                        arguments: tool.arguments,
                    },
                });
            }
            Some(calls)
        };

        let mut message = Message::assistant(
            (!self.content.is_empty()).then_some(self.content),
            tool_calls,
        );
        message.reasoning_content =
            (!self.reasoning_content.is_empty()).then_some(self.reasoning_content);
        message.stop_reason = self.finish_reason;
        Ok(message)
    }
}

async fn read_bounded_body(
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

fn validate_non_streaming_message(message: &Message) -> Result<(), ModelError> {
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

fn append_bounded(
    target: &mut String,
    value: &str,
    total: &mut usize,
    limit: usize,
    label: &'static str,
) -> Result<(), ModelError> {
    *total = total
        .checked_add(value.len())
        .ok_or(ModelError::StreamLimit { limit: label })?;
    if *total > limit {
        return Err(ModelError::StreamLimit { limit: label });
    }
    target.push_str(value);
    Ok(())
}

fn merge_metadata(target: &mut String, value: &str, field: &'static str) -> Result<(), ModelError> {
    if value.is_empty() {
        return Ok(());
    }
    if target.is_empty() {
        target.push_str(value);
    } else if target != value {
        return Err(ModelError::StreamProtocol(format!(
            "conflicting {field}: existing {:?}, received {:?}",
            target, value
        )));
    }
    Ok(())
}

fn append_fragment(
    target: &mut String,
    value: &str,
    total: &mut usize,
    limit: usize,
    label: &'static str,
) -> Result<(), ModelError> {
    if value.is_empty() || value == target {
        return Ok(());
    }
    let fragment = value.strip_prefix(target.as_str()).unwrap_or(value);
    append_bounded(target, fragment, total, limit, label)
}

#[derive(Debug, Deserialize)]
struct StreamEnvelope {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    error: Option<StreamErrorBody>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    delta: Option<StreamDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<StreamToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct StreamToolCallDelta {
    #[serde(default)]
    index: usize,
    #[serde(default, rename = "id")]
    id: Option<String>,
    #[serde(default, rename = "type")]
    call_type: Option<String>,
    #[serde(default)]
    function: Option<StreamFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct StreamFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamErrorBody {
    message: Option<String>,
}

fn api_error_message(body: &str) -> String {
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

fn truncate_for_error(body: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use serde_json::Value;
    use wiremock::{matchers::method, Mock, MockServer, ResponseTemplate};

    #[test]
    fn client_appends_chat_completions_path() {
        let client = ModelClient::new("http://localhost:6273/v1/", Duration::from_secs(30))
            .expect("endpoint should be valid");

        assert_eq!(client.endpoint, "http://localhost:6273/v1/chat/completions");
    }

    #[test]
    fn invalid_endpoint_is_rejected() {
        let error = ModelClient::new("localhost:6273", Duration::from_secs(30))
            .expect_err("endpoint without a scheme should fail");

        assert!(matches!(error, ModelError::InvalidEndpoint(_)));
    }

    #[test]
    fn zero_request_timeout_is_rejected() {
        let error = ModelClient::new("http://localhost:6273/v1", Duration::ZERO)
            .expect_err("zero timeout should fail");

        assert!(matches!(error, ModelError::InvalidTimeout));
    }

    #[test]
    fn invalid_temperature_is_rejected_before_request() {
        let error = validate_chat_options(ChatOptions {
            temperature: f32::NAN,
            enable_thinking: None,
        })
        .expect_err("NaN temperature should fail");
        assert!(matches!(error, ModelError::InvalidTemperature));

        let error = validate_chat_options(ChatOptions {
            temperature: 2.1,
            enable_thinking: None,
        })
        .expect_err("temperature above provider range should fail");
        assert!(matches!(error, ModelError::InvalidTemperature));
    }

    #[test]
    fn retryability_is_limited_to_transient_provider_failures() {
        assert!(ModelError::Api {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "busy".to_string(),
        }
        .is_retryable());
        assert!(ModelError::Api {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "overloaded".to_string(),
        }
        .is_retryable());
        assert!(ModelError::StreamIncomplete.is_retryable());
        assert!(!ModelError::Api {
            status: StatusCode::BAD_REQUEST,
            message: "invalid".to_string(),
        }
        .is_retryable());
        assert!(
            !ModelError::Decode(serde_json::from_str::<Value>("{").expect_err("invalid JSON"))
                .is_retryable()
        );
    }

    #[tokio::test]
    async fn model_client_implements_provider_independent_trait() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "trait adapter works"
                    }
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = ModelClient::new(&server.uri(), Duration::from_secs(5))
            .expect("endpoint should be valid");
        let messages = [Message::user("hello")];
        let response = ChatModel::complete(
            &client,
            ChatRequest {
                model: "test-model",
                api_key: None,
                messages: &messages,
                tools: &[],
                options: ChatOptions {
                    temperature: 0.0,
                    enable_thinking: Some(false),
                },
            },
        )
        .await
        .expect("trait adapter should return the model response");

        assert_eq!(response.content.as_deref(), Some("trait adapter works"));
    }

    #[tokio::test]
    async fn non_streaming_finish_reason_is_preserved() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "partial"
                    },
                    "finish_reason": "length"
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = ModelClient::new(&server.uri(), Duration::from_secs(5))
            .expect("endpoint should be valid");
        let messages = [Message::user("hello")];
        let response = client
            .complete_with_options(
                "test-model",
                None,
                &messages,
                &[],
                ChatOptions {
                    temperature: 0.0,
                    enable_thinking: Some(false),
                },
            )
            .await
            .expect("completion should decode");

        assert_eq!(response.stop_reason.as_deref(), Some("length"));
    }

    #[tokio::test]
    async fn non_streaming_response_body_is_bounded() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string("x".repeat(MAX_RESPONSE_BODY_BYTES + 1)),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = ModelClient::new(&server.uri(), Duration::from_secs(5))
            .expect("endpoint should be valid");
        let messages = [Message::user("hello")];
        let error = client
            .complete_with_options(
                "test-model",
                None,
                &messages,
                &[],
                ChatOptions {
                    temperature: 0.0,
                    enable_thinking: Some(false),
                },
            )
            .await
            .expect_err("oversized response should be rejected");

        assert!(matches!(
            error,
            ModelError::ResponseLimit {
                limit: MAX_RESPONSE_BODY_BYTES
            }
        ));
    }

    #[test]
    fn non_streaming_tool_arguments_are_bounded() {
        let message = Message::assistant(
            None,
            Some(vec![ToolCall {
                id: "call-1".to_string(),
                call_type: "function".to_string(),
                function: crate::message::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "x".repeat(MAX_STREAM_TOOL_BYTES + 1),
                },
            }]),
        );

        let error = validate_non_streaming_message(&message)
            .expect_err("oversized tool arguments should be rejected");

        assert!(matches!(
            error,
            ModelError::ResponseLimit {
                limit: MAX_STREAM_TOOL_BYTES
            }
        ));
    }

    #[tokio::test]
    async fn streaming_response_assembles_text_and_split_tool_arguments() {
        let server = MockServer::start().await;
        let body = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"你\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"Ca\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"好\",\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"rgo.toml\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = ModelClient::new(&server.uri(), Duration::from_secs(5))
            .expect("endpoint should be valid");
        let messages = [Message::user("stream")];
        let mut deltas = Vec::new();
        let response = ChatModel::stream(
            &client,
            ChatRequest {
                model: "test-model",
                api_key: None,
                messages: &messages,
                tools: &[],
                options: ChatOptions {
                    temperature: 0.0,
                    enable_thinking: Some(false),
                },
            },
            &mut |delta| deltas.push(delta.to_string()),
        )
        .await
        .expect("stream should complete");

        assert_eq!(deltas, ["你", "好"]);
        assert_eq!(response.content.as_deref(), Some("你好"));
        let call = &response.tool_calls.as_ref().expect("tool call")[0];
        assert_eq!(call.id, "call-1");
        assert_eq!(call.function.name, "read_file");
        assert_eq!(call.function.arguments, r#"{"path":"Cargo.toml"}"#);
    }

    #[tokio::test]
    async fn streaming_eof_without_done_is_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"stop\"}]}\n\n",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = ModelClient::new(&server.uri(), Duration::from_secs(5))
            .expect("endpoint should be valid");
        let messages = [Message::user("stream")];
        let error = client
            .complete_streaming(
                ChatRequest {
                    model: "test-model",
                    api_key: None,
                    messages: &messages,
                    tools: &[],
                    options: ChatOptions {
                        temperature: 0.0,
                        enable_thinking: None,
                    },
                },
                &mut |_| {},
            )
            .await
            .expect_err("missing completion marker should fail");

        assert!(matches!(error, ModelError::StreamIncomplete));
    }

    #[test]
    fn sse_decoder_handles_crlf_and_split_utf8() {
        let mut decoder = SseDecoder::default();
        let bytes = "data: 你好\r\n\r\n".as_bytes();
        let split = "data: ".len() + 1;
        assert!(decoder.push(&bytes[..split]).unwrap().is_empty());
        let payloads = decoder.push(&bytes[split..]).unwrap();

        assert_eq!(payloads, ["你好"]);
        decoder.finish().expect("complete event should be flushed");
    }

    #[test]
    fn repeated_tool_metadata_is_not_appended() {
        let mut aggregate = StreamAggregate::default();
        let first = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\\\"path\\\": \\\""}
                }]}
            }]
        });
        let second = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {"arguments": "probe.txt\\\"}"}
                }]}
            }]
        });
        fn ignore_delta(_: &str) {}
        let mut on_delta = ignore_delta;
        aggregate
            .apply_json(&first.to_string(), &mut on_delta)
            .expect("first tool chunk should parse");
        aggregate
            .apply_json(&second.to_string(), &mut on_delta)
            .expect("repeated metadata should be ignored");

        let message = aggregate.finish().expect("tool call should be complete");
        let call = &message.tool_calls.expect("tool call should exist")[0];
        assert_eq!(call.id, "call-1");
        assert_eq!(call.call_type, "function");
        assert_eq!(call.function.name, "read_file");
        assert_eq!(call.function.arguments, r#"{\"path\": \"probe.txt\"}"#);
    }

    #[test]
    fn streaming_tool_call_deltas_are_forwarded_before_aggregation() {
        let mut aggregate = StreamAggregate::default();
        let first = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"Ca"}
                }]}
            }]
        });
        let second = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "rgo.toml\"}"}
                }]}
            }]
        });
        fn ignore_text(_: &str) {}
        let mut text = ignore_text;
        let mut reasoning = ignore_text;
        let mut deltas = Vec::new();
        let mut on_tool_call = |delta: &ToolCallDelta| deltas.push(delta.clone());

        aggregate
            .apply_json_with_reasoning_and_tools(
                &first.to_string(),
                &mut text,
                &mut reasoning,
                &mut on_tool_call,
            )
            .expect("first tool chunk should parse");
        aggregate
            .apply_json_with_reasoning_and_tools(
                &second.to_string(),
                &mut text,
                &mut reasoning,
                &mut on_tool_call,
            )
            .expect("second tool chunk should parse");

        assert_eq!(
            deltas,
            vec![
                ToolCallDelta {
                    index: 0,
                    id: Some("call-1".to_string()),
                    call_type: Some("function".to_string()),
                    name: Some("read_file".to_string()),
                    arguments: Some("{\"path\":\"Ca".to_string()),
                },
                ToolCallDelta {
                    index: 0,
                    id: None,
                    call_type: None,
                    name: None,
                    arguments: Some("rgo.toml\"}".to_string()),
                },
            ]
        );

        let message = aggregate.finish().expect("tool call should be complete");
        assert_eq!(
            message.tool_calls.expect("tool call should exist")[0]
                .function
                .arguments,
            r#"{"path":"Cargo.toml"}"#
        );
    }

    #[test]
    fn streaming_reasoning_content_is_preserved_separately_from_answer() {
        let mut aggregate = StreamAggregate::default();
        let payload = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "reasoning_content": "先分析",
                    "content": "答案"
                }
            }]
        });
        fn ignore_delta(_: &str) {}
        let mut on_delta = ignore_delta;

        aggregate
            .apply_json(&payload.to_string(), &mut on_delta)
            .expect("reasoning chunk should parse");
        let message = aggregate.finish().expect("aggregate should finish");

        assert_eq!(message.reasoning_content.as_deref(), Some("先分析"));
        assert_eq!(message.content.as_deref(), Some("答案"));
    }

    #[test]
    fn streaming_terminal_reasons_are_preserved_in_the_assistant_message() {
        for reason in ["length", "content_filter"] {
            let mut aggregate = StreamAggregate::default();
            let payload = serde_json::json!({
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "partial",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\":\""
                            }
                        }]
                    },
                    "finish_reason": reason
                }]
            });
            fn ignore_delta(_: &str) {}
            let mut on_delta = ignore_delta;

            aggregate
                .apply_json(&payload.to_string(), &mut on_delta)
                .expect("terminal reason should still produce a message");
            let message = aggregate.finish().expect("partial message should finish");

            assert_eq!(message.stop_reason.as_deref(), Some(reason));
            assert_eq!(message.content.as_deref(), Some("partial"));
            assert!(message.tool_calls.is_some());
        }
    }

    #[test]
    fn truncated_tool_calls_with_incomplete_metadata_become_terminal_messages() {
        let incomplete_calls = [
            json!({
                "index": 0,
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"}
            }),
            json!({
                "index": 0,
                "id": "call-1",
                "function": {"name": "read_file", "arguments": "{}"}
            }),
            json!({
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"arguments": "{}"}
            }),
            json!({
                "index": 0,
                "id": "call-1",
                "type": "custom",
                "function": {"name": "read_file", "arguments": "{}"}
            }),
        ];

        for tool_call in incomplete_calls {
            let mut aggregate = StreamAggregate::default();
            let payload = json!({
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "partial",
                        "tool_calls": [tool_call]
                    },
                    "finish_reason": "length"
                }]
            });
            fn ignore_delta(_: &str) {}
            let mut on_delta = ignore_delta;

            aggregate
                .apply_json(&payload.to_string(), &mut on_delta)
                .expect("truncated tool metadata should parse");
            let message = aggregate
                .finish()
                .expect("truncated message should remain recoverable");

            assert_eq!(message.stop_reason.as_deref(), Some("length"));
            assert_eq!(message.content.as_deref(), Some("partial"));
            assert!(message.tool_calls.is_none());
        }
    }

    #[test]
    fn api_error_prefers_provider_message() {
        let body = r#"{"error":{"message":"rate limit exceeded"}}"#;

        assert_eq!(api_error_message(body), "rate limit exceeded");
    }

    #[test]
    fn unknown_api_error_body_is_truncated() {
        let body = "x".repeat(MAX_ERROR_BODY_BYTES + 1);

        assert_eq!(
            api_error_message(&body).len(),
            MAX_ERROR_BODY_BYTES + "…".len()
        );
    }

    #[test]
    fn request_omits_empty_tools() {
        let request = ChatCompletionRequest {
            model: "demo",
            messages: &[],
            tools: None,
            tool_choice: None,
            temperature: 0.0,
            stream: None,
            chat_template_kwargs: None,
        };
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["model"], Value::String("demo".to_string()));
        assert!(json.get("tools").is_none());
        assert!(json.get("tool_choice").is_none());
        assert!(json.get("stream").is_none());
    }

    #[test]
    fn provider_projection_omits_internal_tool_result_metadata() {
        let messages = [Message::tool_with_result(
            "call-1",
            "hello",
            crate::message::ToolResultMetadata::new(json!({"source": "test"}), false, false),
        )];
        let projected = provider_messages(&messages, false);
        let request = ChatCompletionRequest {
            model: "demo",
            messages: &projected,
            tools: None,
            tool_choice: None,
            temperature: 0.0,
            stream: None,
            chat_template_kwargs: None,
        };

        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["messages"][0]["content"], "hello");
        assert_eq!(json["messages"][0]["tool_call_id"], "call-1");
        assert!(json["messages"][0].get("tool_result").is_none());
    }

    #[test]
    fn provider_projection_omits_internal_reasoning_content() {
        let mut message = Message::assistant(Some("answer".to_string()), None);
        message.reasoning_content = Some("private reasoning".to_string());
        let messages = [message];
        let projected = provider_messages(&messages, false);
        let request = ChatCompletionRequest {
            model: "demo",
            messages: &projected,
            tools: None,
            tool_choice: None,
            temperature: 0.0,
            stream: None,
            chat_template_kwargs: None,
        };

        let json = serde_json::to_value(request).expect("request should serialize");

        assert!(json["messages"][0].get("reasoning_content").is_none());
    }

    #[test]
    fn provider_projection_keeps_reasoning_content_when_thinking_is_enabled() {
        let mut message = Message::assistant(Some("answer".to_string()), None);
        message.reasoning_content = Some("private reasoning".to_string());
        let messages = [message];
        let projected = provider_messages(&messages, true);
        let request = ChatCompletionRequest {
            model: "qwen3.8",
            messages: &projected,
            tools: None,
            tool_choice: None,
            temperature: 0.0,
            stream: None,
            chat_template_kwargs: Some(ChatTemplateKwargs {
                enable_thinking: true,
            }),
        };

        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(
            json["messages"][0]["reasoning_content"],
            "private reasoning"
        );
        assert!(json["messages"][0].get("tool_result").is_none());
    }

    #[test]
    fn streaming_request_sets_stream_flag() {
        let request = ChatCompletionRequest {
            model: "demo",
            messages: &[],
            tools: None,
            tool_choice: None,
            temperature: 0.0,
            stream: Some(true),
            chat_template_kwargs: None,
        };
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["stream"], true);
    }

    #[test]
    fn request_sets_auto_tool_choice_when_tools_are_present() {
        let tools = [ToolDefinition::function(
            "read_file",
            "Read a workspace file",
            serde_json::json!({"type": "object"}),
        )];
        let request = ChatCompletionRequest {
            model: "demo",
            messages: &[],
            tools: Some(&tools),
            tool_choice: Some(ToolChoice::Auto),
            temperature: 0.0,
            stream: None,
            chat_template_kwargs: None,
        };
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["tool_choice"], Value::String("auto".to_string()));
    }

    #[test]
    fn request_can_disable_tool_choice_for_provider_compatibility() {
        let tools = [ToolDefinition::function(
            "read_file",
            "Read a workspace file",
            serde_json::json!({"type": "object"}),
        )];
        let request = ChatCompletionRequest {
            model: "demo",
            messages: &[],
            tools: Some(&tools),
            tool_choice: Some(ToolChoice::None),
            temperature: 0.0,
            stream: None,
            chat_template_kwargs: None,
        };
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["tool_choice"], Value::String("none".to_string()));
    }

    #[test]
    fn request_can_require_a_tool_call() {
        let tools = [ToolDefinition::function(
            "read_file",
            "Read a workspace file",
            serde_json::json!({"type": "object"}),
        )];
        let request = ChatCompletionRequest {
            model: "demo",
            messages: &[],
            tools: Some(&tools),
            tool_choice: Some(ToolChoice::Required),
            temperature: 0.0,
            stream: None,
            chat_template_kwargs: None,
        };
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["tool_choice"], Value::String("required".to_string()));
    }

    #[test]
    fn request_can_select_one_function() {
        let tools = [ToolDefinition::function(
            "read_file",
            "Read a workspace file",
            serde_json::json!({"type": "object"}),
        )];
        let request = ChatCompletionRequest {
            model: "demo",
            messages: &[],
            tools: Some(&tools),
            tool_choice: Some(ToolChoice::Function {
                name: "read_file".to_string(),
            }),
            temperature: 0.0,
            stream: None,
            chat_template_kwargs: None,
        };
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["tool_choice"]["type"], "function");
        assert_eq!(json["tool_choice"]["function"]["name"], "read_file");
    }

    #[test]
    fn request_can_enable_thinking_for_local_models() {
        let request = ChatCompletionRequest {
            model: "qwen3.8",
            messages: &[],
            tools: None,
            tool_choice: None,
            temperature: 0.7,
            stream: None,
            chat_template_kwargs: Some(ChatTemplateKwargs {
                enable_thinking: true,
            }),
        };
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["chat_template_kwargs"]["enable_thinking"], true);
    }
}
