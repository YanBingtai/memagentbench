//! OpenAI-compatible HTTP client implementation.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use reqwest::header;

use crate::protocol::{Message, ToolDefinition};

use super::{
    api::{
        ChatFuture, ChatModel, ChatRequest, DeltaCallback, ModelUsage, ReasoningDeltaCallback,
        ToolCallDelta, ToolCallDeltaCallback,
    },
    error::ModelError,
    limits::*,
    options::{
        provider_messages, provider_request_body, thinking_enabled_for_request,
        validate_chat_options, ChatCompletionRequest, ChatOptions, ChatTemplateKwargs, ToolChoice,
    },
    sse::{
        api_error_message, normalize_inline_thinking, read_bounded_body,
        validate_non_streaming_message, ChatCompletionResponse, Choice, SseDecoder,
        StreamAggregate,
    },
};

/// A small OpenAI-compatible chat-completions client.
///
/// The client owns transport configuration, while the agent owns conversation
/// state and decides when a request should be made. Keeping these concerns
/// separate lets us add other providers without changing the agent loop.
#[derive(Clone, Debug)]
pub struct ModelClient {
    pub(crate) http: reqwest::Client,
    pub(crate) endpoint: String,
    pub(crate) tool_choice: ToolChoice,
    pub(crate) last_usage: Arc<Mutex<Option<ModelUsage>>>,
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
            last_usage: Arc::new(Mutex::new(None)),
        })
    }

    fn clear_last_usage(&self) {
        *self
            .last_usage
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
    }

    fn store_last_usage(&self, usage: Option<ModelUsage>) {
        *self
            .last_usage
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = usage.filter(|value| !value.is_empty());
    }

    /// Take and clear usage from the most recent completed request.
    pub fn take_last_usage(&self) -> Option<ModelUsage> {
        self.last_usage
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
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
        self.clear_last_usage();
        let provider_messages = provider_messages(
            messages,
            thinking_enabled_for_request(model, options.enable_thinking),
        );
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

        let body = provider_request_body(model, &request, options.enable_thinking)?;
        let mut builder = self.http.post(&self.endpoint).json(&body);
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
        let usage = parsed.usage;
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
        let message = if options.enable_thinking == Some(true) {
            normalize_inline_thinking(message)
        } else {
            message
        };
        validate_non_streaming_message(&message)?;
        self.store_last_usage(usage);
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
        self.clear_last_usage();
        let provider_messages = provider_messages(
            request.messages,
            thinking_enabled_for_request(request.model, request.options.enable_thinking),
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

        let body = provider_request_body(request.model, &body, request.options.enable_thinking)?;
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
        let (message, usage) = aggregate.finish_with_usage()?;
        self.store_last_usage(usage);
        Ok(if request.options.enable_thinking == Some(true) {
            normalize_inline_thinking(message)
        } else {
            message
        })
    }
}

impl ChatModel for ModelClient {
    fn take_last_usage(&self) -> Option<ModelUsage> {
        ModelClient::take_last_usage(self)
    }

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
