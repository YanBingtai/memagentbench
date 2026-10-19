use crate::protocol::{Message, ToolDefinition};

use super::super::{
    error::ModelError,
    limits::MAX_RESPONSE_BODY_BYTES,
    options::{
        provider_messages, provider_request_body, thinking_enabled_for_request,
        validate_chat_options, ChatCompletionRequest, ChatOptions, ChatTemplateKwargs,
    },
    sse::{
        api_error_message, normalize_inline_thinking, read_bounded_body,
        validate_non_streaming_message, ChatCompletionResponse, Choice,
    },
};

use super::ModelClient;

impl ModelClient {
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
}
