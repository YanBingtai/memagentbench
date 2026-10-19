//! Build the provider request for a streaming completion.

use crate::model::{
    api::ChatRequest,
    error::ModelError,
    options::{
        provider_messages, provider_request_body, thinking_enabled_for_request,
        ChatCompletionRequest, ChatTemplateKwargs,
    },
};

use super::super::ModelClient;

/// Serialize a provider request and attach authentication headers.
pub(super) fn build(
    client: &ModelClient,
    request: ChatRequest<'_>,
) -> Result<reqwest::RequestBuilder, ModelError> {
    let provider_messages = provider_messages(
        request.messages,
        thinking_enabled_for_request(request.model, request.options.enable_thinking),
    );
    let body = ChatCompletionRequest {
        model: request.model,
        messages: &provider_messages,
        tools: (!request.tools.is_empty()).then_some(request.tools),
        tool_choice: (!request.tools.is_empty()).then_some(client.tool_choice.clone()),
        temperature: request.options.temperature,
        stream: Some(true),
        chat_template_kwargs: request
            .options
            .enable_thinking
            .map(|enable_thinking| ChatTemplateKwargs { enable_thinking }),
    };
    let body = provider_request_body(request.model, &body, request.options.enable_thinking)?;

    let mut builder = client
        .http
        .post(&client.endpoint)
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .json(&body);
    if let Some(api_key) = request.api_key.filter(|key| !key.is_empty()) {
        builder = builder.bearer_auth(api_key);
    }
    Ok(builder)
}
