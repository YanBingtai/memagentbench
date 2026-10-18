//! Provider-independent request and streaming contracts.

use std::{future::Future, pin::Pin};

use serde::{Deserialize, Serialize};

use crate::protocol::{Message, ToolDefinition};

use super::{error::ModelError, options::ChatOptions};

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

/// Token usage reported by one provider completion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct ModelUsage {
    #[serde(default, alias = "prompt_tokens")]
    pub input_tokens: Option<u64>,
    #[serde(default, alias = "completion_tokens")]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
}

impl ModelUsage {
    pub(crate) fn is_empty(self) -> bool {
        self.input_tokens.is_none() && self.output_tokens.is_none() && self.total_tokens.is_none()
    }
}

/// Provider-independent model interface used by the agent runtime.
pub trait ChatModel: Send + Sync {
    fn complete<'a>(&'a self, request: ChatRequest<'a>) -> ChatFuture<'a>;

    /// Take usage from the most recent completed provider request, if the
    /// provider returned usage fields.
    fn take_last_usage(&self) -> Option<ModelUsage> {
        None
    }

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
