//! OpenAI-compatible streaming completion.
//!
//! Request serialization and response transport live in separate modules so
//! the public `ModelClient` methods remain a small, provider-facing facade.

mod request;
mod transport;

use crate::protocol::Message;

use super::super::{
    api::{
        ChatRequest, DeltaCallback, ReasoningDeltaCallback, ToolCallDelta, ToolCallDeltaCallback,
    },
    error::ModelError,
    options::validate_chat_options,
};

use super::ModelClient;

impl ModelClient {
    /// Send an OpenAI-compatible SSE request and aggregate its final message.
    ///
    /// HTTP chunks are not SSE events, and tool arguments are not guaranteed
    /// to arrive in one piece. The transport therefore buffers by SSE event
    /// boundary and delegates canonical aggregation to the SSE state machine.
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
        let builder = request::build(self, request)?;
        transport::execute(
            self,
            builder,
            request.options.enable_thinking,
            on_delta,
            on_reasoning,
            on_tool_call,
        )
        .await
    }
}
