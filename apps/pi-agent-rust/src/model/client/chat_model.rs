use super::super::api::{
    ChatFuture, ChatModel, ChatRequest, DeltaCallback, ModelUsage, ReasoningDeltaCallback,
    ToolCallDeltaCallback,
};

use super::ModelClient;

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
