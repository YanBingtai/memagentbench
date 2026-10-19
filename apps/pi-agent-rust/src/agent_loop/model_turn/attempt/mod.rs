//! One provider attempt and its streaming event projection.

mod events;
mod wait;

use crate::{
    events::AgentEvent,
    model::{ChatModel, ChatRequest, ToolCallDelta},
};

use super::{AgentError, CancellationToken, ModelTurnRequest};

use self::{events::AttemptEvents, wait::wait_for_model};

/// Result of one provider attempt, including whether a stream event was sent.
///
/// The parent loop uses the output marker to discard partial stream output
/// before retrying or returning a terminal error.
pub(super) struct AttemptOutcome {
    pub(super) result: Result<crate::message::Message, AgentError>,
    pub(super) had_output: bool,
}

pub(super) async fn run<M, F>(request: &ModelTurnRequest<'_, M>, emit: &F) -> AttemptOutcome
where
    M: ChatModel,
    F: Fn(AgentEvent) + Sync,
{
    let events = AttemptEvents::new(request.context, request.step, emit);
    let mut emit_delta = |content: &str| events.assistant_delta(content);
    let mut emit_reasoning = |content: &str| events.reasoning_delta(content);
    let mut emit_tool_call = |delta: &ToolCallDelta| events.tool_call_delta(delta);

    let model_request = request.client.stream_with_reasoning_and_tools(
        ChatRequest {
            model: request.model,
            api_key: request.api_key,
            messages: request.messages,
            tools: request.definitions,
            options: request.options,
        },
        &mut emit_delta,
        &mut emit_reasoning,
        &mut emit_tool_call,
    );
    let result = wait_for_model(model_request, request.deadline, request.cancellation).await;

    AttemptOutcome {
        result,
        had_output: events.had_output(),
    }
}
