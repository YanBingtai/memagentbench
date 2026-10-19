//! Cancellation path after an ordered tool call.

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    model::ChatModel,
};

use super::super::super::{AgentError, AgentLoop, RunResult};
use super::emit_turn_finished;

pub(in crate::agent_loop::sequential) struct StopRequest<'a> {
    pub(in crate::agent_loop::sequential) context: RunContext,
    pub(in crate::agent_loop::sequential) step: usize,
    pub(in crate::agent_loop::sequential) tool_calls: &'a [ToolCall],
    pub(in crate::agent_loop::sequential) first_pending: usize,
    pub(in crate::agent_loop::sequential) assistant_message: Message,
    pub(in crate::agent_loop::sequential) messages: &'a mut Vec<Message>,
    pub(in crate::agent_loop::sequential) transcript: &'a mut Vec<Message>,
    pub(in crate::agent_loop::sequential) turn_tool_results: Vec<Message>,
    pub(in crate::agent_loop::sequential) cancellation_reason: String,
    pub(in crate::agent_loop::sequential) error: AgentError,
}

pub(in crate::agent_loop::sequential) async fn stop_after_tool<M>(
    agent: &AgentLoop<M>,
    request: StopRequest<'_>,
) -> Result<Option<RunResult>, AgentError>
where
    M: ChatModel,
{
    let StopRequest {
        context,
        step,
        tool_calls,
        first_pending,
        assistant_message,
        messages,
        transcript,
        turn_tool_results,
        cancellation_reason,
        error,
    } = request;
    agent
        .append_cancelled_tool_results(
            context,
            step,
            tool_calls,
            first_pending,
            messages,
            transcript,
            None,
            &cancellation_reason,
        )
        .await?;
    emit_turn_finished(agent, context, step, assistant_message, turn_tool_results);
    Err(error)
}
