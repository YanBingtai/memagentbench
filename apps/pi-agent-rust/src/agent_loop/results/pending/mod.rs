//! Durable records for tool calls that never complete.
//!
//! The parent loop owns scheduling and runtime admission. These helpers keep
//! the cancellation and pre-execution result paths together so both paths
//! share the same event and session ordering.

mod cancelled;
mod unexecuted;

use crate::{
    events::{AgentEvent, RunContext},
    message::{Message, ToolCall},
    model::ChatModel,
};

use super::super::{AgentError, AgentLoop};

pub(in crate::agent_loop) struct UnexecutedToolTurnRequest<'a> {
    pub(in crate::agent_loop) context: RunContext,
    pub(in crate::agent_loop) step: usize,
    pub(in crate::agent_loop) assistant_message: Message,
    pub(in crate::agent_loop) tool_calls: &'a [ToolCall],
    pub(in crate::agent_loop) messages: &'a mut Vec<Message>,
    pub(in crate::agent_loop) transcript: &'a mut Vec<Message>,
    pub(in crate::agent_loop) reason: String,
    pub(in crate::agent_loop) status: &'a str,
}

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(in crate::agent_loop) async fn append_unexecuted_tool_turn(
        &self,
        request: UnexecutedToolTurnRequest<'_>,
    ) -> Result<Vec<Message>, AgentError> {
        let UnexecutedToolTurnRequest {
            context,
            step,
            assistant_message,
            tool_calls,
            messages,
            transcript,
            reason,
            status,
        } = request;
        let tool_results = self
            .append_unexecuted_tool_results(
                context, step, tool_calls, messages, transcript, &reason, status,
            )
            .await?;
        self.emit(AgentEvent::TurnFinished {
            context,
            step,
            message: assistant_message,
            tool_results: tool_results.clone(),
        });
        Ok(tool_results)
    }
}
