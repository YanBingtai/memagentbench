//! Terminal model responses that cannot enter tool execution.

use crate::{events::RunContext, message::Message};

use super::{super::results::UnexecutedToolTurnRequest, AgentError, AgentLoop};
use crate::model::ChatModel;

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    /// Persist explicit provider termination before returning its agent error.
    ///
    /// Tool execution owns the durable result invariant: a terminal assistant
    /// response with pending calls must receive synthetic results before the
    /// run stops. Keeping this branch here leaves `run_turn` focused on
    /// orchestration while preserving its event and session ordering.
    pub(super) async fn handle_terminal_model_response(
        &self,
        context: RunContext,
        step: usize,
        assistant_message: &Message,
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
    ) -> Result<Option<AgentError>, AgentError> {
        let Some(reason) = assistant_message.stop_reason.as_deref() else {
            return Ok(None);
        };

        let (append_reason, status, error_reason) = match reason {
            "content_filter" => (
                "model output was blocked by content filter before the tool call could execute"
                    .to_string(),
                "rejected",
                "content_filter",
            ),
            "error" | "aborted" => (
                format!("model terminated before tool execution: {reason}"),
                reason,
                reason,
            ),
            _ => return Ok(None),
        };

        let tool_calls = assistant_message.tool_calls.clone().unwrap_or_default();
        self.append_unexecuted_tool_turn(UnexecutedToolTurnRequest {
            context,
            step,
            assistant_message: assistant_message.clone(),
            tool_calls: &tool_calls,
            messages,
            transcript,
            reason: append_reason,
            status,
        })
        .await?;

        Ok(Some(AgentError::ModelTerminated {
            reason: error_reason.to_string(),
        }))
    }
}
