//! Persists synthetic results when a tool batch cannot be executed.

use crate::model::ChatModel;

use super::super::super::super::results::UnexecutedToolTurnRequest;
use super::super::super::{
    super::{AgentError, AgentLoop},
    TurnOutcome,
};
use super::super::ToolBatchRequest;

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    /// Persist synthetic results before a truncated or out-of-budget batch stops.
    pub(in crate::agent_loop::turn::tool_batch) async fn handle_unexecutable_tool_batch(
        &self,
        request: &mut ToolBatchRequest<'_>,
    ) -> Result<Option<TurnOutcome>, AgentError> {
        if request.assistant_message.stop_reason.as_deref() == Some("length") {
            self.append_unexecuted_tool_turn(UnexecutedToolTurnRequest {
                context: request.context,
                step: request.step,
                assistant_message: request.assistant_message.clone(),
                tool_calls: &request.tool_calls,
                messages: request.messages,
                transcript: request.transcript,
                reason: "model output was truncated before the tool call could execute".to_string(),
                status: "truncated",
            })
            .await?;
            if request.step == self.policy.max_steps {
                return Err(AgentError::MaxStepsExceeded {
                    max_steps: self.policy.max_steps,
                });
            }
            return Ok(Some(TurnOutcome::Continue));
        }

        if request.step != self.policy.max_steps {
            return Ok(None);
        }

        self.persist_max_steps_tool_batch(request).await?;
        Err(AgentError::MaxStepsExceeded {
            max_steps: self.policy.max_steps,
        })
    }

    async fn persist_max_steps_tool_batch(
        &self,
        request: &mut ToolBatchRequest<'_>,
    ) -> Result<(), AgentError> {
        self.append_unexecuted_tool_turn(UnexecutedToolTurnRequest {
            context: request.context,
            step: request.step,
            assistant_message: request.assistant_message.clone(),
            tool_calls: &request.tool_calls,
            messages: request.messages,
            transcript: request.transcript,
            reason: "agent reached its maximum step count before executing the tool call"
                .to_string(),
            status: "max_steps",
        })
        .await
        .map(|_| ())
    }
}
