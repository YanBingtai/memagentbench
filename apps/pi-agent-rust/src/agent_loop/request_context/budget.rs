//! Context-budget fitting and optional compaction.

use std::time::Duration;

use crate::{events::RunContext, message::Message, model::ChatModel};

use super::super::{
    await_tool_hook,
    context::{truncate_context, validate_context_structure},
    timeout_error, AgentError, AgentLoop, CancellationToken, ContextCompactionReason,
    ContextCompactionRequest, ToolHookWait,
};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(in crate::agent_loop) async fn fit_context(
        &self,
        context: RunContext,
        step: usize,
        messages: Vec<Message>,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
    ) -> Result<Vec<Message>, AgentError> {
        match truncate_context(
            &messages,
            self.policy.max_context_messages,
            self.policy.max_context_bytes,
        ) {
            Ok(messages) => {
                validate_context_structure(&messages)?;
                Ok(messages)
            }
            Err(error @ AgentError::ContextTooLarge { .. }) => {
                let Some(hook) = &self.context_compaction_hook else {
                    return Err(error);
                };
                let compacted = match await_tool_hook(
                    hook(ContextCompactionRequest {
                        context,
                        step,
                        reason: ContextCompactionReason::ByteBudget,
                        messages,
                        max_context_messages: self.policy.max_context_messages,
                        max_context_bytes: self.policy.max_context_bytes,
                    }),
                    deadline,
                    cancellation,
                )
                .await
                {
                    ToolHookWait::Completed(Ok(messages)) => messages,
                    ToolHookWait::Completed(Err(error)) => {
                        return Err(AgentError::ContextCompaction(error));
                    }
                    ToolHookWait::TimedOut(timeout) => return Err(timeout_error(timeout)),
                    ToolHookWait::Cancelled => return Err(AgentError::RunCancelled),
                };
                let compacted = truncate_context(
                    &compacted,
                    self.policy.max_context_messages,
                    self.policy.max_context_bytes,
                )?;
                validate_context_structure(&compacted)?;
                Ok(compacted)
            }
            Err(error) => Err(error),
        }
    }
}
