//! Finishes turns whose assistant response contains no executable tools.

use crate::{
    events::{AgentEvent, RunContext},
    message::Message,
    model::ChatModel,
};

use super::super::super::{
    super::{AgentError, AgentLoop, RunResult, RunStats},
    TurnOutcome,
};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(in crate::agent_loop::turn) async fn finish_without_tools(
        &self,
        context: RunContext,
        step: usize,
        assistant: Message,
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
        stats: &RunStats,
    ) -> Result<TurnOutcome, AgentError> {
        if assistant.stop_reason.as_deref() == Some("length") {
            self.emit(AgentEvent::TurnFinished {
                context,
                step,
                message: assistant,
                tool_results: Vec::new(),
            });
            return Err(AgentError::ModelTerminated {
                reason: "length".to_string(),
            });
        }

        if step < self.policy.max_steps
            && (self.inject_steering_messages(messages, transcript).await?
                || self.peek_steering().is_some()
                || self.inject_follow_up_message(messages, transcript).await?)
        {
            self.emit(AgentEvent::TurnFinished {
                context,
                step,
                message: assistant.clone(),
                tool_results: Vec::new(),
            });
            return Ok(TurnOutcome::Continue);
        }

        self.emit(AgentEvent::TurnFinished {
            context,
            step,
            message: assistant.clone(),
            tool_results: Vec::new(),
        });
        let (input_tokens, output_tokens, total_tokens) = stats.model_usage();
        Ok(TurnOutcome::Completed(Box::new(RunResult {
            messages: std::mem::take(messages),
            final_message: assistant,
            steps: step,
            input_tokens,
            output_tokens,
            total_tokens,
        })))
    }
}
