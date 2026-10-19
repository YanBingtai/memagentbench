//! Terminal completion for a successfully dispatched parallel batch.

use crate::{
    agent_loop::parallel::ParallelBatchOutcome,
    events::{AgentEvent, RunContext},
    message::Message,
    model::ChatModel,
};

use super::super::super::{AgentError, AgentLoop, RunStats};
use super::super::{super::TurnOutcome, result::completed_tool_turn};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn finish_parallel_batch(
        &self,
        context: RunContext,
        step: usize,
        assistant_message: Message,
        messages: &mut Vec<Message>,
        stats: &RunStats,
        turn_tool_results: Vec<Message>,
        outcome: ParallelBatchOutcome,
    ) -> Result<TurnOutcome, AgentError> {
        self.emit(AgentEvent::TurnFinished {
            context,
            step,
            message: assistant_message,
            tool_results: turn_tool_results,
        });
        if let Some(error) = outcome.terminal_error {
            return Err(error);
        }
        if !outcome.all_tools_terminate {
            return Ok(TurnOutcome::Continue);
        }

        let final_message = outcome
            .last_tool_message
            .expect("a non-empty parallel batch produces a tool message");
        Ok(completed_tool_turn(messages, final_message, step, stats))
    }
}
