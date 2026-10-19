//! Sequential fallback for tool batches that cannot run in parallel.

use std::time::Duration;

use pi_agent_runtime::{EventLedger, RuntimeCoordinator};

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    model::ChatModel,
};

use super::super::super::TurnOutcome;
use super::super::super::{AgentError, AgentLoop, CancellationToken, RunStats};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn dispatch_sequential_batch(
        &self,
        context: RunContext,
        step: usize,
        tool_calls: Vec<ToolCall>,
        assistant_message: Message,
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
        turn_tool_results: Vec<Message>,
        stats: &mut RunStats,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
        runtime_run_id: pi_agent_runtime::RunId,
        runtime: &mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    ) -> Result<TurnOutcome, AgentError> {
        let result = self
            .try_run_sequential_tool_batch(
                context,
                step,
                &tool_calls,
                assistant_message,
                messages,
                transcript,
                turn_tool_results,
                stats,
                deadline,
                cancellation,
                runtime_run_id,
                runtime,
            )
            .await?;

        Ok(match result {
            Some(result) => TurnOutcome::Completed(Box::new(result)),
            None => TurnOutcome::Continue,
        })
    }
}
