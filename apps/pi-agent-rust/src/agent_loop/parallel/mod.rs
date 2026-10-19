//! Parallel execution of independent read-only tool calls.
//!
//! Admission, bounded dispatch, terminal runtime writes, and transcript
//! persistence are separate helpers. This module keeps the ordering contract
//! visible: chunks are dispatched in bounded groups, completions are drained
//! by original call index, and every terminal path is persisted before the
//! next chunk is considered.

mod admission;
mod chunk;
mod completion;
mod dispatch;
mod drain;
mod persist;
mod terminal;

use std::time::Duration;

use pi_agent_runtime::{EventLedger, RunId, RuntimeCoordinator};

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    model::ChatModel,
};

use super::{AgentError, AgentLoop, CancellationToken, RunStats};

use admission::admit_parallel_batch;
pub(in crate::agent_loop) use completion::ParallelBatchOutcome;
use drain::{drain_parallel_chunks, ParallelDrainRequest};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn try_run_parallel_tool_batch(
        &self,
        context: RunContext,
        step: usize,
        tool_calls: &[ToolCall],
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
        turn_tool_results: &mut Vec<Message>,
        stats: &mut RunStats,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
        runtime_run_id: RunId,
        runtime: &mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    ) -> Result<Option<ParallelBatchOutcome>, AgentError> {
        let Some(admission) =
            admit_parallel_batch(self, context, step, tool_calls, runtime_run_id, runtime)
        else {
            return Ok(None);
        };

        let outcome = drain_parallel_chunks(
            ParallelDrainRequest {
                agent: self,
                context,
                step,
                tool_calls,
                messages,
                transcript,
                turn_tool_results,
                stats,
                deadline,
                cancellation,
                runtime,
            },
            admission,
        )
        .await?;

        Ok(Some(outcome))
    }
}
