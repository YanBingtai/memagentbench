//! Drain admitted parallel chunks in deterministic batch order.

use std::time::Duration;

use pi_agent_runtime::{EventLedger, RuntimeCoordinator};

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    model::ChatModel,
};

use super::super::{AgentError, AgentLoop, CancellationToken, RunStats};
use super::admission::ParallelAdmission;
use super::chunk::{drain_parallel_chunk, ParallelChunkRequest};
use super::completion::{ParallelBatchOutcome, ParallelBatchState};
use super::persist::persist_remaining_tool_results;

pub(super) struct ParallelDrainRequest<'a, M: ChatModel> {
    pub(super) agent: &'a AgentLoop<M>,
    pub(super) context: RunContext,
    pub(super) step: usize,
    pub(super) tool_calls: &'a [ToolCall],
    pub(super) messages: &'a mut Vec<Message>,
    pub(super) transcript: &'a mut Vec<Message>,
    pub(super) turn_tool_results: &'a mut Vec<Message>,
    pub(super) stats: &'a mut RunStats,
    pub(super) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(super) cancellation: &'a CancellationToken,
    pub(super) runtime: &'a mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
}

pub(super) async fn drain_parallel_chunks<M>(
    request: ParallelDrainRequest<'_, M>,
    admission: ParallelAdmission,
) -> Result<ParallelBatchOutcome, AgentError>
where
    M: ChatModel,
{
    let ParallelDrainRequest {
        agent,
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
    } = request;
    let ParallelAdmission {
        mut prepared_calls,
        receipt,
        tool_timeout,
    } = admission;
    let max_parallel = agent.policy.max_parallel_tool_calls;
    let mut state = ParallelBatchState::new();
    let mut chunk_start = 0;

    while chunk_start < tool_calls.len() {
        let chunk_end = (chunk_start + max_parallel).min(tool_calls.len());
        drain_parallel_chunk(ParallelChunkRequest {
            agent,
            context,
            step,
            tool_calls,
            prepared_calls: &mut prepared_calls,
            grants: &receipt.grants,
            chunk_start,
            chunk_end,
            tool_timeout,
            deadline,
            cancellation,
            messages,
            transcript,
            turn_tool_results,
            stats,
            runtime,
            state: &mut state,
        })
        .await?;

        chunk_start = chunk_end;
        if !state.has_terminal_error() {
            continue;
        }
        if chunk_start < tool_calls.len() {
            persist_remaining_tool_results(
                agent,
                context,
                step,
                tool_calls,
                chunk_start,
                &receipt.grants,
                runtime,
                messages,
                transcript,
                turn_tool_results,
                "parallel tool batch stopped before remaining calls executed",
            )
            .await?;
        }
        break;
    }

    Ok(state.into_outcome())
}
