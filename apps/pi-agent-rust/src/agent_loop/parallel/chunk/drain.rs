//! Dispatch, terminalize, and persist one parallel tool chunk.

use crate::model::ChatModel;
use std::collections::HashSet;

use super::super::dispatch::{dispatch_parallel_chunk, ParallelChunkRequest as DispatchRequest};
use super::super::persist::persist_worker_error_results;
use super::super::AgentError;
use super::completion::{persist_parallel_completion, ParallelCompletionRequest};
use super::request::ParallelChunkRequest;

pub(crate) async fn drain_parallel_chunk<M>(
    request: ParallelChunkRequest<'_, M>,
) -> Result<(), AgentError>
where
    M: ChatModel,
{
    let ParallelChunkRequest {
        agent,
        context,
        step,
        tool_calls,
        prepared_calls,
        grants,
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
        state,
    } = request;
    let chunk = dispatch_parallel_chunk(DispatchRequest {
        agent,
        context,
        step,
        tool_calls,
        prepared_calls,
        grants,
        chunk_start,
        chunk_end,
        tool_timeout,
        deadline,
        cancellation,
    })
    .await;
    let completed_indices = chunk
        .completed
        .iter()
        .map(|result| result.index)
        .collect::<HashSet<_>>();

    for completion in chunk.completed {
        persist_parallel_completion(ParallelCompletionRequest {
            agent,
            context,
            step,
            tool_calls,
            tool_timeout,
            completion,
            runtime,
            stats,
            messages,
            transcript,
            turn_tool_results,
            state,
        })
        .await?;
    }

    if let Some(reason) = chunk.worker_error {
        persist_worker_error_results(
            agent,
            context,
            step,
            tool_calls,
            chunk_start,
            chunk_end,
            &completed_indices,
            grants,
            &reason,
            runtime,
            stats,
            messages,
            transcript,
            turn_tool_results,
            state,
        )
        .await?;
        state.record_error(Some(AgentError::Runtime(reason)));
    }

    Ok(())
}
