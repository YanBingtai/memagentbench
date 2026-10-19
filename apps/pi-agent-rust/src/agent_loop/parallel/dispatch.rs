//! Bounded parallel dispatch and deterministic collection of worker results.

use std::time::{Duration, Instant};

use pi_agent_runtime::CommitGrant;
use serde_json::Value;

use crate::{
    events::{AgentEvent, RunContext},
    message::ToolCall,
    model::ChatModel,
    tools::PreparedToolCall,
};

use super::super::tool_execution::{
    build_context, execute, ToolExecutionOutcome, ToolExecutionRequest,
};
use super::super::{AgentLoop, CancellationToken};

pub(super) struct ParallelToolCompletion {
    pub(super) index: usize,
    pub(super) grant: CommitGrant,
    pub(super) started_at: Instant,
    pub(super) execution: ToolExecutionOutcome,
}

pub(super) struct ParallelChunkOutcome {
    pub(super) completed: Vec<ParallelToolCompletion>,
    pub(super) worker_error: Option<String>,
}

pub(super) struct ParallelChunkRequest<'a, M: ChatModel> {
    pub(super) agent: &'a AgentLoop<M>,
    pub(super) context: RunContext,
    pub(super) step: usize,
    pub(super) tool_calls: &'a [ToolCall],
    pub(super) prepared_calls: &'a mut [Option<(Value, PreparedToolCall)>],
    pub(super) grants: &'a [CommitGrant],
    pub(super) chunk_start: usize,
    pub(super) chunk_end: usize,
    pub(super) tool_timeout: Duration,
    pub(super) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(super) cancellation: &'a CancellationToken,
}

pub(super) async fn dispatch_parallel_chunk<M>(
    request: ParallelChunkRequest<'_, M>,
) -> ParallelChunkOutcome
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
    } = request;
    let mut tasks = tokio::task::JoinSet::new();
    for index in chunk_start..chunk_end {
        let call = &tool_calls[index];
        agent.emit(AgentEvent::ToolStarted {
            context,
            step,
            call: call.clone(),
        });
        let (_, prepared) = prepared_calls[index]
            .take()
            .expect("parallel tool preparation should be consumed once");
        let tool_context = build_context(agent, context, step, call, cancellation.clone());
        let tools = agent.tools.clone();
        let grant = grants[index].clone();
        let cancellation = cancellation.clone();
        let started_at = Instant::now();
        tasks.spawn(async move {
            let execution = execute(ToolExecutionRequest::new(
                tools,
                prepared,
                tool_context,
                tool_timeout,
                deadline,
                cancellation,
            ))
            .await;
            ParallelToolCompletion {
                index,
                grant,
                started_at,
                execution,
            }
        });
    }

    let mut completed = Vec::with_capacity(chunk_end - chunk_start);
    let mut worker_error = None;
    while let Some(joined) = tasks.join_next().await {
        match joined {
            Ok(result) => completed.push(result),
            Err(error) => {
                worker_error.get_or_insert_with(|| format!("parallel tool worker failed: {error}"));
            }
        }
    }
    completed.sort_by_key(|result| result.index);

    ParallelChunkOutcome {
        completed,
        worker_error,
    }
}
