//! Completion handling for one dispatched parallel chunk.

use std::time::Duration;

use pi_agent_runtime::{EventLedger, RuntimeCoordinator};

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    model::ChatModel,
    tools::ToolResult,
};

use super::super::super::results::PersistToolResultRequest;
use super::super::super::runtime::commit_tool_result;
use super::super::super::{AgentError, AgentLoop, RunStats};
use super::super::completion::ParallelBatchState;
use super::super::dispatch::ParallelToolCompletion;
use super::super::terminal::terminalize_parallel_execution;

pub(super) struct ParallelCompletionRequest<'a, M: ChatModel> {
    pub(super) agent: &'a AgentLoop<M>,
    pub(super) context: RunContext,
    pub(super) step: usize,
    pub(super) tool_calls: &'a [ToolCall],
    pub(super) tool_timeout: Duration,
    pub(super) completion: ParallelToolCompletion,
    pub(super) runtime: &'a mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    pub(super) stats: &'a mut RunStats,
    pub(super) messages: &'a mut Vec<Message>,
    pub(super) transcript: &'a mut Vec<Message>,
    pub(super) turn_tool_results: &'a mut Vec<Message>,
    pub(super) state: &'a mut ParallelBatchState,
}

pub(super) async fn persist_parallel_completion<M>(
    request: ParallelCompletionRequest<'_, M>,
) -> Result<(), AgentError>
where
    M: ChatModel,
{
    let ParallelCompletionRequest {
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
    } = request;
    let ParallelToolCompletion {
        index,
        grant,
        started_at,
        execution,
    } = completion;
    let runtime_identity = Some((grant.action_id.as_uuid(), grant.grant_id));
    let terminalized =
        terminalize_parallel_execution(runtime, stats, grant.clone(), execution, tool_timeout)?;
    let tool_result: ToolResult = agent.tools.normalize_result(terminalized.tool_result);
    if terminalized.should_commit {
        commit_tool_result(runtime, grant, &tool_result)?;
    }
    let tool_message = agent
        .persist_tool_result(PersistToolResultRequest {
            context,
            step,
            tool_call: &tool_calls[index],
            result: &tool_result,
            runtime_identity,
            started_at,
            messages,
            transcript,
            turn_tool_results,
        })
        .await?;
    state.record_result(&tool_result, tool_message, terminalized.stop_error);
    Ok(())
}
