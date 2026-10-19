//! Admission and execution stage for one sequential tool call.

use std::time::{Duration, Instant};

use crate::{events::AgentEvent, model::ChatModel};

use super::super::super::{
    execution::{ToolExecutionOutcome, ToolExecutionRequest},
    AgentError, AgentLoop,
};
use super::finalize::{finalize_tool, FinalizeToolRequest};
use super::request::{ToolStepOutcome, ToolStepRequest};

/// Run one call through admission and execution, then hand the result to the
/// finalization stage for hooks, runtime commit, and transcript persistence.
pub(in crate::agent_loop::sequential) async fn execute_tool<M>(
    agent: &AgentLoop<M>,
    request: ToolStepRequest<'_>,
) -> Result<ToolStepOutcome, AgentError>
where
    M: ChatModel,
{
    let ToolStepRequest {
        context,
        step,
        tool_calls,
        tool_index,
        tool_call,
        preflight,
        deadline,
        cancellation,
        runtime_run_id,
        messages,
        transcript,
        turn_tool_results,
        stats,
        runtime,
    } = request;
    let (arguments, prepared_call, validation_error) = preflight;
    let started_at = Instant::now();
    agent.emit(AgentEvent::ToolStarted {
        context,
        step,
        call: tool_call.clone(),
    });

    let tool_timeout = Duration::from_secs(agent.policy.tool_timeout_secs);
    let now_ms = super::super::super::super::unix_timestamp_ms();
    let execution = agent
        .admit_and_execute_tool(ToolExecutionRequest {
            context,
            step,
            tool_calls,
            tool_index,
            tool_call,
            arguments: arguments.clone(),
            prepared_call,
            validation_error,
            tool_timeout,
            now_ms,
            deadline,
            cancellation,
            runtime_run_id,
            messages,
            transcript,
            stats,
            runtime,
        })
        .await?;

    let ToolExecutionOutcome {
        tool_result,
        runtime_action,
        pending_runtime_terminal,
        stop_after_tool,
    } = execution;
    finalize_tool(FinalizeToolRequest {
        agent,
        context,
        step,
        tool_call,
        arguments: &arguments,
        tool_result,
        runtime_action,
        pending_runtime_terminal,
        stop_after_tool,
        deadline,
        cancellation,
        started_at,
        messages,
        transcript,
        turn_tool_results,
        runtime,
    })
    .await
}
