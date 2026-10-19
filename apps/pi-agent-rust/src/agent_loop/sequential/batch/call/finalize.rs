//! Hook application, runtime commit, and transcript persistence for a result.

use std::time::{Duration, Instant};

use pi_agent_runtime::{CommitGrant, EventLedger, RuntimeCoordinator};
use serde_json::Value;

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    model::ChatModel,
    tools::ToolResult,
};

use super::super::super::super::{results::PersistToolResultRequest, runtime::commit_tool_result};
use super::super::super::{
    hooks::{AfterToolHookOutcome, AfterToolHookRequest},
    AgentError, AgentLoop, CancellationToken,
};
use super::request::ToolStepOutcome;

pub(super) struct FinalizeToolRequest<'a, M: ChatModel> {
    pub(super) agent: &'a AgentLoop<M>,
    pub(super) context: RunContext,
    pub(super) step: usize,
    pub(super) tool_call: &'a ToolCall,
    pub(super) arguments: &'a Value,
    pub(super) tool_result: ToolResult,
    pub(super) runtime_action: Option<(uuid::Uuid, uuid::Uuid)>,
    pub(super) pending_runtime_terminal: Option<CommitGrant>,
    pub(super) stop_after_tool: Option<(AgentError, String)>,
    pub(super) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(super) cancellation: &'a CancellationToken,
    pub(super) started_at: Instant,
    pub(super) messages: &'a mut Vec<Message>,
    pub(super) transcript: &'a mut Vec<Message>,
    pub(super) turn_tool_results: &'a mut Vec<Message>,
    pub(super) runtime: &'a mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
}

pub(super) async fn finalize_tool<M>(
    request: FinalizeToolRequest<'_, M>,
) -> Result<ToolStepOutcome, AgentError>
where
    M: ChatModel,
{
    let FinalizeToolRequest {
        agent,
        context,
        step,
        tool_call,
        arguments,
        mut tool_result,
        runtime_action,
        mut pending_runtime_terminal,
        stop_after_tool,
        deadline,
        cancellation,
        started_at,
        messages,
        transcript,
        turn_tool_results,
        runtime,
    } = request;

    let AfterToolHookOutcome {
        content,
        is_error,
        stop: stop_after_hook,
    } = agent
        .apply_after_tool_hook(AfterToolHookRequest {
            context,
            step,
            tool_call,
            arguments,
            tool_result: &mut tool_result,
            deadline,
            cancellation,
            skip: stop_after_tool.is_some(),
        })
        .await;
    tool_result.content = content;
    tool_result.is_error = is_error;
    let tool_result = agent.tools.normalize_result(tool_result);

    if let Some(grant) = pending_runtime_terminal.take() {
        commit_tool_result(runtime, grant, &tool_result)?;
    }
    let message = agent
        .persist_tool_result(PersistToolResultRequest {
            context,
            step,
            tool_call,
            result: &tool_result,
            runtime_identity: runtime_action,
            started_at,
            messages,
            transcript,
            turn_tool_results,
        })
        .await?;

    Ok(ToolStepOutcome {
        terminate: tool_result.terminate,
        message,
        stop: stop_after_tool.or(stop_after_hook),
    })
}
