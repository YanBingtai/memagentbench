use crate::model::ChatModel;

use super::super::super::super::tool_batch::{await_tool_hook, ToolHookWait};
use super::super::super::super::{
    timeout_error, timeout_millis, AgentError, AgentLoop, BeforeToolDecision,
};
use super::super::context::cancel_pending as append_cancelled;
use super::super::types::ToolExecutionRequest;

pub(super) async fn await_before_tool<M>(
    agent: &AgentLoop<M>,
    request: &mut ToolExecutionRequest<'_>,
) -> Result<BeforeToolDecision, AgentError>
where
    M: ChatModel,
{
    let Some(hook) = &agent.before_tool_hook else {
        return Ok(BeforeToolDecision::Allow);
    };

    let hook_result = await_tool_hook(
        hook(super::super::super::super::ToolHookContext {
            run: request.context,
            step: request.step,
            call: request.tool_call.clone(),
            arguments: request.arguments.clone(),
        }),
        request.deadline,
        request.cancellation,
    )
    .await;

    match hook_result {
        ToolHookWait::Completed(result) => {
            Ok(result.unwrap_or_else(|error| BeforeToolDecision::Block {
                reason: format!("before_tool_call hook failed: {error}"),
            }))
        }
        ToolHookWait::TimedOut(run_timeout) => {
            let reason = format!(
                "before_tool_call hook timed out after {} milliseconds",
                timeout_millis(run_timeout)
            );
            append_cancelled(agent, request, None, &reason).await?;
            Err(timeout_error(run_timeout))
        }
        ToolHookWait::Cancelled => {
            let reason = "before_tool_call hook cancelled by the caller";
            append_cancelled(agent, request, None, reason).await?;
            Err(AgentError::RunCancelled)
        }
    }
}

pub(super) async fn cancel_pending<M>(
    agent: &AgentLoop<M>,
    request: &mut ToolExecutionRequest<'_>,
    runtime_identity: Option<(uuid::Uuid, uuid::Uuid)>,
    reason: &str,
) -> Result<(), AgentError>
where
    M: ChatModel,
{
    super::super::context::cancel_pending(agent, request, runtime_identity, reason).await
}
