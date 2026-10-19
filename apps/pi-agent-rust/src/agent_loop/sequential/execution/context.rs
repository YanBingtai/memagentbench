use crate::{model::ChatModel, tools::ToolExecutionContext};

use super::super::super::{AgentError, AgentLoop};
use super::types::ToolExecutionRequest;

pub(super) fn build_tool_context<M>(
    agent: &AgentLoop<M>,
    request: &ToolExecutionRequest<'_>,
) -> ToolExecutionContext
where
    M: ChatModel,
{
    super::super::super::tool_execution::build_context(
        agent,
        request.context,
        request.step,
        request.tool_call,
        request.cancellation.clone(),
    )
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
    agent
        .append_cancelled_tool_results(
            request.context,
            request.step,
            request.tool_calls,
            request.tool_index,
            &mut *request.messages,
            &mut *request.transcript,
            runtime_identity,
            reason,
        )
        .await
}
