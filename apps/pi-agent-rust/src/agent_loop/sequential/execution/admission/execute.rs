use crate::model::ChatModel;

use super::super::super::super::AgentError;
use super::super::super::super::AgentLoop;
use super::super::context::build_tool_context;
use super::super::decision::admit_tool;
use super::super::outcome::{outcome_for_result, resolve_execution};
use super::super::types::{ToolExecutionOutcome, ToolExecutionRequest};

pub(super) async fn execute_allowed<M>(
    agent: &AgentLoop<M>,
    request: &mut ToolExecutionRequest<'_>,
) -> Result<ToolExecutionOutcome, AgentError>
where
    M: ChatModel,
{
    let grant = match admit_tool(agent, request) {
        Ok(grant) => grant,
        Err(tool_result) => return Ok(outcome_for_result(tool_result)),
    };

    let prepared = request
        .prepared_call
        .take()
        .expect("a call reaching execution must be prepared");
    let tool_context = build_tool_context(agent, request);
    let execution = super::super::super::super::tool_execution::execute(
        super::super::super::super::tool_execution::ToolExecutionRequest::new(
            agent.tools.clone(),
            prepared,
            tool_context,
            request.tool_timeout,
            request.deadline,
            request.cancellation.clone(),
        ),
    )
    .await;
    resolve_execution(agent, request, grant, execution).await
}
