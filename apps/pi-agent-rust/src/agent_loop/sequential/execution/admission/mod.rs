//! Before-hook policy and runtime admission for one sequential tool call.
//!
//! The facade keeps the batch lifecycle small while each stage owns one
//! decision: hook waiting, policy mapping, or execution after admission.

mod decision;
mod execute;
mod hook;

use crate::{model::ChatModel, tools::ToolResult};

use super::super::super::{AgentError, AgentLoop};
use super::outcome::outcome_for_result;
use super::types::{ToolExecutionOutcome, ToolExecutionRequest};

/// Run validation, before-hook policy, runtime admission, and tool execution
/// for one sequential call.
pub(super) async fn admit_and_execute<M>(
    agent: &AgentLoop<M>,
    mut request: ToolExecutionRequest<'_>,
) -> Result<ToolExecutionOutcome, AgentError>
where
    M: ChatModel,
{
    let Some(validation_error) = request.validation_error.take() else {
        let before_decision = hook::await_before_tool(agent, &mut request).await?;
        if request.cancellation.is_cancelled() {
            hook::cancel_pending(
                agent,
                &mut request,
                None,
                "Tool execution cancelled by the caller",
            )
            .await?;
            return Err(AgentError::RunCancelled);
        }
        return decision::execute(agent, request, before_decision).await;
    };

    Ok(outcome_for_result(ToolResult::error(validation_error)))
}
