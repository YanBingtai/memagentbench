use serde_json::json;

use crate::{model::ChatModel, tools::ToolResult};

use super::super::super::super::{AgentError, AgentLoop, BeforeToolDecision};
use super::super::outcome::outcome_for_result;
use super::super::types::{ToolExecutionOutcome, ToolExecutionRequest};
use super::execute::execute_allowed;

pub(super) async fn execute<M>(
    agent: &AgentLoop<M>,
    mut request: ToolExecutionRequest<'_>,
    decision: BeforeToolDecision,
) -> Result<ToolExecutionOutcome, AgentError>
where
    M: ChatModel,
{
    match decision {
        BeforeToolDecision::Allow => execute_allowed(agent, &mut request).await,
        BeforeToolDecision::Block { reason } => Ok(outcome_for_result(
            ToolResult::error(format!("Tool blocked: {reason}")).with_details(json!({
                "status": "rejected",
                "executed": false,
            })),
        )),
        BeforeToolDecision::Terminate { reason } => {
            let reason = format!("Tool execution terminated by policy: {reason}");
            Ok(ToolExecutionOutcome {
                tool_result: ToolResult::error(reason.clone())
                    .with_details(json!({
                        "status": "rejected",
                        "executed": false,
                    }))
                    .terminating(),
                runtime_action: None,
                pending_runtime_terminal: None,
                stop_after_tool: Some((
                    AgentError::ToolTerminated {
                        reason: reason.clone(),
                    },
                    reason,
                )),
            })
        }
    }
}
