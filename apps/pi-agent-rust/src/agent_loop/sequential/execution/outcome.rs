use pi_agent_runtime::CommitGrant;

use crate::{model::ChatModel, tools::ToolResult};

use super::super::super::{unix_timestamp_ms, AgentError, AgentLoop};
use super::context::cancel_pending;
use super::types::{SequentialExecutionOutcome, ToolExecutionOutcome, ToolExecutionRequest};

pub(super) async fn resolve_execution<M>(
    agent: &AgentLoop<M>,
    request: &mut ToolExecutionRequest<'_>,
    grant: CommitGrant,
    execution: SequentialExecutionOutcome,
) -> Result<ToolExecutionOutcome, AgentError>
where
    M: ChatModel,
{
    let runtime_action = Some((grant.action_id.as_uuid(), grant.grant_id));
    let terminal =
        super::super::super::tool_execution::terminalize(&grant, execution, request.tool_timeout);
    let Some(reason) = terminal.unknown_reason else {
        return Ok(ToolExecutionOutcome {
            tool_result: terminal.tool_result,
            runtime_action,
            pending_runtime_terminal: terminal.should_commit.then_some(grant),
            stop_after_tool: None,
        });
    };

    mark_runtime_unknown(request, grant, &reason)?;
    let Some(stop_error) = terminal.stop_error else {
        return Ok(ToolExecutionOutcome {
            tool_result: terminal.tool_result,
            runtime_action,
            pending_runtime_terminal: None,
            stop_after_tool: None,
        });
    };
    if matches!(
        stop_error,
        AgentError::RunTimedOut { .. } | AgentError::RunCancelled
    ) {
        cancel_pending(agent, request, runtime_action, &reason).await?;
        return Err(stop_error);
    }
    Ok(ToolExecutionOutcome {
        tool_result: terminal.tool_result,
        runtime_action,
        pending_runtime_terminal: None,
        stop_after_tool: Some((stop_error, reason)),
    })
}

fn mark_runtime_unknown(
    request: &mut ToolExecutionRequest<'_>,
    grant: CommitGrant,
    reason: &str,
) -> Result<(), AgentError> {
    request
        .runtime
        .mark_unknown(grant, reason.to_string(), unix_timestamp_ms())
        .map_err(|error| AgentError::Runtime(error.to_string()))?;
    request.stats.has_unknown_action = true;
    Ok(())
}

pub(super) fn outcome_for_result(tool_result: ToolResult) -> ToolExecutionOutcome {
    ToolExecutionOutcome {
        tool_result,
        runtime_action: None,
        pending_runtime_terminal: None,
        stop_after_tool: None,
    }
}
