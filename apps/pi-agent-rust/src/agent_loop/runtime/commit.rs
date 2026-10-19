//! Durable terminal commits for normalized tool results.

use pi_agent_runtime::{CommitGrant, EventLedger, RuntimeCoordinator};

use crate::tools::ToolResult;

use super::super::{unix_timestamp_ms, AgentError};

pub(crate) fn commit_tool_result<L: EventLedger>(
    runtime: &mut RuntimeCoordinator<L>,
    grant: CommitGrant,
    result: &ToolResult,
) -> Result<(), AgentError> {
    if result.is_error {
        runtime
            .fail(grant, result.content.clone(), unix_timestamp_ms())
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
    } else {
        let output = serde_json::to_value(result).map_err(|error| {
            AgentError::Runtime(format!(
                "failed to serialize normalized tool result: {error}"
            ))
        })?;
        runtime
            .complete(grant, output, unix_timestamp_ms())
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
    }
    Ok(())
}
