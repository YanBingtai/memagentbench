//! Runtime terminalization for one parallel tool execution.

use std::time::Duration;

use pi_agent_runtime::{CommitGrant, EventLedger, RuntimeCoordinator};

use crate::tools::ToolResult;

use super::super::tool_execution::ToolExecutionOutcome;
use super::super::unix_timestamp_ms;
use super::super::{tool_execution, AgentError, RunStats};

pub(super) struct TerminalizedParallelResult {
    pub(super) tool_result: ToolResult,
    pub(super) stop_error: Option<AgentError>,
    pub(super) should_commit: bool,
}

pub(super) fn terminalize_parallel_execution(
    runtime: &mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    stats: &mut RunStats,
    grant: CommitGrant,
    execution: ToolExecutionOutcome,
    tool_timeout: Duration,
) -> Result<TerminalizedParallelResult, AgentError> {
    let terminal = tool_execution::terminalize(&grant, execution, tool_timeout);
    if let Some(reason) = terminal.unknown_reason.as_deref() {
        runtime
            .mark_unknown(grant, reason.to_string(), unix_timestamp_ms())
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
        stats.has_unknown_action = true;
    }

    Ok(TerminalizedParallelResult {
        tool_result: terminal.tool_result,
        stop_error: terminal.stop_error,
        should_commit: terminal.should_commit,
    })
}
