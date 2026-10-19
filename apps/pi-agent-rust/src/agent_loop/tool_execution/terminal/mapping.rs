use std::time::Duration;

use pi_agent_runtime::CommitGrant;
use serde_json::json;

use crate::{
    agent_loop::{
        outcomes::tool_result_has_unknown_outcome, timeout_error, timeout_millis, AgentError,
    },
    tools::ToolResult,
};

use super::{unknown::unknown_details, unknown::unknown_outcome, TerminalOutcome};
use crate::agent_loop::tool_execution::timeout::ToolExecutionOutcome;

/// Convert a worker result into the result and stop policy shared by all
/// execution schedulers.
pub(crate) fn terminalize(
    grant: &CommitGrant,
    execution: ToolExecutionOutcome,
    tool_timeout: Duration,
) -> TerminalOutcome {
    match execution {
        ToolExecutionOutcome::Completed(Ok(output)) => {
            if tool_result_has_unknown_outcome(&output) {
                let reason = format!(
                    "Tool execution reported an uncertain outcome: {}",
                    output.content
                );
                return unknown_outcome(grant, reason, unknown_details(output.details));
            }
            TerminalOutcome {
                tool_result: output,
                unknown_reason: None,
                stop_error: None,
                should_commit: true,
            }
        }
        ToolExecutionOutcome::Completed(Err(error)) => TerminalOutcome {
            tool_result: ToolResult::error(format!("Tool execution failed: {error}")),
            unknown_reason: None,
            stop_error: None,
            should_commit: true,
        },
        ToolExecutionOutcome::ToolTimedOut => {
            let reason = format!(
                "Tool execution timed out after {} seconds; outcome is unknown",
                tool_timeout.as_secs()
            );
            unknown_outcome(
                grant,
                reason,
                json!({"status": "unknown", "executed": true}),
            )
        }
        ToolExecutionOutcome::RunTimedOut(run_timeout) => {
            let reason = format!(
                "Tool execution cancelled because the agent run timed out after {} milliseconds",
                timeout_millis(run_timeout)
            );
            let mut outcome = unknown_outcome(
                grant,
                reason,
                json!({"status": "unknown", "executed": true}),
            );
            outcome.stop_error = Some(timeout_error(run_timeout));
            outcome
        }
        ToolExecutionOutcome::Cancelled => {
            let reason = "Tool execution cancelled by the caller".to_string();
            let mut outcome = unknown_outcome(
                grant,
                reason,
                json!({"status": "unknown", "executed": true}),
            );
            outcome.stop_error = Some(AgentError::RunCancelled);
            outcome
        }
    }
}
