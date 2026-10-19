//! Tool result classification and run terminal mapping.

use serde_json::Value;

use crate::{events::RunOutcome, tools::ToolResult};

use super::AgentError;

pub(crate) fn tool_result_status(result: &ToolResult) -> String {
    let detail_status = result.details.get("status").and_then(Value::as_str);
    if detail_status.is_some_and(|status| {
        matches!(
            status,
            "cancelled"
                | "timed_out"
                | "unknown"
                | "rejected"
                | "truncated"
                | "max_steps"
                | "reconciled"
        )
    }) {
        return detail_status.expect("status was checked above").to_string();
    }
    if result.is_error {
        "failed".to_string()
    } else {
        "succeeded".to_string()
    }
}

pub(crate) fn run_outcome_for_error(error: &AgentError) -> RunOutcome {
    match error {
        AgentError::RuntimeActionUnknown { .. } => RunOutcome::Unknown,
        AgentError::ModelTerminated { reason } if reason == "aborted" => RunOutcome::Cancelled,
        AgentError::RunTimedOut { .. } | AgentError::RunCancelled => RunOutcome::Cancelled,
        _ => RunOutcome::Failed,
    }
}

pub(crate) fn run_outcome_for_error_with_runtime(
    error: &AgentError,
    has_unknown_action: bool,
) -> RunOutcome {
    if has_unknown_action {
        RunOutcome::Unknown
    } else {
        run_outcome_for_error(error)
    }
}

pub(crate) fn tool_result_has_unknown_outcome(result: &ToolResult) -> bool {
    result
        .details
        .get("status")
        .and_then(Value::as_str)
        .is_some_and(|status| matches!(status, "unknown" | "cancelled" | "timed_out"))
}
