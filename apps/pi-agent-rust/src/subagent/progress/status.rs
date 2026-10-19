use crate::{agent_loop::AgentError, events::RunOutcome};

pub(crate) fn child_error_status(error: &AgentError) -> &'static str {
    match error {
        AgentError::RunCancelled => "cancelled",
        AgentError::RunTimedOut { .. } => "timed_out",
        AgentError::RuntimeActionUnknown { .. } => "unknown",
        _ => "failed",
    }
}

pub(crate) fn child_error_status_with_outcome(
    error: &AgentError,
    outcome: Option<RunOutcome>,
) -> &'static str {
    if outcome == Some(RunOutcome::Unknown) {
        "unknown"
    } else {
        child_error_status(error)
    }
}
