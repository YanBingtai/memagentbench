use pi_agent_runtime::CommitGrant;
use serde_json::{json, Value};

use crate::{agent_loop::AgentError, tools::ToolResult};

use super::TerminalOutcome;

pub(super) fn unknown_outcome(
    grant: &CommitGrant,
    reason: String,
    details: Value,
) -> TerminalOutcome {
    TerminalOutcome {
        tool_result: ToolResult {
            content: reason.clone(),
            details,
            is_error: true,
            terminate: false,
        },
        unknown_reason: Some(reason.clone()),
        stop_error: Some(AgentError::RuntimeActionUnknown {
            action_id: grant.action_id.as_uuid(),
            tool_name: grant.tool_name.clone(),
            reason,
        }),
        should_commit: false,
    }
}

pub(super) fn unknown_details(mut details: Value) -> Value {
    if let Value::Object(fields) = &mut details {
        fields.insert("status".to_string(), Value::String("unknown".to_string()));
        fields.insert("executed".to_string(), Value::Bool(true));
        return details;
    }
    json!({"status": "unknown", "executed": true})
}
