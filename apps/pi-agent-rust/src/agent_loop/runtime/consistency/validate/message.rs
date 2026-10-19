//! Validation of session tool-result metadata against ledger actions.

use serde_json::Value;

use crate::{
    message::{Message, ToolResultMetadata},
    AgentError,
};

use super::ledger::LedgerActions;

pub(super) fn validate(message: &Message, actions: &LedgerActions) -> Result<(), AgentError> {
    let Some(metadata) = message.tool_result.as_ref() else {
        return Ok(());
    };
    let Some(action_id) = metadata.action_id else {
        return validate_anonymous(message, metadata);
    };

    let action_id_text = action_id.to_string();
    let Some(expected_grant_id) = actions.admitted.get(&action_id_text) else {
        return Err(AgentError::RuntimeSessionMismatch {
            action_id,
            reason: "durable admission event is missing".to_string(),
        });
    };
    if !actions.terminal.contains(&action_id_text) {
        return Err(AgentError::RuntimeSessionMismatch {
            action_id,
            reason: "durable terminal event is missing".to_string(),
        });
    }
    let Some(grant_id) = metadata.grant_id else {
        return Err(AgentError::RuntimeSessionMismatch {
            action_id,
            reason: "session result is missing its grant identity".to_string(),
        });
    };
    if grant_id.to_string() != *expected_grant_id {
        return Err(AgentError::RuntimeSessionMismatch {
            action_id,
            reason: format!(
                "grant identity {grant_id} does not match durable grant {expected_grant_id}"
            ),
        });
    }
    Ok(())
}

fn validate_anonymous(message: &Message, metadata: &ToolResultMetadata) -> Result<(), AgentError> {
    let Some(status) = metadata.details.get("status").and_then(Value::as_str) else {
        return Ok(());
    };
    let uncertain = match status {
        "unknown" | "timed_out" => true,
        // Cancellation before runtime admission cannot have caused an
        // external side effect. If execution is absent or true, the result
        // still needs reconciliation.
        "cancelled" => metadata.details.get("executed") != Some(&Value::Bool(false)),
        _ => false,
    };
    if uncertain {
        return Err(AgentError::UnidentifiedToolOutcome {
            call_id: message.tool_call_id.clone().unwrap_or_default(),
            status: status.to_string(),
        });
    }
    Ok(())
}
