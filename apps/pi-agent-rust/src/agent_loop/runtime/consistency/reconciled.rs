//! Projection of durable reconciliation outcomes into session tool messages.

use std::collections::HashMap;

use pi_agent_runtime::{EventLedger, EventSequence, RuntimeCoordinator};
use serde_json::{json, Value};

use crate::{
    message::{Message, ToolResultMetadata},
    tools::ToolResult,
    AgentError,
};

pub(crate) fn apply_reconciled_tool_results<L: EventLedger>(
    messages: &mut [Message],
    runtime: &RuntimeCoordinator<L>,
    max_output_bytes: usize,
) -> Result<(), AgentError> {
    let reconciled = collect_reconciled_events(runtime)?;
    for message in messages {
        apply_message_projection(message, &reconciled, max_output_bytes);
    }
    Ok(())
}

fn collect_reconciled_events<L: EventLedger>(
    runtime: &RuntimeCoordinator<L>,
) -> Result<HashMap<String, Value>, AgentError> {
    let events = runtime
        .ledger()
        .scan_after(EventSequence::INITIAL)
        .map_err(|error| AgentError::Runtime(error.to_string()))?;
    let mut reconciled = HashMap::new();
    for event in events {
        if event.draft.kind.as_str() != pi_agent_runtime::ACTION_RECONCILED_EVENT {
            continue;
        }
        let Some(action_id) = event.draft.payload.get("action_id").and_then(Value::as_str) else {
            continue;
        };
        reconciled.insert(action_id.to_string(), event.draft.payload);
    }
    Ok(reconciled)
}

fn apply_message_projection(
    message: &mut Message,
    reconciled: &HashMap<String, Value>,
    max_output_bytes: usize,
) {
    let Some(metadata) = message.tool_result.as_ref() else {
        return;
    };
    let Some(action_id) = metadata.action_id else {
        return;
    };
    let Some(event) = reconciled.get(&action_id.to_string()) else {
        return;
    };

    let bounded = reconciled_result(event, metadata, max_output_bytes);
    message.content = Some(bounded.content);
    if let Some(metadata) = message.tool_result.as_mut() {
        metadata.details = bounded.details;
        metadata.is_error = bounded.is_error;
    }
}

fn reconciled_result(
    event: &Value,
    metadata: &ToolResultMetadata,
    max_output_bytes: usize,
) -> ToolResult {
    let outcome = event
        .get("outcome")
        .and_then(Value::as_str)
        .unwrap_or("failed");
    let is_error = outcome == "failed";
    let content = if is_error {
        event
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("runtime action failed during reconciliation")
            .to_string()
    } else {
        match event.get("output").cloned().unwrap_or(Value::Null) {
            Value::String(content) => content,
            output => output.to_string(),
        }
    };
    ToolResult {
        content,
        details: json!({
            "status": "reconciled",
            "outcome": outcome,
            "evidence": event.get("evidence").cloned().unwrap_or(Value::Null),
            "executed": true,
        }),
        is_error,
        terminate: metadata.terminate,
    }
    .bounded(max_output_bytes)
}
