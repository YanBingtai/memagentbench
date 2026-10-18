use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    agent_loop::AgentError,
    events::{AgentEvent, RunOutcome},
    tools::{ToolExecutionContext, ToolUpdate},
};

pub(super) fn child_details(
    context: Option<&ToolExecutionContext>,
    child_run_id: Option<Uuid>,
    role: Option<&str>,
    model: &str,
    max_steps: usize,
    depth: u16,
    status: &str,
) -> Value {
    let mut details = json!({
        "status": status,
        "child_run_id": child_run_id,
        "role": role,
        "model": model,
        "max_steps": max_steps,
        "depth": depth,
    });
    if let (Some(context), Value::Object(fields)) = (context, &mut details) {
        if let Some(parent_run_id) = context.parent_run_id() {
            fields.insert("parent_run_id".to_string(), json!(parent_run_id));
        }
        if let Some(parent_turn_id) = context.parent_turn_id() {
            fields.insert("parent_turn_id".to_string(), json!(parent_turn_id));
        }
        if let Some(parent_step) = context.parent_step() {
            fields.insert("parent_step".to_string(), json!(parent_step));
        }
        if let Some(parent_tool_call_id) = context.parent_tool_call_id() {
            fields.insert(
                "parent_tool_call_id".to_string(),
                Value::String(parent_tool_call_id.to_string()),
            );
        }
    }
    details
}

pub(super) fn child_error_status(error: &AgentError) -> &'static str {
    match error {
        AgentError::RunCancelled => "cancelled",
        AgentError::RunTimedOut { .. } => "timed_out",
        AgentError::RuntimeActionUnknown { .. } => "unknown",
        _ => "failed",
    }
}

pub(super) fn child_error_status_with_outcome(
    error: &AgentError,
    outcome: Option<RunOutcome>,
) -> &'static str {
    if outcome == Some(RunOutcome::Unknown) {
        "unknown"
    } else {
        child_error_status(error)
    }
}

pub(super) fn subagent_progress(event: &AgentEvent) -> Option<ToolUpdate> {
    match event {
        AgentEvent::ModelStarted { step, model, .. } => Some(
            ToolUpdate::text(format!("subagent model step {step}: {model}")).with_details(json!({
                "status": "model_started",
                "step": step,
            })),
        ),
        AgentEvent::ToolStarted { call, .. } => Some(
            ToolUpdate::text(format!("subagent tool started: {}", call.function.name))
                .with_details(json!({
                    "status": "tool_started",
                    "tool": call.function.name,
                    "call_id": call.id,
                })),
        ),
        AgentEvent::ToolFinished {
            tool_name,
            is_error,
            ..
        } => Some(
            ToolUpdate::text(format!(
                "subagent tool {}: {}",
                tool_name,
                if *is_error { "failed" } else { "completed" }
            ))
            .with_details(json!({
                "status": "tool_finished",
                "tool": tool_name,
                "is_error": is_error,
            })),
        ),
        AgentEvent::RunFinished {
            steps,
            outcome,
            error,
            ..
        } => {
            let status = match outcome {
                RunOutcome::Completed => "completed",
                RunOutcome::Cancelled => "cancelled",
                RunOutcome::Failed => "failed",
                RunOutcome::Unknown => "unknown",
            };
            Some(
                ToolUpdate::text(format!("subagent {status} after {steps} step(s)")).with_details(
                    json!({
                        "status": status,
                        "steps": steps,
                        "error": error,
                    }),
                ),
            )
        }
        _ => None,
    }
}

pub(super) fn truncate_output(mut text: String, max_bytes: usize) -> String {
    if max_bytes == 0 {
        return String::new();
    }
    if text.len() <= max_bytes {
        return text;
    }
    let marker = "…";
    if max_bytes < marker.len() {
        let mut end = max_bytes.min(text.len());
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        return text;
    }
    let limit = max_bytes.saturating_sub(marker.len());
    let mut end = limit.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text.push_str(marker);
    text
}
