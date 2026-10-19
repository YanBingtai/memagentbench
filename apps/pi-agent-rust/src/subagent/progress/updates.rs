use serde_json::json;

use crate::{events::AgentEvent, tools::ToolUpdate};

pub(crate) fn subagent_progress(event: &AgentEvent) -> Option<ToolUpdate> {
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
                crate::events::RunOutcome::Completed => "completed",
                crate::events::RunOutcome::Cancelled => "cancelled",
                crate::events::RunOutcome::Failed => "failed",
                crate::events::RunOutcome::Unknown => "unknown",
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
