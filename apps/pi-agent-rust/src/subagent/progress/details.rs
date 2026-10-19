use serde_json::{json, Value};
use uuid::Uuid;

use crate::tools::ToolExecutionContext;

pub(crate) fn child_details(
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
