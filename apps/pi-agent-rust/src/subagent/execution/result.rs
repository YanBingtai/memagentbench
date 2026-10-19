use serde_json::{json, Value};

use crate::{
    agent_loop::{AgentError, RunResult},
    tools::{ToolError, ToolExecutionContext, ToolResult},
};

use super::super::progress::{child_details, child_error_status_with_outcome, truncate_output};
use super::{run::Observation, setup::ChildSpec};

pub(super) fn build(
    child: ChildSpec,
    context: Option<&ToolExecutionContext>,
    observation: Observation,
    outcome: Result<RunResult, AgentError>,
    max_output_bytes: usize,
) -> Result<ToolResult, ToolError> {
    match outcome {
        Ok(result) => Ok(success(
            child,
            context,
            observation.run_id(),
            result,
            max_output_bytes,
        )),
        Err(error) => Ok(failure(child, context, observation, error)),
    }
}

fn success(
    child: ChildSpec,
    context: Option<&ToolExecutionContext>,
    child_run_id: Option<uuid::Uuid>,
    result: RunResult,
    max_output_bytes: usize,
) -> ToolResult {
    let summary = match result.final_message.content {
        Some(content) => content,
        None => serde_json::to_string(&result.final_message).unwrap_or_default(),
    };
    let mut details = child_details(
        context,
        child_run_id,
        child.role.as_deref(),
        &child.model,
        child.max_steps,
        child.depth,
        "completed",
    );
    if let Value::Object(fields) = &mut details {
        fields.insert("steps".to_string(), json!(result.steps));
    }
    ToolResult::text(truncate_output(summary, max_output_bytes)).with_details(details)
}

fn failure(
    child: ChildSpec,
    context: Option<&ToolExecutionContext>,
    observation: Observation,
    error: AgentError,
) -> ToolResult {
    let status = child_error_status_with_outcome(&error, observation.outcome());
    let message = error.to_string();
    let mut details = child_details(
        context,
        observation.run_id(),
        child.role.as_deref(),
        &child.model,
        child.max_steps,
        child.depth,
        status,
    );
    if let Value::Object(fields) = &mut details {
        fields.insert("error".to_string(), Value::String(message.clone()));
    }
    ToolResult::error(format!("subagent {status}: {message}")).with_details(details)
}
