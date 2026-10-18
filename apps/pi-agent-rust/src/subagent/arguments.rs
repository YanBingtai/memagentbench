use serde::Deserialize;

use super::super::tools::ToolError;

const MAX_TASK_BYTES: usize = 16 * 1024;
const MAX_ROLE_BYTES: usize = 256;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SubagentArguments {
    pub(super) task: String,
    pub(super) role: Option<String>,
    pub(super) max_steps: Option<usize>,
}

pub(super) fn parse_arguments(
    arguments: &serde_json::Value,
) -> Result<SubagentArguments, ToolError> {
    serde_json::from_value(arguments.clone()).map_err(|source| ToolError::InvalidArguments {
        name: "spawn_subagent".to_string(),
        source,
    })
}

pub(super) fn validate_arguments(arguments: &SubagentArguments) -> Result<(), ToolError> {
    let task = arguments.task.trim();
    if task.is_empty() {
        return Err(ToolError::Subagent {
            message: "delegated task must not be empty".to_string(),
        });
    }
    if task.len() > MAX_TASK_BYTES {
        return Err(ToolError::Subagent {
            message: format!(
                "delegated task is too large ({} bytes, limit {} bytes)",
                task.len(),
                MAX_TASK_BYTES
            ),
        });
    }
    if let Some(role) = arguments.role.as_deref() {
        if role.trim().len() > MAX_ROLE_BYTES {
            return Err(ToolError::Subagent {
                message: format!(
                    "delegated role is too large ({} bytes, limit {} bytes)",
                    role.trim().len(),
                    MAX_ROLE_BYTES
                ),
            });
        }
    }
    if arguments.max_steps == Some(0) {
        return Err(ToolError::Subagent {
            message: "delegated task must have at least one model step".to_string(),
        });
    }
    Ok(())
}
