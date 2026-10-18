//! Implementation of the bounded `read_file` workspace tool.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::protocol::ToolDefinition;
use pi_agent_runtime::RiskClass;

use super::super::{
    errors::ToolError, limits::*, registry::Tool, ToolConcurrency, ToolContext, ToolFuture,
    ToolReplayPolicy,
};

use std::path::PathBuf;

use super::path::resolve_workspace_file;

#[derive(Debug, Clone)]
pub struct ReadFileTool {
    context: ToolContext,
}

impl ReadFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "read_file",
            "Read a UTF-8 text file inside the configured workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path of the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-indexed line number to start reading from"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum number of lines to return"
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        )
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Safe
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let parsed: ReadFileArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "read_file".to_string(),
                    source,
                }
            })?;
        if parsed.offset == Some(0) {
            return Err(ToolError::InvalidRange {
                tool: "read_file".to_string(),
                reason: "offset must be greater than zero".to_string(),
            });
        }
        if parsed.limit == Some(0) {
            return Err(ToolError::InvalidRange {
                tool: "read_file".to_string(),
                reason: "limit must be greater than zero".to_string(),
            });
        }
        Ok(())
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: ReadFileArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "read_file".to_string(),
                        source,
                    }
                })?;
            let path =
                resolve_workspace_file(&self.context.workspace, &arguments.path, "read_file")
                    .await?;
            let metadata =
                tokio::fs::metadata(&path)
                    .await
                    .map_err(|source| ToolError::InspectFile {
                        path: path.clone(),
                        source,
                    })?;

            if !metadata.is_file() {
                return Err(ToolError::InvalidPath {
                    tool: "read_file".to_string(),
                    path,
                });
            }
            if metadata.len() > self.context.max_file_bytes as u64 {
                return Err(ToolError::FileTooLarge {
                    path,
                    size: metadata.len(),
                    limit: self.context.max_file_bytes,
                });
            }

            let bytes = tokio::fs::read(&path)
                .await
                .map_err(|source| ToolError::ReadFile {
                    path: path.clone(),
                    source,
                })?;
            let text = String::from_utf8(bytes)
                .map_err(|_| ToolError::InvalidUtf8 { path: path.clone() })?;
            let lines: Vec<&str> = text.split('\n').collect();
            let start = arguments.offset.unwrap_or(1);
            if start == 0 || start > lines.len() {
                return Err(ToolError::InvalidRange {
                    tool: "read_file".to_string(),
                    reason: format!(
                        "offset {start} is outside the file, which contains {} lines",
                        lines.len()
                    ),
                });
            }
            let start_index = start - 1;
            let end_index = match arguments.limit {
                Some(0) => {
                    return Err(ToolError::InvalidRange {
                        tool: "read_file".to_string(),
                        reason: "limit must be greater than zero".to_string(),
                    });
                }
                Some(limit) => start_index.saturating_add(limit).min(lines.len()),
                None => lines.len(),
            };
            let selected = lines[start_index..end_index].join("\n");

            Ok(truncate_output(
                selected,
                self.context.max_tool_output_bytes,
            ))
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileArguments {
    path: PathBuf,
    offset: Option<usize>,
    limit: Option<usize>,
}
