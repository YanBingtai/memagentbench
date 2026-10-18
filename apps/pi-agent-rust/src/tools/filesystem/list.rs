//! Implementation of the bounded, deterministic `list_files` tool.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::protocol::ToolDefinition;
use pi_agent_runtime::RiskClass;

use super::super::{
    errors::ToolError, limits::*, registry::Tool, ToolConcurrency, ToolContext, ToolFuture,
    ToolReplayPolicy,
};

use std::path::PathBuf;

use super::path::{collect_directory_entries, display_relative_path, resolve_workspace_directory};

/// List workspace entries with bounded depth and output.
#[derive(Debug, Clone)]
pub struct ListFilesTool {
    context: ToolContext,
}

impl ListFilesTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for ListFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "list_files",
            "List files and directories inside the configured workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory; defaults to the workspace root"
                    },
                    "max_depth": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": MAX_LIST_DEPTH,
                        "description": "Number of directory levels below path to include"
                    },
                    "max_entries": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_LIST_ENTRIES,
                        "description": "Maximum number of entries to return"
                    }
                },
                "additionalProperties": false
            }),
        )
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let parsed: ListFilesArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "list_files".to_string(),
                    source,
                }
            })?;
        validate_list_limits(parsed.max_depth, parsed.max_entries)
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Safe
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: ListFilesArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "list_files".to_string(),
                        source,
                    }
                })?;
            let max_depth = arguments.max_depth.unwrap_or(DEFAULT_LIST_MAX_DEPTH);
            let max_entries = arguments.max_entries.unwrap_or(DEFAULT_LIST_MAX_ENTRIES);
            validate_list_limits(Some(max_depth), Some(max_entries))?;

            let requested = arguments.path.unwrap_or_else(|| PathBuf::from("."));
            let root =
                resolve_workspace_directory(&self.context.workspace, &requested, "list_files")
                    .await?;
            let prefix = display_relative_path(&requested);
            let (entries, truncated) =
                collect_directory_entries(&root, &root, &prefix, max_depth, max_entries).await?;

            let mut output = if entries.is_empty() {
                "(empty)".to_string()
            } else {
                entries.join("\n")
            };
            if truncated {
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(&format!("[truncated after {max_entries} entries]"));
            }
            Ok(truncate_output(output, self.context.max_tool_output_bytes))
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ListFilesArguments {
    path: Option<PathBuf>,
    max_depth: Option<usize>,
    max_entries: Option<usize>,
}

fn validate_list_limits(
    max_depth: Option<usize>,
    max_entries: Option<usize>,
) -> Result<(), ToolError> {
    if max_depth.is_some_and(|depth| depth > MAX_LIST_DEPTH) {
        return Err(ToolError::InvalidArgumentValue {
            name: "list_files".to_string(),
            reason: format!("max_depth must be at most {MAX_LIST_DEPTH}"),
        });
    }
    if max_entries.is_some_and(|entries| entries == 0 || entries > MAX_LIST_ENTRIES) {
        return Err(ToolError::InvalidArgumentValue {
            name: "list_files".to_string(),
            reason: format!("max_entries must be between 1 and {MAX_LIST_ENTRIES}"),
        });
    }
    Ok(())
}
