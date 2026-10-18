//! Implementation of the exact-match `edit_file` tool.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::protocol::ToolDefinition;
use pi_agent_runtime::RiskClass;

use super::super::{errors::ToolError, registry::Tool, ToolConcurrency, ToolContext, ToolFuture};

use std::path::PathBuf;

use super::{path::resolve_workspace_file, write::atomic_write_workspace_file};

/// Replace one exact text span in a UTF-8 workspace file.
#[derive(Debug, Clone)]
pub struct EditFileTool {
    context: ToolContext,
}

impl EditFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "edit_file",
            "Replace exactly one matching text span in a UTF-8 workspace file.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path of the file to edit"
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text that must occur once"
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text"
                    }
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": false
            }),
        )
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let parsed: EditFileArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "edit_file".to_string(),
                    source,
                }
            })?;
        if parsed.old_text.is_empty() {
            return Err(ToolError::EditTargetEmpty { path: parsed.path });
        }
        Ok(())
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::Reversible
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: EditFileArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "edit_file".to_string(),
                        source,
                    }
                })?;
            if arguments.old_text.is_empty() {
                return Err(ToolError::EditTargetEmpty {
                    path: arguments.path,
                });
            }

            let path =
                resolve_workspace_file(&self.context.workspace, &arguments.path, "edit_file")
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
                    tool: "edit_file".to_string(),
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
            let matches = text.match_indices(&arguments.old_text).count();
            match matches {
                0 => return Err(ToolError::EditTargetNotFound { path }),
                1 => {}
                matches => return Err(ToolError::EditTargetNotUnique { path, matches }),
            }

            let updated = text.replacen(&arguments.old_text, &arguments.new_text, 1);
            if updated.len() > self.context.max_file_bytes {
                return Err(ToolError::FileTooLarge {
                    path: arguments.path,
                    size: updated.len() as u64,
                    limit: self.context.max_file_bytes,
                });
            }
            atomic_write_workspace_file(
                &self.context.workspace,
                &arguments.path,
                updated.as_bytes(),
            )
            .await?;

            Ok(format!("Edited {}", arguments.path.display()))
        })
    }
}

/// Read the complete instructions for a registered skill by exact name.

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EditFileArguments {
    path: PathBuf,
    old_text: String,
    new_text: String,
}
