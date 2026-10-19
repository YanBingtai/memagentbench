//! Implementation of the `write_file` tool and its atomic writer.

mod arguments;
mod atomic;

use serde_json::{json, Value};

use crate::protocol::ToolDefinition;
use pi_agent_runtime::RiskClass;

use self::arguments::WriteFileArguments;
use super::super::{errors::ToolError, registry::Tool, ToolConcurrency, ToolContext, ToolFuture};

pub(super) use atomic::atomic_write_workspace_file;

/// Atomically write UTF-8 text inside the configured workspace.
#[derive(Debug, Clone)]
pub struct WriteFileTool {
    context: ToolContext,
}

impl WriteFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "write_file",
            "Atomically write UTF-8 text to a file inside the configured workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path of the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 file contents"
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        )
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let parsed: WriteFileArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "write_file".to_string(),
                    source,
                }
            })?;
        let size = parsed.content.len() as u64;
        if size > self.context.max_file_bytes as u64 {
            return Err(ToolError::FileTooLarge {
                path: parsed.path,
                size,
                limit: self.context.max_file_bytes,
            });
        }
        Ok(())
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::Reversible
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: WriteFileArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "write_file".to_string(),
                        source,
                    }
                })?;
            let size = arguments.content.len() as u64;
            if size > self.context.max_file_bytes as u64 {
                return Err(ToolError::FileTooLarge {
                    path: arguments.path.clone(),
                    size,
                    limit: self.context.max_file_bytes,
                });
            }

            atomic_write_workspace_file(
                &self.context.workspace,
                &arguments.path,
                arguments.content.as_bytes(),
            )
            .await?;

            Ok(format!(
                "Wrote {} bytes to {}",
                size,
                arguments.path.display()
            ))
        })
    }
}
