//! Workspace-bounded exact-match `edit_file` tool.

mod arguments;
mod operation;

use serde_json::{json, Value};

use crate::protocol::ToolDefinition;
use pi_agent_runtime::RiskClass;

use super::super::{errors::ToolError, registry::Tool, ToolConcurrency, ToolContext, ToolFuture};

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
        arguments::validate(arguments)
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::Reversible
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments = arguments::from_value(arguments)?;
            operation::execute(&self.context, arguments).await
        })
    }
}
