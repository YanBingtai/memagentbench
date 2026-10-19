//! Workspace-bounded `read_file` tool.

mod arguments;
mod operation;

use serde_json::{json, Value};

use crate::protocol::ToolDefinition;
use pi_agent_runtime::RiskClass;

use super::super::{
    errors::ToolError, registry::Tool, ToolConcurrency, ToolContext, ToolFuture, ToolReplayPolicy,
};

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
        arguments::validate(arguments)
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments = arguments::from_value(arguments)?;
            operation::execute(&self.context, arguments).await
        })
    }
}
