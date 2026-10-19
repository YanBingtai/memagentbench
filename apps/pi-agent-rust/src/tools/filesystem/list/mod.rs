//! Bounded, deterministic `list_files` workspace tool.

mod arguments;
mod operation;

use serde_json::{json, Value};

use crate::protocol::ToolDefinition;
use pi_agent_runtime::RiskClass;

use crate::tools::limits::{MAX_LIST_DEPTH, MAX_LIST_ENTRIES};

use super::super::{
    errors::ToolError, registry::Tool, ToolConcurrency, ToolContext, ToolFuture, ToolReplayPolicy,
};

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
        arguments::validate(arguments)
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
            let arguments = arguments::from_value(arguments)?;
            operation::execute(&self.context, arguments).await
        })
    }
}
