use pi_agent_runtime::RiskClass;
use serde_json::json;

use crate::{
    message::ToolDefinition,
    tools::{Tool, ToolError, ToolExecutionContext, ToolFuture, ToolResultFuture},
};

use super::{
    arguments::{parse_arguments, validate_arguments},
    SubagentTool,
};

impl Tool for SubagentTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "spawn_subagent",
            "Delegate one well-scoped task to an isolated child agent and return its bounded summary.",
            json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Complete task instructions for the child agent",
                        "minLength": 1
                    },
                    "role": {
                        "type": "string",
                        "description": "Optional role such as researcher, reviewer, or planner"
                    },
                    "max_steps": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional child step limit; the parent policy remains the upper bound"
                    }
                },
                "required": ["task"],
                "additionalProperties": false
            }),
        )
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn validate_arguments(&self, arguments: &serde_json::Value) -> Result<(), ToolError> {
        let arguments = parse_arguments(arguments)?;
        validate_arguments(&arguments)
    }

    fn execute<'a>(&'a self, arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move { Ok(self.execute_result(arguments).await?.content) })
    }

    fn execute_result<'a>(&'a self, arguments: serde_json::Value) -> ToolResultFuture<'a> {
        Box::pin(self.run_child(arguments, None))
    }

    fn execute_result_with_context<'a>(
        &'a self,
        arguments: serde_json::Value,
        context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(self.run_child(arguments, Some(context)))
    }
}
