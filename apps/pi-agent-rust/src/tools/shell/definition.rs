//! Public tool contract and lifecycle for the `bash` tool.

use pi_agent_runtime::RiskClass;
use serde_json::{json, Value};

use crate::agent_loop::CancellationToken;
use crate::protocol::ToolDefinition;

use super::super::{
    errors::ToolError, limits::*, registry::Tool, ToolConcurrency, ToolExecutionContext,
    ToolFuture, ToolReplayPolicy, ToolResultFuture,
};
use super::{
    arguments::{validate_shell_arguments, ShellArguments},
    ShellTool,
};

impl Tool for ShellTool {
    fn definition(&self) -> ToolDefinition {
        let description = match self.sandbox_mode {
            pi_agent_runtime::SandboxMode::Managed => {
                "Execute a bash command in the managed workspace sandbox. The workspace is writable, network access is disabled, and stdout/stderr are bounded."
            }
            pi_agent_runtime::SandboxMode::Disabled => {
                "Execute a bash command in the configured workspace on the host. Host execution is explicitly unsandboxed; stdout/stderr are bounded."
            }
            pi_agent_runtime::SandboxMode::External | pi_agent_runtime::SandboxMode::DeltaBox => {
                "Execute a bash command through an external supervisor."
            }
        };
        ToolDefinition::function(
            "bash",
            description,
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_SHELL_COMMAND_BYTES,
                        "description": "Bash script to execute in the configured workspace"
                    },
                    "timeout": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SHELL_TIMEOUT_SECS,
                        "description": "Maximum wall time in seconds; defaults to 120"
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        )
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let arguments: ShellArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "bash".to_string(),
                    source,
                }
            })?;
        validate_shell_arguments(&arguments)
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        let context = ToolExecutionContext::new(CancellationToken::new())
            .with_output_limit(self.context.max_tool_output_bytes);
        Box::pin(async move { Ok(self.run(arguments, context).await?.content) })
    }

    fn execute_result<'a>(&'a self, arguments: Value) -> ToolResultFuture<'a> {
        let context = ToolExecutionContext::new(CancellationToken::new())
            .with_output_limit(self.context.max_tool_output_bytes);
        Box::pin(self.run(arguments, context))
    }

    fn execute_result_with_context<'a>(
        &'a self,
        arguments: Value,
        context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(self.run(arguments, context))
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Never
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::Irreversible
    }
}
