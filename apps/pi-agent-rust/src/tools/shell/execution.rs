//! Sandbox command construction and execution for the `bash` tool.

use std::time::Duration;

use pi_agent_runtime::{CommandGateway, CommandSpec, SandboxMode, SandboxPolicy};
use serde_json::{json, Value};

use super::super::{errors::ToolError, limits::*, ToolExecutionContext, ToolResult};
use super::{
    arguments::{validate_shell_arguments, ShellArguments},
    output::{shell_command_result, shell_gateway_error_result},
    ShellTool,
};

impl ShellTool {
    pub(super) async fn run(
        &self,
        arguments: Value,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        let arguments: ShellArguments =
            serde_json::from_value(arguments).map_err(|source| ToolError::InvalidArguments {
                name: "bash".to_string(),
                source,
            })?;
        validate_shell_arguments(&arguments)?;

        let timeout_secs = arguments.timeout.unwrap_or(DEFAULT_SHELL_TIMEOUT_SECS);
        let workspace =
            std::fs::canonicalize(&self.context.workspace).map_err(|source| ToolError::Shell {
                message: format!(
                    "failed to resolve shell workspace {}: {source}",
                    self.context.workspace.display()
                ),
            })?;
        let policy = match self.sandbox_mode {
            SandboxMode::Managed => SandboxPolicy::managed(workspace),
            SandboxMode::Disabled => SandboxPolicy::disabled(workspace),
            SandboxMode::External | SandboxMode::DeltaBox => {
                return Err(ToolError::Shell {
                    message: "external sandbox mode requires a supervisor".to_string(),
                });
            }
        };
        let gateway = CommandGateway::new(self.executor.clone(), policy)
            .and_then(|gateway| gateway.with_max_output_bytes(self.context.max_tool_output_bytes))
            .and_then(|gateway| gateway.with_timeout(Duration::from_secs(timeout_secs)))
            .map_err(|error| ToolError::Shell {
                message: error.to_string(),
            })?;

        context.update(
            super::super::ToolUpdate::text("running bash command")
                .with_details(json!({"status": "started", "timeout_secs": timeout_secs})),
        );

        let command = arguments.command;
        let spec = CommandSpec::new("/bin/bash")
            .map_err(|error| ToolError::Shell {
                message: error.to_string(),
            })?
            .args(["-lc", command.as_str()]);
        let cancellation = context.cancellation_token();
        let output = tokio::task::spawn_blocking(move || {
            gateway.execute_with_cancel(&spec, || cancellation.is_cancelled())
        })
        .await
        .map_err(|error| ToolError::Shell {
            message: format!("bash worker failed: {error}"),
        })?;

        Ok(match output {
            Ok(output) => shell_command_result(output),
            Err(error) => shell_gateway_error_result(error),
        })
    }
}
