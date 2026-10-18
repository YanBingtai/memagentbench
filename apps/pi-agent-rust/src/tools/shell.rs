use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::protocol::ToolDefinition;

use pi_agent_runtime::{
    CommandGateway, CommandGatewayError, CommandSpec, RiskClass, SandboxExecutor, SandboxMode,
    SandboxPolicy,
};

use crate::agent_loop::CancellationToken;

use super::{
    errors::ToolError, limits::*, registry::Tool, ToolConcurrency, ToolContext,
    ToolExecutionContext, ToolFuture, ToolReplayPolicy, ToolResult, ToolResultFuture, ToolUpdate,
};

#[derive(Debug, Clone)]
pub struct ShellTool {
    context: ToolContext,
    sandbox_mode: SandboxMode,
    executor: SandboxExecutor,
}

impl ShellTool {
    /// Build a shell tool with explicit host execution.
    ///
    /// The CLI uses [`Self::with_sandbox_mode`] and defaults to managed
    /// isolation. This constructor remains useful for trusted library callers
    /// and deterministic unit tests where the host controls the boundary.
    pub fn new(context: ToolContext) -> Self {
        Self {
            context,
            sandbox_mode: SandboxMode::Disabled,
            executor: SandboxExecutor::from_path("/usr/bin/bwrap")
                .expect("the fixed bubblewrap path is absolute"),
        }
    }

    /// Build a shell tool after validating the requested sandbox boundary.
    ///
    /// Managed mode probes the actual bubblewrap capability instead of only
    /// checking that a binary exists. A failed probe is returned to the CLI so
    /// it cannot silently fall back to host execution.
    pub fn with_sandbox_mode(
        context: ToolContext,
        sandbox_mode: SandboxMode,
    ) -> Result<Self, ToolError> {
        match sandbox_mode {
            SandboxMode::Disabled => Ok(Self::new(context)),
            SandboxMode::Managed => {
                let executor =
                    SandboxExecutor::discover_and_probe().map_err(|error| ToolError::Shell {
                        message: format!("managed sandbox is unavailable: {error}"),
                    })?;
                Ok(Self {
                    context,
                    sandbox_mode,
                    executor,
                })
            }
            SandboxMode::External | SandboxMode::DeltaBox => Err(ToolError::Shell {
                message:
                    "external sandbox mode requires a supervisor and is not configured by this CLI"
                        .to_string(),
            }),
        }
    }

    async fn run(
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
            ToolUpdate::text("running bash command")
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

impl Tool for ShellTool {
    fn definition(&self) -> ToolDefinition {
        let description = match self.sandbox_mode {
            SandboxMode::Managed => {
                "Execute a bash command in the managed workspace sandbox. The workspace is writable, network access is disabled, and stdout/stderr are bounded."
            }
            SandboxMode::Disabled => {
                "Execute a bash command in the configured workspace on the host. Host execution is explicitly unsandboxed; stdout/stderr are bounded."
            }
            SandboxMode::External | SandboxMode::DeltaBox => {
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

/// Arguments accepted by the `bash` tool.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellArguments {
    command: String,
    timeout: Option<u64>,
}

fn validate_shell_arguments(arguments: &ShellArguments) -> Result<(), ToolError> {
    if arguments.command.trim().is_empty() {
        return Err(ToolError::InvalidArgumentValue {
            name: "bash".to_string(),
            reason: "command must not be empty".to_string(),
        });
    }
    if arguments.command.len() > MAX_SHELL_COMMAND_BYTES {
        return Err(ToolError::InvalidArgumentValue {
            name: "bash".to_string(),
            reason: format!("command must not exceed {MAX_SHELL_COMMAND_BYTES} bytes"),
        });
    }
    if arguments
        .timeout
        .is_some_and(|timeout| !(1..=MAX_SHELL_TIMEOUT_SECS).contains(&timeout))
    {
        return Err(ToolError::InvalidArgumentValue {
            name: "bash".to_string(),
            reason: format!("timeout must be between 1 and {MAX_SHELL_TIMEOUT_SECS} seconds"),
        });
    }
    Ok(())
}

fn shell_command_result(output: pi_agent_runtime::CommandOutput) -> ToolResult {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut content = String::new();
    if !stdout.is_empty() {
        content.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !content.is_empty() && !content.ends_with('\n') {
            content.push('\n');
        }
        content.push_str("stderr:\n");
        content.push_str(&stderr);
    }
    if content.is_empty() {
        content.push_str("(no output)");
    }
    let details = json!({
        "status": if output.success { "completed" } else { "exited_nonzero" },
        "status_code": output.status_code,
        "stdout_bytes": output.stdout.len(),
        "stderr_bytes": output.stderr.len(),
        "sandboxed": output.sandboxed,
        "duration_ms": output.duration_ms,
    });
    if output.success {
        ToolResult::text(content).with_details(details)
    } else {
        ToolResult::error(content).with_details(details)
    }
}

fn shell_gateway_error_result(error: CommandGatewayError) -> ToolResult {
    let details = match &error {
        CommandGatewayError::OutputLimitExceeded { stream, limit } => json!({
            "status": "truncated",
            "stream": match stream {
                pi_agent_runtime::OutputStream::Stdout => "stdout",
                pi_agent_runtime::OutputStream::Stderr => "stderr",
            },
            "limit_bytes": limit,
        }),
        CommandGatewayError::TimedOut { timeout } => json!({
            "status": "timed_out",
            "timeout_ms": timeout.as_millis().min(u64::MAX as u128) as u64,
        }),
        CommandGatewayError::Cancelled => json!({
            "status": "cancelled",
        }),
        _ => json!({"status": error.status()}),
    };
    ToolResult::error(format!("bash command failed: {error}")).with_details(details)
}
