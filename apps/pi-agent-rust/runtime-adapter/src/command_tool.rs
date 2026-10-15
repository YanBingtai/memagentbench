//! A fixed-program tool that bridges [`CommandGateway`] into the Pi tool API.
//!
//! The executable is selected by the host when the tool is constructed. The
//! model can provide only an argv vector, so it cannot choose a new binary or
//! inject shell syntax. The runtime gateway remains responsible for policy,
//! sandboxing, cancellation, and output bounds.

use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use pi_agent_runtime::{
    CommandGateway, CommandGatewayError, CommandSpec, CommandSpecError, OutputStream, RiskClass,
};
use pi_agent_rust::{
    CancellationToken, Tool, ToolConcurrency, ToolDefinition, ToolError, ToolExecutionContext,
    ToolFuture, ToolResult, ToolResultFuture,
};
use serde::Deserialize;
use serde_json::{json, Value};

const TOOL_NAME_MAX_BYTES: usize = 128;
const DESCRIPTION_MAX_BYTES: usize = 1_024;

/// A tool whose executable is fixed by the host and whose arguments come from
/// a structured model call.
#[derive(Clone)]
pub struct FixedCommandTool {
    name: String,
    description: String,
    program: PathBuf,
    gateway: Arc<CommandGateway>,
    max_args: usize,
    risk: RiskClass,
}

impl FixedCommandTool {
    /// Creates a fixed-program tool.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        program: impl AsRef<Path>,
        gateway: CommandGateway,
        max_args: usize,
        risk: RiskClass,
    ) -> Result<Self, FixedCommandToolError> {
        let name = name.into();
        if name.trim().is_empty() || name.len() > TOOL_NAME_MAX_BYTES {
            return Err(FixedCommandToolError::InvalidName);
        }
        let description = description.into();
        if description.trim().is_empty() || description.len() > DESCRIPTION_MAX_BYTES {
            return Err(FixedCommandToolError::InvalidDescription);
        }
        let program = CommandSpec::new(program)?.program().to_path_buf();
        Ok(Self {
            name,
            description,
            program,
            gateway: Arc::new(gateway),
            max_args,
            risk,
        })
    }

    pub fn program(&self) -> &Path {
        &self.program
    }

    pub fn max_args(&self) -> usize {
        self.max_args
    }

    async fn run(
        &self,
        arguments: Value,
        cancellation: Option<CancellationToken>,
    ) -> Result<ToolResult, ToolError> {
        let arguments: CommandArguments =
            serde_json::from_value(arguments).map_err(|source| ToolError::InvalidArguments {
                name: self.name.clone(),
                source,
            })?;
        if arguments.args.len() > self.max_args {
            return Err(ToolError::InvalidRange {
                tool: self.name.clone(),
                reason: format!(
                    "at most {} arguments are allowed, got {}",
                    self.max_args,
                    arguments.args.len()
                ),
            });
        }

        let spec = CommandSpec::new(&self.program)
            .expect("FixedCommandTool validates its program during construction")
            .args(arguments.args);
        let gateway = Arc::clone(&self.gateway);
        let cancellation_flag = Arc::new(AtomicBool::new(false));
        let cancel_on_drop = CancelOnDrop(Arc::clone(&cancellation_flag));
        let worker_flag = Arc::clone(&cancellation_flag);
        let worker = tokio::task::spawn_blocking(move || {
            gateway.execute_with_cancel(&spec, || {
                worker_flag.load(Ordering::Acquire)
                    || cancellation
                        .as_ref()
                        .is_some_and(CancellationToken::is_cancelled)
            })
        });
        let output = worker.await.map_err(|error| ToolError::Subagent {
            message: format!("fixed command worker failed: {error}"),
        })?;
        drop(cancel_on_drop);

        match output {
            Ok(output) => Ok(command_result(output)),
            Err(CommandGatewayError::Cancelled) => {
                Ok(ToolResult::error("fixed command execution was cancelled")
                    .with_details(json!({"status": "cancelled"})))
            }
            Err(CommandGatewayError::TimedOut { timeout }) => Ok(ToolResult::error(format!(
                "fixed command exceeded its {:?} wall-time limit",
                timeout
            ))
            .with_details(json!({
                "status": "timed_out",
                "timeout_ms": timeout.as_millis().min(u64::MAX as u128) as u64,
            }))),
            Err(error) => {
                let details = gateway_error_details(&error);
                Ok(
                    ToolResult::error(format!("fixed command execution failed: {error}"))
                        .with_details(details),
                )
            }
        }
    }
}

impl Tool for FixedCommandTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            &self.name,
            &self.description,
            json!({
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "description": format!(
                            "Arguments passed directly to the fixed executable; at most {} arguments",
                            self.max_args
                        ),
                        "items": {"type": "string"}
                    }
                },
                "required": ["args"],
                "additionalProperties": false
            }),
        )
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let arguments: CommandArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: self.name.clone(),
                    source,
                }
            })?;
        if arguments.args.len() > self.max_args {
            return Err(ToolError::InvalidRange {
                tool: self.name.clone(),
                reason: format!(
                    "at most {} arguments are allowed, got {}",
                    self.max_args,
                    arguments.args.len()
                ),
            });
        }
        Ok(())
    }

    fn risk_class(&self) -> RiskClass {
        self.risk
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move { Ok(self.run(arguments, None).await?.content) })
    }

    fn execute_result<'a>(&'a self, arguments: Value) -> ToolResultFuture<'a> {
        Box::pin(self.run(arguments, None))
    }

    fn execute_result_with_context<'a>(
        &'a self,
        arguments: Value,
        context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(self.run(arguments, Some(context.cancellation_token())))
    }
}

/// Construction failures for a fixed command tool.
#[derive(Debug, thiserror::Error)]
pub enum FixedCommandToolError {
    #[error("fixed command tool name must be 1..={TOOL_NAME_MAX_BYTES} bytes")]
    InvalidName,
    #[error("fixed command tool description must be 1..={DESCRIPTION_MAX_BYTES} bytes")]
    InvalidDescription,
    #[error("invalid fixed command program: {0}")]
    Program(#[from] CommandSpecError),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandArguments {
    args: Vec<String>,
}

fn command_result(output: pi_agent_runtime::CommandOutput) -> ToolResult {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let content = if output.success {
        stdout
    } else if stderr.is_empty() {
        format!("fixed command exited with status {:?}", output.status_code)
    } else {
        stderr.clone()
    };
    let details = json!({
        "status": if output.success { "completed" } else { "exited_nonzero" },
        "status_code": output.status_code,
        "stderr": stderr,
        "sandboxed": output.sandboxed,
        "duration_ms": output.duration_ms,
    });
    if output.success {
        ToolResult::text(content).with_details(details)
    } else {
        ToolResult::error(content).with_details(details)
    }
}

fn gateway_error_details(error: &CommandGatewayError) -> Value {
    match error {
        CommandGatewayError::OutputLimitExceeded { stream, limit } => json!({
            "status": error.status(),
            "stream": match stream {
                OutputStream::Stdout => "stdout",
                OutputStream::Stderr => "stderr",
            },
            "limit_bytes": limit,
        }),
        _ => json!({"status": error.status()}),
    }
}

struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_rust::{CancellationToken, Tool, ToolRegistry};

    fn gateway() -> CommandGateway {
        CommandGateway::new(
            pi_agent_runtime::SandboxExecutor::from_path("/usr/bin/bwrap")
                .expect("absolute bwrap path should validate"),
            pi_agent_runtime::SandboxPolicy::disabled("/tmp"),
        )
        .expect("disabled policy should validate")
    }

    #[tokio::test]
    async fn fixed_tool_executes_the_host_selected_program() {
        let tool = FixedCommandTool::new(
            "runtime_echo",
            "Emit a fixed command result.",
            "/usr/bin/printf",
            gateway(),
            2,
            RiskClass::ReadOnly,
        )
        .expect("fixed tool should build");
        let result = tool
            .execute_result(json!({"args": ["hello"]}))
            .await
            .expect("fixed command should execute");
        assert!(!result.is_error);
        assert_eq!(result.content, "hello");
        assert!(result.details["duration_ms"].is_u64());
    }

    #[test]
    fn fixed_tool_rejects_arguments_above_the_host_limit() {
        let tool = FixedCommandTool::new(
            "runtime_echo",
            "Emit a fixed command result.",
            "/usr/bin/printf",
            gateway(),
            1,
            RiskClass::ReadOnly,
        )
        .expect("fixed tool should build");
        let error = tool
            .validate_arguments(&json!({"args": ["a", "b"]}))
            .expect_err("too many arguments should be rejected");
        assert!(matches!(error, ToolError::InvalidRange { .. }));
    }

    #[test]
    fn fixed_tool_definition_is_accepted_by_agent_registry() {
        let tool = FixedCommandTool::new(
            "runtime_echo",
            "Emit a fixed command result.",
            "/usr/bin/printf",
            gateway(),
            1,
            RiskClass::ReadOnly,
        )
        .expect("fixed tool should build");
        ToolRegistry::new()
            .register(tool)
            .expect("fixed command schema should use the agent-supported subset");
    }

    #[tokio::test]
    async fn cancellation_reaches_the_external_process_gateway() {
        let tool = FixedCommandTool::new(
            "runtime_sleep",
            "Run a fixed sleep command.",
            "/usr/bin/sleep",
            gateway(),
            1,
            RiskClass::ReadOnly,
        )
        .expect("fixed tool should build");
        let token = CancellationToken::new();
        token.cancel();
        let result = tool
            .execute_result_with_context(json!({"args": ["5"]}), ToolExecutionContext::new(token))
            .await
            .expect("cancellation should be a structured tool result");
        assert!(result.is_error);
        assert_eq!(result.details["status"], "cancelled");
    }

    #[tokio::test]
    async fn output_limits_are_returned_as_structured_tool_errors() {
        let gateway = gateway()
            .with_max_output_bytes(8)
            .expect("positive output limit should validate");
        let tool = FixedCommandTool::new(
            "runtime_echo",
            "Emit a fixed command result.",
            "/usr/bin/printf",
            gateway,
            1,
            RiskClass::ReadOnly,
        )
        .expect("fixed tool should build");
        let result = tool
            .execute_result(json!({"args": ["0123456789"]}))
            .await
            .expect("output overflow should be represented as a tool result");
        assert!(result.is_error);
        assert_eq!(result.details["status"], "output_limit");
        assert_eq!(result.details["stream"], "stdout");
        assert_eq!(result.details["limit_bytes"], 8);
        assert!(result.content.contains("exceeded"));
    }

    #[tokio::test]
    async fn timeout_is_returned_as_a_distinct_structured_tool_error() {
        let gateway = gateway()
            .with_timeout(std::time::Duration::from_millis(10))
            .expect("positive timeout should validate");
        let tool = FixedCommandTool::new(
            "runtime_sleep",
            "Run a fixed sleep command.",
            "/usr/bin/sleep",
            gateway,
            1,
            RiskClass::ReadOnly,
        )
        .expect("fixed tool should build");
        let result = tool
            .execute_result(json!({"args": ["5"]}))
            .await
            .expect("timeout should be represented as a tool result");
        assert!(result.is_error);
        assert_eq!(result.details["status"], "timed_out");
        assert_eq!(result.details["timeout_ms"], 10);
    }
}
