//! Bounded result formatting for the `bash` tool.

use pi_agent_runtime::{CommandGatewayError, CommandOutput, OutputStream};
use serde_json::json;

use super::super::ToolResult;

pub(super) fn shell_command_result(output: CommandOutput) -> ToolResult {
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

pub(super) fn shell_gateway_error_result(error: CommandGatewayError) -> ToolResult {
    let details = match &error {
        CommandGatewayError::OutputLimitExceeded { stream, limit } => json!({
            "status": "truncated",
            "stream": match stream {
                OutputStream::Stdout => "stdout",
                OutputStream::Stderr => "stderr",
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
