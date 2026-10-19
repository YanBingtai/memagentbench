//! Argument decoding and validation for the `bash` tool.

use serde::Deserialize;

use super::super::{errors::ToolError, limits::*};

/// Arguments accepted by the `bash` tool.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ShellArguments {
    pub(super) command: String,
    pub(super) timeout: Option<u64>,
}

pub(super) fn validate_shell_arguments(arguments: &ShellArguments) -> Result<(), ToolError> {
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
