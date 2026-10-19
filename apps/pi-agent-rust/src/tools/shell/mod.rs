mod arguments;
mod definition;
mod execution;
mod output;

use pi_agent_runtime::{SandboxExecutor, SandboxMode};

use super::{errors::ToolError, ToolContext};

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
}
