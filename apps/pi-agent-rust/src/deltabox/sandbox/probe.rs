use std::time::Duration;

use pi_agent_runtime::DeltaBoxClient;

use crate::ToolError;

use super::super::shell_error;

/// Verify that the connected host can enforce command confinement before the
/// sandbox is exposed to model-driven tool calls.
pub(super) fn verify(client: &DeltaBoxClient) -> Result<(), ToolError> {
    let probe = client
        .exec(
            &["/bin/true".to_string()],
            Some("capability-probe"),
            Duration::from_secs(10),
            || false,
        )
        .map_err(|error| {
            shell_error(format!(
                "DeltaBox command confinement is unavailable: {error}; host execution was not enabled"
            ))
        })?;
    if probe.exit_code != Some(0) || probe.truncated {
        return Err(shell_error(format!(
            "DeltaBox command capability probe failed: {}",
            probe.stderr
        )));
    }
    Ok(())
}
