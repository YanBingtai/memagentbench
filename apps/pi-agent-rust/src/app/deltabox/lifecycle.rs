use std::time::Duration;

use pi_agent_rust::{
    deltabox::{DeltaBoxDisposition, DeltaBoxSandbox},
    Config, RunResult,
};
use serde_json::Value;

use crate::{
    app_error::AppError,
    cli::{Args, SandboxModeArg},
};

use super::config::deltabox_config;

pub(crate) fn start_if_requested(
    args: &Args,
    config: &Config,
) -> Result<Option<DeltaBoxSandbox>, AppError> {
    if args.sandbox_mode != SandboxModeArg::Deltabox {
        return Ok(None);
    }

    let sandbox = DeltaBoxSandbox::start(
        deltabox_config(args, config)?,
        Duration::from_secs(config.tool_timeout_secs),
    )
    .map_err(super::shell_error)?;
    let info = sandbox.client().info();
    eprintln!(
        "[deltabox] session {}: {:?}, visibility={}, provider={}",
        sandbox.client().session_id(),
        info.resolved_fs_mode,
        info.host_visibility,
        info.isolation_provider
    );
    if let Some(reason) = &info.fallback_reason {
        eprintln!("[deltabox] filesystem fallback: {reason}");
    }
    if !info.degraded_protections.is_empty() {
        eprintln!(
            "[deltabox] unavailable protections: {:?}",
            info.degraded_protections
        );
    }
    Ok(Some(sandbox))
}

pub(crate) async fn finish(
    sandbox: DeltaBoxSandbox,
    args: &Args,
    result: &Result<RunResult, AppError>,
) -> Result<(), AppError> {
    let disposition = if result.is_ok() {
        args.deltabox_on_success.into()
    } else {
        DeltaBoxDisposition::Abort
    };
    let session_id = sandbox.client().session_id().to_string();
    let finish = tokio::task::spawn_blocking(move || sandbox.finish(disposition))
        .await
        .map_err(super::shell_error)?;
    let receipt = finish.map_err(|error| AppError::ShellSandbox {
        message: format!(
            "session {session_id} needs recovery: {error}; agent result: {:?}",
            result.as_ref().err()
        ),
    })?;
    eprintln!(
        "[deltabox] {disposition:?} session {session_id}: {}",
        serde_json::to_string(&receipt)?
    );
    if receipt
        .get("folded_publications")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        > 0
    {
        eprintln!("[deltabox] sibling changes were included; re-run project checks after commit");
    }
    Ok(())
}
