use std::{
    sync::{atomic::Ordering, Arc, Mutex},
    time::Duration,
};

use pi_agent_runtime::{DeltaBoxClient, DeltaBoxConfig, DeltaBoxError};
use serde_json::Value;
use tempfile::Builder;

use crate::ToolError;

use super::super::{shell_error, DeltaBoxDisposition};
use super::{policy, probe, DeltaBoxSandbox};
use crate::deltabox::session::DeltaBoxSession;

pub(super) fn start(
    mut config: DeltaBoxConfig,
    timeout: Duration,
) -> Result<DeltaBoxSandbox, ToolError> {
    if timeout.is_zero() {
        return Err(shell_error("tool timeout must be positive"));
    }
    let worker = std::env::current_exe().map_err(shell_error)?;
    let control_dir = Builder::new()
        .prefix("pi-agent-deltabox-")
        .tempdir()
        .map_err(shell_error)?;
    config = policy::apply(config, &worker, &control_dir)?;
    let client = DeltaBoxClient::connect(config).map_err(shell_error)?;
    let sandbox = DeltaBoxSandbox {
        inner: Arc::new(DeltaBoxSession {
            client,
            worker,
            control_dir: Mutex::new(Some(control_dir)),
            timeout,
            finished: std::sync::atomic::AtomicBool::new(false),
        }),
    };
    // Session metadata alone does not prove the host can enforce its
    // advertised boundary. Probe actual command confinement before any
    // model request; Drop aborts/stops a session whose probe fails.
    probe::verify(sandbox.client())?;
    Ok(sandbox)
}

pub(super) fn finish(
    sandbox: &DeltaBoxSandbox,
    disposition: DeltaBoxDisposition,
) -> Result<Value, DeltaBoxError> {
    let result = match disposition {
        DeltaBoxDisposition::Review => sandbox.client().diff(),
        DeltaBoxDisposition::Commit => sandbox.client().commit(),
        DeltaBoxDisposition::Abort => sandbox.client().abort(),
    };
    if result.is_err() || matches!(disposition, DeltaBoxDisposition::Review) {
        sandbox.inner.preserve_policy();
    } else if let Err(error) = sandbox.client().stop() {
        sandbox.inner.preserve_policy();
        sandbox.inner.finished.store(true, Ordering::Release);
        return Err(error);
    } else {
        sandbox
            .inner
            .control_dir
            .lock()
            .expect("control directory lock")
            .take();
    }
    sandbox.inner.finished.store(true, Ordering::Release);
    sandbox.client().close_transport();
    result
}
