use std::time::Duration;

use serde_json::{json, Value};
use tempfile::NamedTempFile;

use crate::{ToolContext, ToolError, ToolExecutionContext, ToolResult, ToolUpdate};

use super::super::{
    shell_error, unknown_result,
    worker::{chunk_request, WorkerRequest, DELTABOX_WORKER_FLAG},
    CLEANUP_GRACE, MAX_WORKER_REQUEST_BYTES,
};
use super::DeltaBoxSandbox;

pub(super) async fn execute(
    sandbox: &DeltaBoxSandbox,
    name: String,
    arguments: Value,
    limits: ToolContext,
    context: ToolExecutionContext,
) -> Result<ToolResult, ToolError> {
    let lease = {
        let directory = sandbox.inner.control_dir.lock().map_err(shell_error)?;
        let directory = directory
            .as_ref()
            .ok_or_else(|| shell_error("DeltaBox session is already finalized"))?;
        NamedTempFile::new_in(directory.path()).map_err(shell_error)?
    };
    let timeout = if name == "bash" {
        arguments
            .get("timeout")
            .and_then(Value::as_u64)
            .map(Duration::from_secs)
            .unwrap_or(sandbox.inner.timeout)
            .min(sandbox.inner.timeout)
    } else {
        sandbox.inner.timeout
    };
    let request = WorkerRequest {
        name: name.clone(),
        arguments,
        max_file_bytes: limits.max_file_bytes,
        max_output_bytes: limits.max_tool_output_bytes,
        lease: lease.path().to_path_buf(),
        timeout_ms: timeout.as_millis().min(u64::MAX as u128) as u64,
    };
    let encoded = serde_json::to_string(&request).map_err(shell_error)?;
    if encoded.len() > MAX_WORKER_REQUEST_BYTES {
        return Err(shell_error(
            "DeltaBox worker request exceeds argument limit",
        ));
    }
    let mut argv = vec![
        sandbox.inner.worker.to_string_lossy().into_owned(),
        DELTABOX_WORKER_FLAG.to_string(),
    ];
    argv.extend(chunk_request(&encoded));
    let client = sandbox.client().clone();
    let session_id = client.session_id().to_string();
    let label = format!(
        "{}:{}:{}",
        context
            .parent_run_id()
            .map(|id| id.to_string())
            .unwrap_or_default(),
        name,
        context.parent_tool_call_id().unwrap_or_default()
    );
    context.update(ToolUpdate::text(format!("running {name} in DeltaBox")));
    // The lease belongs to this future. Dropping it on timeout/cancellation
    // tells the confined worker to cancel and reap its own process group.
    let mut worker = tokio::task::spawn_blocking(move || {
        client.exec(&argv, Some(&label), timeout + CLEANUP_GRACE, || false)
    });
    let cancellation = context.cancellation_token();
    let output = tokio::select! {
        result = &mut worker => result,
        _ = cancellation.cancelled() => {
            drop(lease);
            worker.await
        }
    };

    let mut result = match output {
        Ok(Ok(output)) if output.exit_code == Some(0) && !output.truncated => {
            match serde_json::from_str::<ToolResult>(&output.stdout) {
                Ok(result) => result,
                Err(error) => unknown_result(&session_id, error),
            }
        }
        Ok(Ok(output)) => unknown_result(
            &session_id,
            format!("worker exited {:?}: {}", output.exit_code, output.stderr),
        ),
        Ok(Err(error)) => unknown_result(&session_id, error),
        Err(error) => unknown_result(&session_id, error),
    };
    if !result.details.is_object() {
        result.details = json!({});
    }
    result.details["sandboxed"] = json!(true);
    result.details["supervisor"] = json!("deltabox");
    result.details["deltabox_session_id"] = json!(session_id);
    Ok(result.bounded(limits.max_tool_output_bytes))
}
