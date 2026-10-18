use std::{
    fs,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use pi_agent_runtime::{DeltaBoxClient, DeltaBoxConfig, DeltaBoxError};
use serde_json::{json, Value};
use tempfile::NamedTempFile;

use crate::{
    EditFileTool, ListFilesTool, ReadFileTool, RegistryError, SearchFilesTool, ShellTool, Tool,
    ToolContext, ToolError, ToolExecutionContext, ToolRegistry, ToolResult, ToolUpdate,
    WriteFileTool,
};

use super::{
    session::DeltaBoxSession,
    shell_error,
    tools::DeltaBoxTool,
    unknown_result,
    worker::{chunk_request, WorkerRequest, DELTABOX_WORKER_FLAG},
    DeltaBoxDisposition, CLEANUP_GRACE, MAX_WORKER_REQUEST_BYTES,
};

#[derive(Clone, Debug)]
pub struct DeltaBoxSandbox {
    pub(super) inner: Arc<DeltaBoxSession>,
}

impl DeltaBoxSandbox {
    /// Create a confined session with network denied and a scrubbed child
    /// environment. The generated policy stays available for durable review.
    pub fn start(mut config: DeltaBoxConfig, timeout: Duration) -> Result<Self, ToolError> {
        if timeout.is_zero() {
            return Err(shell_error("tool timeout must be positive"));
        }
        let worker = std::env::current_exe().map_err(shell_error)?;
        let control_dir = tempfile::Builder::new()
            .prefix("pi-agent-deltabox-")
            .tempdir()
            .map_err(shell_error)?;
        let policy_path = control_dir.path().join("policy.toml");
        let policy = json!({
            "filesystem": {
                "read": [".", "/usr", "/bin", "/lib", "/lib64",
                    "/etc/ld.so.cache", "/etc/passwd", "/etc/group",
                    "/etc/nsswitch.conf", worker, control_dir.path()],
                "write": ["."],
                "deny": [".env", ".env.*", "environment.env", "*.pem", "*.key",
                    ".ssh", ".aws", ".gnupg", ".git-credentials", "id_rsa", "id_ed25519"],
            },
            "network": {"mode": "deny_all"},
            "limits": {
                "max_processes": 32, "max_open_files": 1024,
                "max_write_bytes": 16 * 1024 * 1024,
                "max_storage_bytes": 1024_u64 * 1024 * 1024,
            },
        });
        let policy = toml::to_string(&policy).map_err(shell_error)?;
        fs::write(&policy_path, policy).map_err(shell_error)?;
        config.policy = Some(policy_path);
        let client = DeltaBoxClient::connect(config).map_err(shell_error)?;
        let sandbox = Self {
            inner: Arc::new(DeltaBoxSession {
                client,
                worker,
                control_dir: Mutex::new(Some(control_dir)),
                timeout,
                finished: AtomicBool::new(false),
            }),
        };
        // Session metadata alone does not prove the host can enforce its
        // advertised boundary. Probe actual command confinement before any
        // model request; Drop aborts/stops a session whose probe fails.
        let probe = sandbox
            .client()
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
        Ok(sandbox)
    }

    pub fn client(&self) -> &DeltaBoxClient {
        &self.inner.client
    }

    /// Register all six workspace tools against one consistent sandbox view.
    pub fn tools(&self, context: ToolContext) -> Result<ToolRegistry, RegistryError> {
        let mut tools = ToolRegistry::new().with_output_limit(context.max_tool_output_bytes);
        let native: Vec<Arc<dyn Tool>> = vec![
            Arc::new(ReadFileTool::new(context.clone())),
            Arc::new(ListFilesTool::new(context.clone())),
            Arc::new(SearchFilesTool::new(context.clone())),
            Arc::new(WriteFileTool::new(context.clone())),
            Arc::new(EditFileTool::new(context.clone())),
            Arc::new(ShellTool::new(context.clone())),
        ];
        for tool in native {
            tools.register(DeltaBoxTool {
                native: tool,
                sandbox: self.clone(),
                context: context.clone(),
            })?;
        }
        Ok(tools)
    }

    /// Finish only after tools have quiesced. A failed commit preserves the
    /// session and policy for inspection instead of discarding pending work.
    pub fn finish(&self, disposition: DeltaBoxDisposition) -> Result<Value, DeltaBoxError> {
        let result = match disposition {
            DeltaBoxDisposition::Review => self.client().diff(),
            DeltaBoxDisposition::Commit => self.client().commit(),
            DeltaBoxDisposition::Abort => self.client().abort(),
        };
        if result.is_err() || matches!(disposition, DeltaBoxDisposition::Review) {
            self.inner.preserve_policy();
        } else {
            if let Err(error) = self.client().stop() {
                self.inner.preserve_policy();
                self.inner.finished.store(true, Ordering::Release);
                return Err(error);
            }
            self.inner
                .control_dir
                .lock()
                .expect("control directory lock")
                .take();
        }
        self.inner.finished.store(true, Ordering::Release);
        self.client().close_transport();
        result
    }

    pub(super) async fn execute(
        &self,
        name: String,
        arguments: Value,
        limits: ToolContext,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        let lease = {
            let directory = self.inner.control_dir.lock().map_err(shell_error)?;
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
                .unwrap_or(self.inner.timeout)
                .min(self.inner.timeout)
        } else {
            self.inner.timeout
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
            self.inner.worker.to_string_lossy().into_owned(),
            DELTABOX_WORKER_FLAG.to_string(),
        ];
        argv.extend(chunk_request(&encoded));
        let client = self.client().clone();
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
}
