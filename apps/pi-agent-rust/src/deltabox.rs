//! Core tool adapter for a DeltaBox workspace.
//!
//! Every filesystem tool runs the same native Rust implementation in a
//! confined child of this executable. This keeps bash and file tools on the
//! same private view without duplicating path validation or file-edit logic.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use pi_agent_runtime::{DeltaBoxClient, DeltaBoxConfig, DeltaBoxError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tempfile::{NamedTempFile, TempDir};

use crate::{
    CancellationToken, EditFileTool, ListFilesTool, ReadFileTool, RegistryError, SearchFilesTool,
    ShellTool, Tool, ToolConcurrency, ToolContext, ToolDefinition, ToolError, ToolExecutionContext,
    ToolFuture, ToolRegistry, ToolReplayPolicy, ToolResult, ToolResultFuture, ToolUpdate,
    WriteFileTool,
};

/// Internal process entry point. It never starts a model or reads API keys.
pub const DELTABOX_WORKER_FLAG: &str = "--deltabox-tool-worker";
const WORKER_ARG_BYTES: usize = 32 * 1024;
const MAX_WORKER_REQUEST_BYTES: usize = 1_100_000;
const CLEANUP_GRACE: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy)]
pub enum DeltaBoxDisposition {
    /// Keep the session private and available to explicit review commands.
    Review,
    Commit,
    Abort,
}

/// One transaction shared by the CLI's workspace tools.
#[derive(Clone, Debug)]
pub struct DeltaBoxSandbox {
    inner: Arc<Session>,
}

#[derive(Debug)]
struct Session {
    client: DeltaBoxClient,
    worker: PathBuf,
    control_dir: Mutex<Option<TempDir>>,
    timeout: Duration,
    finished: AtomicBool,
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
            inner: Arc::new(Session {
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

    async fn execute(
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

impl Session {
    fn preserve_policy(&self) {
        if let Some(directory) = self
            .control_dir
            .lock()
            .expect("control directory lock")
            .take()
        {
            let _ = directory.keep();
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if self.finished.load(Ordering::Acquire) {
            return;
        }
        if let Err(error) = self.client.abort().and_then(|_| self.client.stop()) {
            self.preserve_policy();
            eprintln!(
                "[deltabox] session {} needs recovery: {error}",
                self.client.session_id()
            );
        }
    }
}

struct DeltaBoxTool {
    native: Arc<dyn Tool>,
    sandbox: DeltaBoxSandbox,
    context: ToolContext,
}

impl Tool for DeltaBoxTool {
    fn definition(&self) -> ToolDefinition {
        let mut definition = self.native.definition();
        if definition.function.name == "bash" {
            definition.function.description = "Execute bash in the DeltaBox workspace. Use relative paths. Network is denied; changes follow the session review/commit policy.".into();
        }
        definition
    }
    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        self.native.validate_arguments(arguments)
    }
    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }
    fn replay_policy(&self) -> ToolReplayPolicy {
        self.native.replay_policy()
    }
    fn risk_class(&self) -> pi_agent_runtime::RiskClass {
        self.native.risk_class()
    }
    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move { Ok(self.execute_result(arguments).await?.content) })
    }
    fn execute_result<'a>(&'a self, arguments: Value) -> ToolResultFuture<'a> {
        self.execute_result_with_context(
            arguments,
            ToolExecutionContext::new(CancellationToken::new()),
        )
    }
    fn execute_result_with_context<'a>(
        &'a self,
        arguments: Value,
        context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(self.sandbox.execute(
            self.native.definition().function.name,
            arguments,
            self.context.clone(),
            context,
        ))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerRequest {
    name: String,
    arguments: Value,
    max_file_bytes: usize,
    max_output_bytes: usize,
    lease: PathBuf,
    timeout_ms: u64,
}

/// Internal CLI worker. Execute a fixed native tool in the inherited DeltaBox
/// view; never discover providers, load config, or initialize another sandbox.
pub async fn run_deltabox_worker(chunks: Vec<String>) -> Result<ToolResult, ToolError> {
    if chunks.iter().map(String::len).sum::<usize>() > MAX_WORKER_REQUEST_BYTES {
        return Err(shell_error("DeltaBox worker request is too large"));
    }
    let request: WorkerRequest = serde_json::from_str(&chunks.concat()).map_err(shell_error)?;
    if request.max_output_bytes == 0 || request.timeout_ms == 0 {
        return Err(shell_error("DeltaBox worker limits must be positive"));
    }
    let context = ToolContext {
        workspace: std::env::current_dir().map_err(shell_error)?,
        max_file_bytes: request.max_file_bytes,
        max_tool_output_bytes: request.max_output_bytes,
    };
    let tool: Box<dyn Tool> = match request.name.as_str() {
        "read_file" => Box::new(ReadFileTool::new(context)),
        "list_files" => Box::new(ListFilesTool::new(context)),
        "search_files" => Box::new(SearchFilesTool::new(context)),
        "write_file" => Box::new(WriteFileTool::new(context)),
        "edit_file" => Box::new(EditFileTool::new(context)),
        "bash" => Box::new(ShellTool::new(context)),
        _ => return Err(shell_error("unsupported DeltaBox worker tool")),
    };
    tool.validate_arguments(&request.arguments)?;
    let cancellation = CancellationToken::new();
    let execution_context = ToolExecutionContext::new(cancellation.clone());
    // A missing lease means the owning future was dropped. Denied access is a
    // configuration error, not evidence that the caller cancelled.
    if !lease_is_live(&request.lease)? {
        return Ok(ToolResult::error("tool cancelled before execution")
            .with_details(json!({"status":"cancelled", "executed":false})));
    }
    let work = tool.execute_result_with_context(request.arguments, execution_context);
    tokio::pin!(work);
    let watch = async {
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            if !lease_is_live(&request.lease)? {
                return Ok::<(), ToolError>(());
            }
        }
    };
    let result = tokio::select! {
        result = &mut work => result,
        missing = watch => {
            cancellation.cancel();
            let result = work.await;
            missing?;
            result
        }
        _ = tokio::time::sleep(Duration::from_millis(request.timeout_ms)) => {
            cancellation.cancel();
            let _ = work.await;
            Ok(ToolResult::error("tool wall-time limit exceeded")
                .with_details(json!({"status":"timed_out", "executed":true})))
        }
    };
    Ok(match result {
        Ok(result) => result,
        Err(error) => ToolResult::error(error.to_string()),
    }
    .bounded(request.max_output_bytes))
}

fn lease_is_live(path: &Path) -> Result<bool, ToolError> {
    match fs::metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(shell_error(format!(
            "cannot read cancellation lease: {error}"
        ))),
    }
}

fn chunk_request(mut request: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    while !request.is_empty() {
        let mut end = request.len().min(WORKER_ARG_BYTES);
        while !request.is_char_boundary(end) {
            end -= 1;
        }
        chunks.push(request[..end].to_string());
        request = &request[end..];
    }
    chunks
}

fn shell_error(error: impl std::fmt::Display) -> ToolError {
    ToolError::Shell {
        message: error.to_string(),
    }
}

fn unknown_result(session_id: &str, error: impl std::fmt::Display) -> ToolResult {
    ToolResult::error(format!(
        "DeltaBox execution outcome is unknown in session {session_id}: {error}"
    ))
    .with_details(json!({"status":"unknown", "executed":true}))
    .terminating()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_unicode_requests_use_small_argv_chunks_without_changing_content() {
        let text = "智能体🦀".repeat(20_000);
        let chunks = chunk_request(&text);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.len() <= WORKER_ARG_BYTES));
        assert_eq!(chunks.concat(), text);
    }

    #[tokio::test]
    async fn missing_lease_prevents_writes() {
        let request = WorkerRequest {
            name: "write_file".into(),
            arguments: json!({"path":"never-written.txt", "content":"test"}),
            max_file_bytes: 1024,
            max_output_bytes: 1024,
            lease: PathBuf::from("/tmp/pi-agent-nonexistent-lease"),
            timeout_ms: 1000,
        };
        let result = run_deltabox_worker(vec![serde_json::to_string(&request).unwrap()])
            .await
            .unwrap();
        assert_eq!(result.details["executed"], false);
    }
}
