//! Bounded stdio JSON-RPC transport for the external DeltaBox-lite supervisor.
//!
//! A session is durable in DeltaBox; the serve process is only its transport.
//! A failed request is never replayed. Control operations can reconnect to the
//! same session after a transport failure, so callers can inspect or abort it.
//! Killing the transport alone does not prove a dispatched command stopped.

use std::{
    env, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Arc, Mutex, MutexGuard, TryLockError},
    thread,
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::os::{fd::AsRawFd, unix::process::CommandExt};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Filesystem provider requested from DeltaBox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaBoxFsMode {
    #[serde(rename = "cbcow-through")]
    CbCowThrough,
    #[serde(rename = "cbcow-private")]
    CbCowPrivate,
    #[serde(rename = "copy-private")]
    CopyPrivate,
}

/// Configuration for a project-bound DeltaBox transport.
#[derive(Debug, Clone)]
pub struct DeltaBoxConfig {
    pub binary: PathBuf,
    pub project: PathBuf,
    pub fs_mode: DeltaBoxFsMode,
    pub require_fs_mode: bool,
    /// Policy file persists for the lifetime of the durable session.
    pub policy: Option<PathBuf>,
    /// Deadline for startup and control RPCs. Command RPCs use their own limit.
    pub control_timeout: Duration,
    /// Maximum bytes in either one request line or one response line.
    pub max_frame_bytes: usize,
}

impl DeltaBoxConfig {
    pub fn discover(project: impl Into<PathBuf>) -> Result<Self, DeltaBoxError> {
        let binary = env::var_os("PATH")
            .into_iter()
            .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
            .map(|directory| directory.join("deltabox"))
            .find(|path| is_executable_file(path))
            .ok_or(DeltaBoxError::BinaryUnavailable)?;
        Ok(Self::from_binary(binary, project))
    }

    pub fn from_binary(binary: impl Into<PathBuf>, project: impl Into<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
            project: project.into(),
            fs_mode: DeltaBoxFsMode::CbCowPrivate,
            require_fs_mode: false,
            policy: None,
            control_timeout: Duration::from_secs(10),
            max_frame_bytes: 16 * 1024 * 1024,
        }
    }

    fn resolve(mut self) -> Result<Self, DeltaBoxError> {
        if self.control_timeout.is_zero() || self.max_frame_bytes == 0 {
            return Err(DeltaBoxError::InvalidConfig(
                "RPC timeout and frame limit must be positive".into(),
            ));
        }
        // Resolve before changing the child's working directory.
        self.binary = fs::canonicalize(&self.binary)?;
        self.project = fs::canonicalize(&self.project)?;
        if !self.project.is_dir() {
            return Err(DeltaBoxError::InvalidConfig(
                "project must be a directory".into(),
            ));
        }
        if let Some(policy) = &self.policy {
            self.policy = Some(fs::canonicalize(policy)?);
        }
        Ok(self)
    }
}

/// Actual capabilities reported by DeltaBox, including visible fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaBoxSessionInfo {
    pub session_id: String,
    pub project: PathBuf,
    pub status: String,
    pub resolved_fs_mode: DeltaBoxFsMode,
    pub host_visibility: String,
    pub isolation: String,
    pub isolation_provider: String,
    pub fallback_reason: Option<String>,
    #[serde(default)]
    pub degraded_protections: Vec<String>,
}

/// Result of one externally confined command.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeltaBoxExecResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
    pub truncated: bool,
}

/// Cloneable handle to a durable session and its serialized RPC transport.
#[derive(Clone, Debug)]
pub struct DeltaBoxClient {
    config: Arc<DeltaBoxConfig>,
    info: Arc<DeltaBoxSessionInfo>,
    transport: Arc<Mutex<Option<RpcProcess>>>,
}

#[derive(Debug)]
struct RpcProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    next_id: u64,
}

impl DeltaBoxClient {
    pub fn connect(config: DeltaBoxConfig) -> Result<Self, DeltaBoxError> {
        Self::open(config, None)
    }

    /// Reconnect for explicit status/diff/commit/abort; never repeat execution.
    pub fn attach(config: DeltaBoxConfig, session_id: &str) -> Result<Self, DeltaBoxError> {
        if session_id.trim().is_empty() {
            return Err(DeltaBoxError::InvalidConfig(
                "session ID must not be empty".into(),
            ));
        }
        Self::open(config, Some(session_id))
    }

    fn open(config: DeltaBoxConfig, session_id: Option<&str>) -> Result<Self, DeltaBoxError> {
        let config = config.resolve()?;
        let mut process = RpcProcess::spawn(&config)?;
        let (method, params) = match session_id {
            Some(id) => ("session.status", json!({"session_id": id})),
            None => (
                "session.start",
                json!({
                    "project": config.project,
                    "policy": config.policy,
                    "fs_mode": config.fs_mode,
                    "require_fs_mode": config.require_fs_mode,
                    "durability": "fast",
                }),
            ),
        };
        let value = process.request(
            method,
            params,
            config.max_frame_bytes,
            config.control_timeout,
            &|| false,
        )?;
        let info: DeltaBoxSessionInfo = serde_json::from_value(value)?;
        if info.session_id.is_empty()
            || session_id.is_some_and(|id| id != info.session_id)
            || info.project != config.project
            || info.isolation != "deltabox"
            || !matches!(info.isolation_provider.as_str(), "sandlock" | "seatbelt")
        {
            return Err(DeltaBoxError::Protocol(
                "session identity or isolation capabilities did not match the request".into(),
            ));
        }
        if session_id.is_none()
            && config.fs_mode != DeltaBoxFsMode::CbCowThrough
            && info.host_visibility != "on-commit"
        {
            return Err(DeltaBoxError::Protocol(
                "requested private session has immediate host visibility".into(),
            ));
        }
        Ok(Self {
            config: Arc::new(config),
            info: Arc::new(info),
            transport: Arc::new(Mutex::new(Some(process))),
        })
    }

    pub fn session_id(&self) -> &str {
        &self.info.session_id
    }
    pub fn project(&self) -> &Path {
        &self.config.project
    }
    pub fn info(&self) -> &DeltaBoxSessionInfo {
        &self.info
    }

    /// Execute exactly once. A transport timeout means execution is unknown;
    /// the caller must inspect or abort the durable session before proceeding.
    pub fn exec<F>(
        &self,
        argv: &[String],
        label: Option<&str>,
        timeout: Duration,
        should_cancel: F,
    ) -> Result<DeltaBoxExecResult, DeltaBoxError>
    where
        F: Fn() -> bool,
    {
        if argv.is_empty() || timeout.is_zero() {
            return Err(DeltaBoxError::InvalidConfig(
                "argv and timeout must be non-empty".into(),
            ));
        }
        let result = self
            .request(
                "session.exec",
                json!({
                    "session_id": self.session_id(), "argv": argv, "label": label,
                    "execution_profile": "workspace-write",
                }),
                timeout,
                should_cancel,
            )
            .map_err(|error| {
                // Sandlock can lose the child's setup error when its control
                // pipe closes. Add a local capability diagnostic, keeping the
                // original RPC code/data and never retrying the command.
                #[cfg(target_os = "linux")]
                if self.info.isolation_provider == "sandlock"
                    && matches!(error, DeltaBoxError::Remote { .. })
                {
                    return with_landlock_diagnostic(error, landlock_abi());
                }
                error
            })?;
        Ok(serde_json::from_value(result)?)
    }

    pub fn status(&self) -> Result<Value, DeltaBoxError> {
        self.control("session.status")
    }
    pub fn diff(&self) -> Result<Value, DeltaBoxError> {
        self.control("diff.list")
    }
    pub fn commit(&self) -> Result<Value, DeltaBoxError> {
        self.control("changes.commit")
    }
    pub fn abort(&self) -> Result<Value, DeltaBoxError> {
        self.control("changes.abort")
    }

    pub fn stop(&self) -> Result<Value, DeltaBoxError> {
        let result = self.control("session.stop");
        self.close_transport();
        result
    }

    /// Close an idle stdio transport, keeping the session available for review.
    /// An in-flight request retains ownership until its own deadline, so this
    /// operation cannot turn a short control timeout into an unbounded wait.
    pub fn close_transport(&self) {
        if let Ok(mut transport) = self.transport.try_lock() {
            transport.take();
        }
    }

    fn control(&self, method: &str) -> Result<Value, DeltaBoxError> {
        self.request(
            method,
            json!({"session_id": self.session_id()}),
            self.config.control_timeout,
            || false,
        )
    }

    fn lock<F>(
        &self,
        started: Instant,
        timeout: Duration,
        cancel: &F,
    ) -> Result<MutexGuard<'_, Option<RpcProcess>>, DeltaBoxError>
    where
        F: Fn() -> bool,
    {
        loop {
            check_deadline(started, timeout, cancel)?;
            match self.transport.try_lock() {
                Ok(guard) => return Ok(guard),
                Err(TryLockError::WouldBlock) => thread::sleep(POLL_INTERVAL),
                Err(TryLockError::Poisoned(_)) => return Err(DeltaBoxError::Poisoned),
            }
        }
    }

    fn request<F>(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
        cancel: F,
    ) -> Result<Value, DeltaBoxError>
    where
        F: Fn() -> bool,
    {
        let started = Instant::now();
        let mut transport = self.lock(started, timeout, &cancel)?;
        if transport.is_none() {
            *transport = Some(RpcProcess::spawn(&self.config)?);
        }
        let result = transport
            .as_mut()
            .expect("transport was initialized")
            .request(
                method,
                params,
                self.config.max_frame_bytes,
                timeout.saturating_sub(started.elapsed()),
                &cancel,
            );
        // Remote errors consumed a complete response. Any framing/I/O error
        // discards this transport instead of accepting a late reply next time.
        if result
            .as_ref()
            .is_err_and(|error| !matches!(error, DeltaBoxError::Remote { .. }))
        {
            transport.take();
        }
        result
    }
}

impl RpcProcess {
    #[cfg(unix)]
    fn spawn(config: &DeltaBoxConfig) -> Result<Self, DeltaBoxError> {
        let mut command = Command::new(&config.binary);
        command
            .arg("serve")
            .current_dir(&config.project)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .process_group(0);
        let mut child = command.spawn()?;
        let stdin = child.stdin.take().expect("stdin was configured as piped");
        let stdout = child.stdout.take().expect("stdout was configured as piped");
        let process = Self {
            child,
            stdin,
            stdout,
            next_id: 1,
        };
        set_nonblocking(&process.stdin)?;
        set_nonblocking(&process.stdout)?;
        Ok(process)
    }

    #[cfg(not(unix))]
    fn spawn(_config: &DeltaBoxConfig) -> Result<Self, DeltaBoxError> {
        Err(DeltaBoxError::InvalidConfig(
            "DeltaBox transport requires Unix".into(),
        ))
    }

    fn request<F>(
        &mut self,
        method: &str,
        params: Value,
        limit: usize,
        timeout: Duration,
        cancel: &F,
    ) -> Result<Value, DeltaBoxError>
    where
        F: Fn() -> bool,
    {
        let started = Instant::now();
        check_deadline(started, timeout, cancel)?;
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| DeltaBoxError::Protocol("RPC ID exhausted".into()))?;
        let mut request =
            serde_json::to_vec(&json!({"id": id, "method": method, "params": params}))?;
        request.push(b'\n');
        if request.len() > limit {
            return Err(DeltaBoxError::FrameTooLarge { limit });
        }
        let mut offset = 0;
        while offset < request.len() {
            check_deadline(started, timeout, cancel)?;
            match self.stdin.write(&request[offset..]) {
                Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero).into()),
                Ok(count) => offset += count,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(POLL_INTERVAL)
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error.into()),
            }
        }
        let mut response = Vec::new();
        let mut buffer = [0_u8; 8192];
        loop {
            check_deadline(started, timeout, cancel)?;
            match self.stdout.read(&mut buffer) {
                Ok(0) => {
                    return Err(DeltaBoxError::Protocol(
                        "serve exited before replying".into(),
                    ))
                }
                Ok(count) => {
                    if response.len().saturating_add(count) > limit {
                        return Err(DeltaBoxError::FrameTooLarge { limit });
                    }
                    response.extend_from_slice(&buffer[..count]);
                    if let Some(newline) = response.iter().position(|byte| *byte == b'\n') {
                        if !response[newline + 1..].iter().all(u8::is_ascii_whitespace) {
                            return Err(DeltaBoxError::Protocol(
                                "unexpected trailing response data".into(),
                            ));
                        }
                        return decode_response(&response[..newline], id);
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(POLL_INTERVAL)
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
}

impl Drop for RpcProcess {
    fn drop(&mut self) {
        // This group belongs only to our serve process. Session-scoped helpers
        // may outlive it; durable session.stop owns their normal shutdown.
        #[cfg(unix)]
        unsafe {
            libc::kill(-(self.child.id() as i32), libc::SIGKILL);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(unix)]
fn set_nonblocking(fd: &impl AsRawFd) -> io::Result<()> {
    // SAFETY: the pipe is owned and remains open for both fcntl calls.
    let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
    if flags < 0
        || unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn landlock_abi() -> io::Result<libc::c_long> {
    const LANDLOCK_CREATE_RULESET_VERSION: u32 = 1;
    // SAFETY: the VERSION query requires a null attributes pointer and size
    // zero. It reads kernel capabilities without creating or applying rules.
    let abi = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            std::ptr::null::<libc::c_void>(),
            0_usize,
            LANDLOCK_CREATE_RULESET_VERSION,
        )
    };
    if abi < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(abi)
    }
}

#[cfg(target_os = "linux")]
fn with_landlock_diagnostic(
    mut error: DeltaBoxError,
    abi: io::Result<libc::c_long>,
) -> DeltaBoxError {
    if let (DeltaBoxError::Remote { message, .. }, Err(cause)) = (&mut error, abi) {
        let guidance = match cause.raw_os_error() {
            Some(libc::ENOSYS) => {
                "ENOSYS: Landlock is not implemented by the running kernel or is hidden by an \
                 outer syscall filter; use a kernel built with CONFIG_SECURITY_LANDLOCK=y \
                 and enable landlock in its LSM list, or fix the outer filter"
            }
            Some(libc::EOPNOTSUPP) => {
                "EOPNOTSUPP: Landlock is compiled in but disabled; enable landlock in the \
                 boot-time LSM list while preserving the other security modules"
            }
            Some(libc::EPERM | libc::EACCES) => {
                "Landlock access was denied; check the outer sandbox/syscall policy"
            }
            _ => "check kernel Landlock support and the outer sandbox/syscall policy",
        };
        message.push_str(&format!(
            "; additional host capability diagnostic: Landlock ABI query failed ({cause}); \
             {guidance}; copy-private removes the FUSE requirement but still needs Landlock"
        ));
    }
    error
}

fn decode_response(bytes: &[u8], id: u64) -> Result<Value, DeltaBoxError> {
    let value: Value = serde_json::from_slice(bytes)?;
    if value.get("id").and_then(Value::as_u64) != Some(id) {
        return Err(DeltaBoxError::Protocol(
            "response ID did not match request".into(),
        ));
    }
    if let Some(error) = value.get("error").filter(|value| !value.is_null()) {
        return Err(DeltaBoxError::Remote {
            code: error.get("code").and_then(Value::as_i64),
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unspecified RPC error")
                .to_string(),
            data: error.get("data").cloned(),
        });
    }
    value
        .get("result")
        .cloned()
        .ok_or_else(|| DeltaBoxError::Protocol("response has neither result nor error".into()))
}

fn check_deadline<F>(started: Instant, timeout: Duration, cancel: &F) -> Result<(), DeltaBoxError>
where
    F: Fn() -> bool,
{
    if cancel() {
        return Err(DeltaBoxError::Cancelled);
    }
    if started.elapsed() >= timeout {
        return Err(DeltaBoxError::TimedOut { timeout });
    }
    Ok(())
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        metadata.is_file()
    }
}

#[derive(Debug, Error)]
pub enum DeltaBoxError {
    #[error("deltabox executable was not found on PATH")]
    BinaryUnavailable,
    #[error("invalid DeltaBox configuration: {0}")]
    InvalidConfig(String),
    #[error("DeltaBox I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("invalid DeltaBox JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid DeltaBox response: {0}")]
    Protocol(String),
    #[error("DeltaBox request failed ({code:?}): {message}")]
    Remote {
        code: Option<i64>,
        message: String,
        data: Option<Value>,
    },
    #[error("DeltaBox RPC frame exceeds {limit} bytes")]
    FrameTooLarge { limit: usize },
    #[error("DeltaBox RPC exceeded {timeout:?}; dispatched execution may still be running")]
    TimedOut { timeout: Duration },
    #[error("DeltaBox RPC cancelled; dispatched execution may still be running")]
    Cancelled,
    #[error("DeltaBox transport lock was poisoned")]
    Poisoned,
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use tempfile::{tempdir, TempDir};

    fn fixture() -> (TempDir, DeltaBoxClient) {
        let directory = tempdir().unwrap();
        // Execute the installed shell rather than a just-written executable:
        // parallel forks can briefly inherit another test's writable script
        // descriptor and cause ETXTBSY. The transport's serve argument names
        // this script relative to its project directory.
        let script_path = directory.path().join("serve");
        let script = r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *session.start*|*session.status*) printf '{"id":%s,"result":{"session_id":"test-session","project":"%s","status":"running","resolved_fs_mode":"copy-private","host_visibility":"on-commit","isolation":"deltabox","isolation_provider":"sandlock","fallback_reason":null}}\n' "$id" "$PWD" ;;
    *partial*) printf '{"id":'; sleep 5 ;;
    *oversized*) head -c 2048 /dev/zero | tr '\000' x; sleep 5 ;;
    *remote-error*) printf '{"id":%s,"error":{"code":-32000,"message":"rejected","data":{"type":"project-conflict"}}}\n' "$id" ;;
    *session.exec*) printf '{"id":%s,"result":{"exit_code":0,"stdout":"ok","stderr":"","duration_ms":1,"truncated":false}}\n' "$id" ;;
    *) printf '{"id":%s,"result":{}}\n' "$id" ;;
  esac
done
"#;
        fs::write(&script_path, script).unwrap();
        let mut config = DeltaBoxConfig::from_binary("/bin/sh", directory.path());
        config.max_frame_bytes = 1024;
        let client = DeltaBoxClient::connect(config).unwrap();
        (directory, client)
    }

    #[test]
    fn executes_and_controls_the_same_session() {
        let (_directory, client) = fixture();
        assert_eq!(
            client
                .exec(
                    &["/bin/true".into()],
                    Some("bash"),
                    Duration::from_secs(1),
                    || false
                )
                .unwrap()
                .stdout,
            "ok"
        );
        client.diff().unwrap();
        client.commit().unwrap();
        client.abort().unwrap();
        client.stop().unwrap();
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn landlock_diagnostic_preserves_remote_error_and_distinguishes_remediation() {
        for (errno, expected) in [
            (libc::ENOSYS, "CONFIG_SECURITY_LANDLOCK=y"),
            (libc::EOPNOTSUPP, "boot-time LSM list"),
            (libc::EPERM, "access was denied"),
            (libc::EACCES, "access was denied"),
        ] {
            let error = with_landlock_diagnostic(
                DeltaBoxError::Remote {
                    code: Some(-32000),
                    message: "child control pipe closed".into(),
                    data: Some(json!({"type":"backend-error"})),
                },
                Err(io::Error::from_raw_os_error(errno)),
            );
            let DeltaBoxError::Remote {
                code,
                message,
                data,
            } = error
            else {
                panic!("diagnostic replaced the original error type");
            };
            assert_eq!(code, Some(-32000));
            assert_eq!(data, Some(json!({"type":"backend-error"})));
            assert!(message.starts_with("child control pipe closed;"));
            assert!(message.contains(expected), "{message}");
            assert!(message.contains("copy-private"));
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn supported_landlock_does_not_blame_kernel_for_remote_failure() {
        let error = with_landlock_diagnostic(
            DeltaBoxError::Remote {
                code: Some(-32000),
                message: "project conflict".into(),
                data: Some(json!({"path":"notes.txt"})),
            },
            Ok(3),
        );
        assert!(
            matches!(error, DeltaBoxError::Remote { message, data: Some(data), .. }
            if message == "project conflict" && data["path"] == "notes.txt")
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn unavailable_landlock_does_not_reclassify_unknown_execution() {
        let timeout = Duration::from_secs(1);
        let error = with_landlock_diagnostic(
            DeltaBoxError::TimedOut { timeout },
            Err(io::Error::from_raw_os_error(libc::ENOSYS)),
        );
        assert!(matches!(error, DeltaBoxError::TimedOut { timeout: actual } if actual == timeout));
    }

    #[test]
    fn partial_response_honors_deadline_and_control_can_reconnect() {
        let (_directory, client) = fixture();
        let started = Instant::now();
        assert!(matches!(
            client.exec(&["partial".into()], None, Duration::from_millis(60), || {
                false
            }),
            Err(DeltaBoxError::TimedOut { .. })
        ));
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(client.status().unwrap()["session_id"], "test-session");
    }

    #[test]
    fn oversized_response_is_bounded_before_newline() {
        let (_directory, client) = fixture();
        assert!(matches!(
            client.exec(&["oversized".into()], None, Duration::from_secs(1), || {
                false
            }),
            Err(DeltaBoxError::FrameTooLarge { limit: 1024 })
        ));
    }

    #[test]
    fn cancellation_interrupts_a_partial_response() {
        let (_directory, client) = fixture();
        let started = Instant::now();
        assert!(matches!(
            client.exec(
                &["partial".into()],
                None,
                Duration::from_secs(1),
                || started.elapsed() >= Duration::from_millis(30)
            ),
            Err(DeltaBoxError::Cancelled)
        ));
    }

    #[test]
    fn preserves_remote_conflict_data_without_replaying() {
        let (_directory, client) = fixture();
        assert!(
            matches!(client.exec(&["remote-error".into()], None, Duration::from_secs(1),
            || false), Err(DeltaBoxError::Remote { data: Some(data), .. })
            if data["type"] == "project-conflict")
        );
        client.diff().unwrap();
    }
    #[test]
    fn stop_deadline_includes_waiting_for_another_request() {
        let (_directory, mut client) = fixture();
        Arc::make_mut(&mut client.config).control_timeout = Duration::from_millis(30);
        let other = client.clone();
        let worker = thread::spawn(move || {
            other.exec(
                &["partial".into()],
                None,
                Duration::from_millis(200),
                || false,
            )
        });
        let waiting = Instant::now();
        while client.transport.try_lock().is_ok() {
            assert!(waiting.elapsed() < Duration::from_secs(1));
            thread::sleep(Duration::from_millis(1));
        }
        let started = Instant::now();
        assert!(matches!(client.stop(), Err(DeltaBoxError::TimedOut { .. })));
        assert!(started.elapsed() < Duration::from_millis(150));
        assert!(matches!(
            worker.join().unwrap(),
            Err(DeltaBoxError::TimedOut { .. })
        ));
    }

    #[test]
    #[ignore = "requires PI_AGENT_DELTABOX_BIN and a writable DELTABOX_STATE_DIR"]
    fn real_deltabox_private_files_commit_abort_and_conflicts() {
        let binary = env::var_os("PI_AGENT_DELTABOX_BIN").expect("set PI_AGENT_DELTABOX_BIN");
        let project = tempdir().unwrap();
        fs::write(project.path().join("notes.txt"), "original").unwrap();
        let mut config = DeltaBoxConfig::from_binary(binary, project.path());
        config.fs_mode = DeltaBoxFsMode::CopyPrivate;
        config.require_fs_mode = true;
        let client = DeltaBoxClient::connect(config.clone()).unwrap();
        let write = |content: &str| {
            client.request(
            "fs.write",
            json!({"session_id":client.session_id(),"path":"notes.txt","content_base64":content}),
            Duration::from_secs(5), || false,
        ).unwrap()
        };
        write("cHJpdmF0ZQ=="); // private
        assert_eq!(
            fs::read_to_string(project.path().join("notes.txt")).unwrap(),
            "original"
        );
        assert_eq!(client.diff().unwrap()["pending_changes"], 1);
        client.abort().unwrap();
        assert_eq!(client.diff().unwrap()["pending_changes"], 0);
        write("cHJpdmF0ZQ==");
        client.close_transport();
        let attached = DeltaBoxClient::attach(config, client.session_id()).unwrap();
        attached.commit().unwrap();
        assert_eq!(
            fs::read_to_string(project.path().join("notes.txt")).unwrap(),
            "private"
        );
        write("cGVuZGluZw=="); // pending
        fs::write(project.path().join("notes.txt"), "external-edit").unwrap();
        assert!(
            client.commit().is_err(),
            "host conflict must not be overwritten"
        );
        assert_eq!(
            fs::read_to_string(project.path().join("notes.txt")).unwrap(),
            "external-edit"
        );
        client.abort().unwrap();
        client.stop().unwrap();
        assert_eq!(
            fs::read_to_string(project.path().join("notes.txt")).unwrap(),
            "external-edit"
        );
    }
}
