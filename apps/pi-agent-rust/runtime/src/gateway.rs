//! A bounded process gateway for tools that must cross the runtime boundary.
//!
//! The gateway deliberately accepts a program and an argument vector rather
//! than a shell command string. This keeps argument parsing out of the
//! security boundary and lets [`SandboxExecutor`](crate::SandboxExecutor)
//! own the fixed sandbox policy. It is suitable for external tools and
//! controllers; in-process Rust tools should continue to use their native
//! tool API.

use std::{
    ffi::{OsStr, OsString},
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Child, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use crate::{PolicyError, SandboxError, SandboxExecutor, SandboxPolicy};

/// A direct executable invocation with no shell interpretation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSpec {
    program: PathBuf,
    args: Vec<OsString>,
}

impl CommandSpec {
    /// Creates a command for an absolute executable path.
    pub fn new(program: impl AsRef<Path>) -> Result<Self, CommandSpecError> {
        let program = program.as_ref().to_path_buf();
        if program.as_os_str().is_empty() {
            return Err(CommandSpecError::EmptyProgram);
        }
        if !program.is_absolute() {
            return Err(CommandSpecError::ProgramMustBeAbsolute(program));
        }
        Ok(Self {
            program,
            args: Vec::new(),
        })
    }

    /// Appends one argument without interpreting it as shell syntax.
    pub fn arg(mut self, argument: impl AsRef<OsStr>) -> Self {
        self.args.push(argument.as_ref().to_os_string());
        self
    }

    /// Appends several arguments without interpreting them as shell syntax.
    pub fn args<I, S>(mut self, arguments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.args.extend(
            arguments
                .into_iter()
                .map(|argument| argument.as_ref().to_os_string()),
        );
        self
    }

    pub fn program(&self) -> &Path {
        &self.program
    }

    pub fn arguments(&self) -> &[OsString] {
        &self.args
    }
}

/// Errors raised while constructing an external command specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandSpecError {
    EmptyProgram,
    ProgramMustBeAbsolute(PathBuf),
}

impl std::fmt::Display for CommandSpecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyProgram => formatter.write_str("command program must not be empty"),
            Self::ProgramMustBeAbsolute(path) => {
                write!(
                    formatter,
                    "command program must be absolute: {}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for CommandSpecError {}

/// Which output stream exceeded the configured capture limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputStream {
    Stdout,
    Stderr,
}

/// Bounded result of one external command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    pub status_code: Option<i32>,
    pub success: bool,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub sandboxed: bool,
    /// Wall-clock time spent preparing and waiting for the command, in milliseconds.
    pub duration_ms: u64,
}

/// Runtime-owned process gateway.
#[derive(Debug, Clone)]
pub struct CommandGateway {
    executor: SandboxExecutor,
    policy: SandboxPolicy,
    max_output_bytes: usize,
    timeout: Option<Duration>,
}

impl CommandGateway {
    /// Creates a gateway with a 1 MiB per-stream output limit.
    pub fn new(
        executor: SandboxExecutor,
        policy: SandboxPolicy,
    ) -> Result<Self, CommandGatewayError> {
        policy
            .validate()
            .map_err(CommandGatewayError::InvalidPolicy)?;
        Ok(Self {
            executor,
            policy,
            max_output_bytes: 1024 * 1024,
            timeout: None,
        })
    }

    /// Replaces the per-stream output limit.
    pub fn with_max_output_bytes(
        mut self,
        max_output_bytes: usize,
    ) -> Result<Self, CommandGatewayError> {
        if max_output_bytes == 0 {
            return Err(CommandGatewayError::InvalidOutputLimit);
        }
        self.max_output_bytes = max_output_bytes;
        Ok(self)
    }

    /// Sets an optional wall-time limit for one child process.
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self, CommandGatewayError> {
        if timeout.is_zero() {
            return Err(CommandGatewayError::InvalidTimeout);
        }
        self.timeout = Some(timeout);
        Ok(self)
    }

    pub fn policy(&self) -> &SandboxPolicy {
        &self.policy
    }

    /// Executes one direct command and captures both streams with bounds.
    ///
    /// The policy workspace must already exist and be a directory. The
    /// gateway checks this before creating a child process so a bad runtime
    /// configuration cannot be mistaken for a command failure.
    pub fn execute(&self, spec: &CommandSpec) -> Result<CommandOutput, CommandGatewayError> {
        self.execute_with_cancel(spec, || false)
    }

    /// Executes a command while polling a caller-owned cancellation signal.
    ///
    /// The callback is evaluated by the waiting thread, so a caller can
    /// connect an agent cancellation token without making this crate depend
    /// on the agent implementation. Cancellation kills and waits for the
    /// child before returning, avoiding an orphaned process.
    pub fn execute_with_cancel<F>(
        &self,
        spec: &CommandSpec,
        should_cancel: F,
    ) -> Result<CommandOutput, CommandGatewayError>
    where
        F: Fn() -> bool,
    {
        let started = Instant::now();
        let workspace = &self.policy.filesystem.workspace;
        let metadata = fs::metadata(workspace).map_err(|source| {
            CommandGatewayError::WorkspaceUnavailable {
                path: workspace.clone(),
                source,
            }
        })?;
        if !metadata.is_dir() {
            return Err(CommandGatewayError::WorkspaceNotDirectory {
                path: workspace.clone(),
            });
        }
        let command =
            self.executor
                .build_command(&self.policy, spec.program(), spec.arguments())?;
        let sandboxed = command.sandboxed();
        let mut process = command.into_command();
        configure_process_group(&mut process);
        process.current_dir(workspace);
        let mut child = process
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(CommandGatewayError::Spawn)?;

        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => return terminate_child(child, CommandGatewayError::MissingStdout),
        };
        let stderr = match child.stderr.take() {
            Some(stderr) => stderr,
            None => return terminate_child(child, CommandGatewayError::MissingStderr),
        };
        let stop_requested = Arc::new(AtomicBool::new(false));
        let stdout_stop = Arc::clone(&stop_requested);
        let stderr_stop = Arc::clone(&stop_requested);
        let limit = self.max_output_bytes;
        let stdout_thread = thread::spawn(move || capture_stream(stdout, limit, stdout_stop));
        let stderr_thread = thread::spawn(move || capture_stream(stderr, limit, stderr_stop));

        let status = match wait_for_child(&mut child, &stop_requested, should_cancel, self.timeout)
        {
            Ok(status) => status,
            Err(error) => {
                // A wait failure leaves ownership of the child here. Kill and
                // reap it before returning, then join both capture workers so
                // a transient OS error cannot leak a process or thread.
                kill_process_tree(&mut child);
                let _ = child.wait();
                let _ = join_stream(stdout_thread, OutputStream::Stdout);
                let _ = join_stream(stderr_thread, OutputStream::Stderr);
                return Err(CommandGatewayError::Wait(error));
            }
        };
        let stdout_result = join_stream(stdout_thread, OutputStream::Stdout);
        let stderr_result = join_stream(stderr_thread, OutputStream::Stderr);
        let stdout = stdout_result?;
        let stderr = stderr_result?;

        let status = match status {
            WaitOutcome::Completed(status) => status,
            WaitOutcome::Cancelled => return Err(CommandGatewayError::Cancelled),
            WaitOutcome::TimedOut { timeout } => {
                return Err(CommandGatewayError::TimedOut { timeout })
            }
        };

        if stdout.exceeded {
            return Err(CommandGatewayError::OutputLimitExceeded {
                stream: OutputStream::Stdout,
                limit: self.max_output_bytes,
            });
        }
        if stderr.exceeded {
            return Err(CommandGatewayError::OutputLimitExceeded {
                stream: OutputStream::Stderr,
                limit: self.max_output_bytes,
            });
        }

        Ok(CommandOutput {
            status_code: status.code(),
            success: status.success(),
            stdout: stdout.bytes,
            stderr: stderr.bytes,
            sandboxed,
            duration_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        })
    }
}

/// Errors raised before, during, or after a gateway command.
#[derive(Debug)]
pub enum CommandGatewayError {
    InvalidPolicy(PolicyError),
    Sandbox(SandboxError),
    Spawn(io::Error),
    WorkspaceUnavailable { path: PathBuf, source: io::Error },
    WorkspaceNotDirectory { path: PathBuf },
    Wait(io::Error),
    MissingStdout,
    MissingStderr,
    Cancelled,
    TimedOut { timeout: Duration },
    CapturePanicked(OutputStream),
    Capture(io::Error),
    InvalidOutputLimit,
    InvalidTimeout,
    OutputLimitExceeded { stream: OutputStream, limit: usize },
}

impl CommandGatewayError {
    /// Returns a stable machine-readable category for tool adapters.
    pub const fn status(&self) -> &'static str {
        match self {
            Self::InvalidPolicy(_) => "invalid_policy",
            Self::Sandbox(_) => "sandbox_rejected",
            Self::Spawn(_) => "spawn_failed",
            Self::WorkspaceUnavailable { .. } => "workspace_unavailable",
            Self::WorkspaceNotDirectory { .. } => "workspace_not_directory",
            Self::Wait(_) => "wait_failed",
            Self::MissingStdout => "missing_stdout",
            Self::MissingStderr => "missing_stderr",
            Self::Cancelled => "cancelled",
            Self::TimedOut { .. } => "timed_out",
            Self::CapturePanicked(_) => "capture_panicked",
            Self::Capture(_) => "capture_failed",
            Self::InvalidOutputLimit => "invalid_output_limit",
            Self::InvalidTimeout => "invalid_timeout",
            Self::OutputLimitExceeded { .. } => "output_limit",
        }
    }
}

impl std::fmt::Display for CommandGatewayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPolicy(error) => write!(formatter, "invalid command policy: {error}"),
            Self::Sandbox(error) => write!(formatter, "sandbox command rejected: {error}"),
            Self::Spawn(error) => write!(formatter, "failed to spawn command: {error}"),
            Self::WorkspaceUnavailable { path, source } => write!(
                formatter,
                "command workspace {} is unavailable: {source}",
                path.display()
            ),
            Self::WorkspaceNotDirectory { path } => write!(
                formatter,
                "command workspace is not a directory: {}",
                path.display()
            ),
            Self::Wait(error) => write!(formatter, "failed to wait for command: {error}"),
            Self::MissingStdout => formatter.write_str("command stdout pipe was unavailable"),
            Self::MissingStderr => formatter.write_str("command stderr pipe was unavailable"),
            Self::Cancelled => formatter.write_str("command execution was cancelled"),
            Self::TimedOut { timeout } => {
                write!(
                    formatter,
                    "command exceeded its {:?} wall-time limit",
                    timeout
                )
            }
            Self::CapturePanicked(stream) => {
                write!(formatter, "{stream:?} capture thread panicked")
            }
            Self::Capture(error) => write!(formatter, "failed to capture command output: {error}"),
            Self::InvalidOutputLimit => {
                formatter.write_str("command output limit must be positive")
            }
            Self::InvalidTimeout => formatter.write_str("command timeout must be positive"),
            Self::OutputLimitExceeded { stream, limit } => {
                write!(
                    formatter,
                    "command {stream:?} exceeded the {limit}-byte limit"
                )
            }
        }
    }
}

impl std::error::Error for CommandGatewayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPolicy(error) => Some(error),
            Self::Sandbox(error) => Some(error),
            Self::Spawn(error)
            | Self::Wait(error)
            | Self::Capture(error)
            | Self::WorkspaceUnavailable { source: error, .. } => Some(error),
            Self::MissingStdout
            | Self::MissingStderr
            | Self::WorkspaceNotDirectory { .. }
            | Self::Cancelled
            | Self::TimedOut { .. }
            | Self::CapturePanicked(_)
            | Self::InvalidOutputLimit
            | Self::InvalidTimeout
            | Self::OutputLimitExceeded { .. } => None,
        }
    }
}

impl From<SandboxError> for CommandGatewayError {
    fn from(error: SandboxError) -> Self {
        Self::Sandbox(error)
    }
}

struct CapturedStream {
    bytes: Vec<u8>,
    exceeded: bool,
}

fn capture_stream<R: Read>(
    mut reader: R,
    limit: usize,
    stop_requested: Arc<AtomicBool>,
) -> io::Result<CapturedStream> {
    let mut bytes = Vec::with_capacity(limit.min(8192));
    let mut chunk = [0_u8; 8192];
    loop {
        let remaining = limit.saturating_add(1).saturating_sub(bytes.len());
        if remaining == 0 {
            stop_requested.store(true, Ordering::Release);
            return Ok(CapturedStream {
                bytes,
                exceeded: true,
            });
        }
        let read_len = remaining.min(chunk.len());
        match reader.read(&mut chunk[..read_len]) {
            Ok(0) => {
                return Ok(CapturedStream {
                    bytes,
                    exceeded: false,
                });
            }
            Ok(read) => {
                bytes.extend_from_slice(&chunk[..read]);
                if bytes.len() > limit {
                    stop_requested.store(true, Ordering::Release);
                    return Ok(CapturedStream {
                        bytes,
                        exceeded: true,
                    });
                }
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                stop_requested.store(true, Ordering::Release);
                return Err(error);
            }
        }
    }
}

fn terminate_child<T>(
    mut child: Child,
    error: CommandGatewayError,
) -> Result<T, CommandGatewayError> {
    kill_process_tree(&mut child);
    let _ = child.wait();
    Err(error)
}

#[cfg(unix)]
fn configure_process_group(command: &mut std::process::Command) {
    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut std::process::Command) {}

fn kill_process_tree(child: &mut Child) {
    #[cfg(unix)]
    if let Ok(pid) = libc::pid_t::try_from(child.id()) {
        // The command is placed in a dedicated process group before spawn.
        // A negative pid addresses that group and also terminates descendants.
        unsafe {
            let _ = libc::kill(-pid, libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

fn wait_for_child(
    child: &mut Child,
    stop_requested: &AtomicBool,
    should_cancel: impl Fn() -> bool,
    timeout: Option<Duration>,
) -> io::Result<WaitOutcome> {
    let started = Instant::now();
    loop {
        if should_cancel() {
            kill_process_tree(child);
            child.wait()?;
            return Ok(WaitOutcome::Cancelled);
        }
        if let Some(timeout) = timeout {
            if started.elapsed() >= timeout {
                kill_process_tree(child);
                child.wait()?;
                return Ok(WaitOutcome::TimedOut { timeout });
            }
        }
        if stop_requested.load(Ordering::Acquire) {
            kill_process_tree(child);
        }
        if let Some(status) = child.try_wait()? {
            return Ok(WaitOutcome::Completed(status));
        }
        thread::sleep(Duration::from_millis(1));
    }
}

enum WaitOutcome {
    Completed(std::process::ExitStatus),
    Cancelled,
    TimedOut { timeout: Duration },
}

fn join_stream(
    handle: thread::JoinHandle<io::Result<CapturedStream>>,
    stream: OutputStream,
) -> Result<CapturedStream, CommandGatewayError> {
    match handle.join() {
        Ok(result) => result.map_err(CommandGatewayError::Capture),
        Err(_) => Err(CommandGatewayError::CapturePanicked(stream)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disabled_gateway() -> CommandGateway {
        CommandGateway::new(
            SandboxExecutor::from_path("/usr/bin/bwrap").expect("absolute bwrap path is valid"),
            SandboxPolicy::disabled("/tmp"),
        )
        .expect("disabled policy should validate")
    }

    #[test]
    fn command_spec_requires_an_absolute_program() {
        assert!(matches!(
            CommandSpec::new("printf"),
            Err(CommandSpecError::ProgramMustBeAbsolute(_))
        ));
    }

    #[test]
    fn disabled_gateway_executes_direct_argv() {
        let output = disabled_gateway()
            .execute(&CommandSpec::new("/usr/bin/printf").unwrap().arg("hello"))
            .expect("printf should execute");
        assert!(output.success);
        assert_eq!(output.stdout, b"hello");
        assert!(output.stderr.is_empty());
        assert!(!output.sandboxed);
    }

    #[test]
    fn gateway_terminates_when_a_stream_exceeds_the_limit() {
        let gateway = disabled_gateway()
            .with_max_output_bytes(8)
            .expect("positive output limit should validate");
        let error = gateway
            .execute(
                &CommandSpec::new("/usr/bin/printf")
                    .unwrap()
                    .arg("0123456789"),
            )
            .expect_err("output above the limit should fail");
        assert!(matches!(
            error,
            CommandGatewayError::OutputLimitExceeded {
                stream: OutputStream::Stdout,
                limit: 8
            }
        ));
    }

    #[test]
    fn gateway_kills_a_cancelled_child() {
        let error = disabled_gateway()
            .execute_with_cancel(
                &CommandSpec::new("/usr/bin/sleep").unwrap().arg("5"),
                || true,
            )
            .expect_err("cancelled command should fail explicitly");
        assert!(matches!(error, CommandGatewayError::Cancelled));
    }

    #[test]
    fn gateway_rejects_a_zero_timeout() {
        let error = disabled_gateway()
            .with_timeout(Duration::ZERO)
            .expect_err("a zero timeout would terminate every command immediately");
        assert!(matches!(error, CommandGatewayError::InvalidTimeout));
    }

    #[test]
    fn gateway_kills_a_child_that_exceeds_its_timeout() {
        let gateway = disabled_gateway()
            .with_timeout(Duration::from_millis(10))
            .expect("positive timeout should validate");
        let error = gateway
            .execute(&CommandSpec::new("/usr/bin/sleep").unwrap().arg("5"))
            .expect_err("a command beyond the timeout should fail explicitly");
        assert!(matches!(
            error,
            CommandGatewayError::TimedOut { timeout } if timeout == Duration::from_millis(10)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn gateway_timeout_kills_descendant_processes() {
        let gateway = disabled_gateway()
            .with_timeout(Duration::from_millis(10))
            .expect("positive timeout should validate");
        let started = Instant::now();
        let error = gateway
            .execute(&CommandSpec::new("/bin/sh").unwrap().args(["-c", "sleep 5"]))
            .expect_err("a command tree beyond the timeout should fail explicitly");
        assert!(matches!(error, CommandGatewayError::TimedOut { .. }));
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "descendant process kept the output pipe open"
        );
    }

    #[test]
    fn disabled_gateway_starts_in_the_policy_workspace() {
        let output = disabled_gateway()
            .execute(&CommandSpec::new("/bin/pwd").expect("pwd path should be valid"))
            .expect("pwd should execute");
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "/tmp");
    }

    #[test]
    fn gateway_reports_a_missing_workspace_before_spawning() {
        let workspace = std::env::temp_dir().join(format!(
            "pi-agent-runtime-missing-workspace-{}",
            uuid::Uuid::new_v4()
        ));
        let gateway = CommandGateway::new(
            SandboxExecutor::from_path("/usr/bin/bwrap").expect("absolute bwrap path is valid"),
            SandboxPolicy::disabled(&workspace),
        )
        .expect("disabled policy should validate");

        let error = gateway
            .execute(&CommandSpec::new("/bin/true").expect("true path should be valid"))
            .expect_err("a missing workspace must stop execution before spawn");
        assert!(matches!(
            error,
            CommandGatewayError::WorkspaceUnavailable { path, .. } if path == workspace
        ));
    }

    #[test]
    fn gateway_rejects_a_workspace_file() {
        let workspace = std::env::temp_dir().join(format!(
            "pi-agent-runtime-workspace-file-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&workspace, b"not a directory").expect("workspace fixture should write");
        let gateway = CommandGateway::new(
            SandboxExecutor::from_path("/usr/bin/bwrap").expect("absolute bwrap path is valid"),
            SandboxPolicy::disabled(&workspace),
        )
        .expect("disabled policy should validate");

        let error = gateway
            .execute(&CommandSpec::new("/bin/true").expect("true path should be valid"))
            .expect_err("a file cannot be used as a workspace");
        assert!(matches!(
            error,
            CommandGatewayError::WorkspaceNotDirectory { path } if path == workspace
        ));
        std::fs::remove_file(workspace).expect("workspace fixture should remove");
    }

    #[test]
    fn gateway_errors_expose_stable_status_categories() {
        assert_eq!(
            CommandGatewayError::InvalidTimeout.status(),
            "invalid_timeout"
        );
        assert_eq!(
            CommandGatewayError::OutputLimitExceeded {
                stream: OutputStream::Stdout,
                limit: 8,
            }
            .status(),
            "output_limit"
        );
    }
}
