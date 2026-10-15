//! Bubblewrap-backed command boundary for validated [`SandboxPolicy`] values.
//!
//! The executor owns the fixed bubblewrap arguments. A caller can provide a
//! program and its arguments, but cannot inject bubblewrap flags because the
//! user command is appended after the `--` separator.

use std::{
    env,
    ffi::{OsStr, OsString},
    fmt, fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use crate::policy::{NetworkMode, ResourceLimits, SandboxMode, SandboxPolicy};

const BWRAP_BINARY: &str = "bwrap";

/// A command line assembled by the sandbox executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxCommand {
    argv: Vec<OsString>,
    sandboxed: bool,
}

impl SandboxCommand {
    /// Returns the exact argv, useful for auditing and deterministic tests.
    pub fn argv(&self) -> &[OsString] {
        &self.argv
    }

    /// Whether this command is protected by the managed sandbox boundary.
    pub const fn sandboxed(&self) -> bool {
        self.sandboxed
    }

    /// Converts the assembled argv into a standard process command.
    pub fn into_command(self) -> Command {
        let mut argv = self.argv.into_iter();
        let program = argv.next().expect("sandbox command always has a program");
        let mut command = Command::new(program);
        command.args(argv);
        command
    }
}

/// Builds and executes commands under a validated sandbox policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxExecutor {
    bwrap: PathBuf,
}

impl SandboxExecutor {
    /// Finds `bwrap` on the current PATH.
    pub fn discover() -> Result<Self, SandboxError> {
        let path = env::var_os("PATH")
            .into_iter()
            .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
            .map(|directory| directory.join(BWRAP_BINARY))
            .find(|path| is_executable_file(path))
            .ok_or(SandboxError::SandboxUnavailable)?;
        Ok(Self { bwrap: path })
    }

    /// Probes whether the current host can actually construct a managed
    /// sandbox. Finding the binary alone is insufficient because namespace
    /// creation may still be denied by the kernel or container runtime.
    pub fn probe(&self) -> Result<(), SandboxError> {
        let policy = SandboxPolicy::managed("/tmp");
        let output = self
            .build_command(&policy, "/bin/true", [] as [&str; 0])?
            .into_command()
            .output()
            .map_err(SandboxError::Execution)?;
        if output.status.success() {
            return Ok(());
        }
        Err(SandboxError::CapabilityProbeFailed {
            status: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        })
    }

    /// Discovers bubblewrap and verifies that the managed boundary works.
    pub fn discover_and_probe() -> Result<Self, SandboxError> {
        let executor = Self::discover()?;
        executor.probe()?;
        Ok(executor)
    }

    /// Uses a caller-selected bubblewrap binary. Capability probing is kept in
    /// [`Self::discover`] so tests and external supervisors can provide a
    /// deterministic path without executing anything.
    pub fn from_path(path: impl Into<PathBuf>) -> Result<Self, SandboxError> {
        let path = path.into();
        if !path.is_absolute() || path.file_name().is_none() {
            return Err(SandboxError::InvalidBubblewrapPath(path));
        }
        Ok(Self { bwrap: path })
    }

    /// Builds a command without starting a process.
    pub fn build_command<I, S>(
        &self,
        policy: &SandboxPolicy,
        program: impl AsRef<Path>,
        args: I,
    ) -> Result<SandboxCommand, SandboxError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        policy.validate().map_err(SandboxError::InvalidPolicy)?;
        let program = program.as_ref();
        if program.as_os_str().is_empty() {
            return Err(SandboxError::EmptyProgram);
        }

        match policy.mode {
            SandboxMode::Disabled => {
                validate_executor_capabilities(policy)?;
                let mut argv = vec![program.as_os_str().to_os_string()];
                argv.extend(
                    args.into_iter()
                        .map(|argument| argument.as_ref().to_os_string()),
                );
                Ok(SandboxCommand {
                    argv,
                    sandboxed: false,
                })
            }
            SandboxMode::External => Err(SandboxError::ExternalSupervisorRequired),
            SandboxMode::Managed => {
                validate_executor_capabilities(policy)?;
                let mut argv = self.managed_prefix(policy);
                argv.push(OsString::from("--"));
                argv.push(program.as_os_str().to_os_string());
                argv.extend(
                    args.into_iter()
                        .map(|argument| argument.as_ref().to_os_string()),
                );
                Ok(SandboxCommand {
                    argv,
                    sandboxed: true,
                })
            }
        }
    }

    /// Executes a command after building and validating its boundary.
    pub fn execute<I, S>(
        &self,
        policy: &SandboxPolicy,
        program: impl AsRef<Path>,
        args: I,
    ) -> Result<Output, SandboxError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.build_command(policy, program, args)?
            .into_command()
            .output()
            .map_err(SandboxError::Execution)
    }

    fn managed_prefix(&self, policy: &SandboxPolicy) -> Vec<OsString> {
        let mut argv = vec![self.bwrap.as_os_str().to_os_string()];
        argv.extend([
            OsString::from("--die-with-parent"),
            OsString::from("--new-session"),
            OsString::from("--unshare-user"),
            OsString::from("--unshare-pid"),
            OsString::from("--unshare-net"),
            OsString::from("--cap-drop"),
            OsString::from("ALL"),
            OsString::from("--clearenv"),
            OsString::from("--setenv"),
            OsString::from("PATH"),
            OsString::from("/usr/bin:/bin"),
            OsString::from("--setenv"),
            OsString::from("HOME"),
            OsString::from("/tmp"),
            OsString::from("--ro-bind"),
            OsString::from("/"),
            OsString::from("/"),
            OsString::from("--dev"),
            OsString::from("/dev"),
            OsString::from("--proc"),
            OsString::from("/proc"),
            OsString::from("--tmpfs"),
            OsString::from("/tmp"),
        ]);

        let workspace = policy.filesystem.workspace.as_os_str();
        argv.push(if policy.filesystem.workspace_writable {
            OsString::from("--bind")
        } else {
            OsString::from("--ro-bind")
        });
        argv.push(workspace.to_os_string());
        argv.push(workspace.to_os_string());
        for path in &policy.filesystem.read_only_paths {
            argv.push(OsString::from("--ro-bind"));
            argv.push(path.as_os_str().to_os_string());
            argv.push(path.as_os_str().to_os_string());
        }
        argv.push(OsString::from("--chdir"));
        argv.push(workspace.to_os_string());
        argv
    }
}

/// Errors raised before or during sandbox command execution.
#[derive(Debug)]
pub enum SandboxError {
    SandboxUnavailable,
    InvalidBubblewrapPath(PathBuf),
    InvalidPolicy(crate::policy::PolicyError),
    ExternalSupervisorRequired,
    UnsupportedNetwork(NetworkMode),
    UnsupportedResources(ResourceLimits),
    CapabilityProbeFailed { status: Option<i32>, stderr: String },
    EmptyProgram,
    Execution(std::io::Error),
}

impl fmt::Display for SandboxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SandboxUnavailable => formatter.write_str("bubblewrap is unavailable"),
            Self::InvalidBubblewrapPath(path) => {
                write!(formatter, "invalid bubblewrap path: {}", path.display())
            }
            Self::InvalidPolicy(error) => write!(formatter, "invalid sandbox policy: {error}"),
            Self::ExternalSupervisorRequired => {
                formatter.write_str("sandbox policy requires an external supervisor")
            }
            Self::UnsupportedNetwork(network) => {
                write!(
                    formatter,
                    "sandbox executor does not support network mode {network:?}"
                )
            }
            Self::UnsupportedResources(_) => {
                formatter.write_str("sandbox executor does not support requested resource limits")
            }
            Self::CapabilityProbeFailed { status, stderr } => write!(
                formatter,
                "managed sandbox capability probe failed (status {status:?}): {stderr}"
            ),
            Self::EmptyProgram => formatter.write_str("sandbox program must not be empty"),
            Self::Execution(error) => write!(formatter, "sandbox execution failed: {error}"),
        }
    }
}

impl std::error::Error for SandboxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPolicy(error) => Some(error),
            Self::Execution(error) => Some(error),
            Self::SandboxUnavailable
            | Self::InvalidBubblewrapPath(_)
            | Self::ExternalSupervisorRequired
            | Self::UnsupportedNetwork(_)
            | Self::UnsupportedResources(_)
            | Self::CapabilityProbeFailed { .. }
            | Self::EmptyProgram => None,
        }
    }
}

fn validate_executor_capabilities(policy: &SandboxPolicy) -> Result<(), SandboxError> {
    if !matches!(policy.network, NetworkMode::None) {
        return Err(SandboxError::UnsupportedNetwork(policy.network.clone()));
    }
    if policy.resources != ResourceLimits::default() {
        return Err(SandboxError::UnsupportedResources(policy.resources));
    }
    Ok(())
}

fn is_executable_file(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|metadata| {
        if !metadata.is_file() {
            return false;
        }

        #[cfg(unix)]
        {
            metadata.permissions().mode() & 0o111 != 0
        }
        #[cfg(not(unix))]
        {
            true
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{FilesystemPolicy, NamespacePolicy};

    fn executor() -> SandboxExecutor {
        SandboxExecutor::from_path("/usr/bin/bwrap").expect("absolute bwrap path should be valid")
    }

    #[test]
    fn managed_command_places_user_program_after_separator() {
        let policy = SandboxPolicy::managed("/workspace");
        let command = executor()
            .build_command(&policy, "/bin/echo", ["--unshare-user", "hello"])
            .expect("managed command should build");
        let argv = command.argv();
        let separator = argv
            .iter()
            .position(|argument| argument == OsStr::new("--"))
            .expect("argv should contain command separator");
        assert!(command.sandboxed());
        assert_eq!(argv[separator + 1], OsStr::new("/bin/echo"));
        assert_eq!(argv[separator + 2], OsStr::new("--unshare-user"));
        assert!(argv[..separator]
            .windows(2)
            .any(|window| window == [OsString::from("--cap-drop"), OsString::from("ALL")]));
    }

    #[test]
    fn disabled_policy_is_explicit_host_execution() {
        let policy = SandboxPolicy::disabled("/workspace");
        let command = executor()
            .build_command(&policy, "/bin/echo", ["hello"])
            .expect("disabled command should build");
        assert!(!command.sandboxed());
        assert_eq!(
            command.argv(),
            &[OsString::from("/bin/echo"), OsString::from("hello")]
        );
    }

    #[test]
    fn external_and_unsupported_capabilities_fail_closed() {
        let external = SandboxPolicy {
            mode: SandboxMode::External,
            ..SandboxPolicy::managed("/workspace")
        };
        assert!(matches!(
            executor().build_command(&external, "/bin/true", [] as [&str; 0]),
            Err(SandboxError::ExternalSupervisorRequired)
        ));

        let network = SandboxPolicy {
            network: NetworkMode::Loopback,
            ..SandboxPolicy::managed("/workspace")
        };
        assert!(matches!(
            executor().build_command(&network, "/bin/true", [] as [&str; 0]),
            Err(SandboxError::UnsupportedNetwork(NetworkMode::Loopback))
        ));

        let resources = SandboxPolicy {
            resources: ResourceLimits {
                wall_time_ms: Some(1_000),
                ..ResourceLimits::default()
            },
            ..SandboxPolicy::managed("/workspace")
        };
        assert!(matches!(
            executor().build_command(&resources, "/bin/true", [] as [&str; 0]),
            Err(SandboxError::UnsupportedResources(_))
        ));
    }

    #[test]
    fn read_only_workspace_uses_read_only_bind() {
        let policy = SandboxPolicy {
            filesystem: FilesystemPolicy {
                workspace: PathBuf::from("/workspace"),
                workspace_writable: false,
                read_only_paths: vec![PathBuf::from("/workspace/config")],
            },
            namespaces: NamespacePolicy {
                mount: true,
                user: true,
                pid: true,
                network: true,
            },
            ..SandboxPolicy::managed("/workspace")
        };
        let command = executor()
            .build_command(&policy, "/bin/true", [] as [&str; 0])
            .expect("read-only command should build");
        let argv = command.argv();
        assert!(argv.windows(3).any(|window| window
            == [
                OsString::from("--ro-bind"),
                OsString::from("/workspace"),
                OsString::from("/workspace")
            ]));
    }

    #[cfg(unix)]
    #[test]
    fn discovery_ignores_regular_files_without_execute_permission() {
        let path = std::env::temp_dir().join(format!(
            "pi-agent-runtime-non-executable-{}",
            uuid::Uuid::new_v4()
        ));
        fs::write(&path, b"not executable").expect("fixture should write");
        assert!(!is_executable_file(&path));

        let mut permissions = fs::metadata(&path)
            .expect("fixture metadata should be readable")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).expect("fixture should become executable");
        assert!(is_executable_file(&path));

        fs::remove_file(path).expect("fixture should remove");
    }
}
