//! Typed sandbox policy shared by runtime admission and sandbox executors.
//!
//! This module describes the requested isolation boundary. It does not probe
//! the host or execute a command; those side effects belong to `sandbox.rs`.
//! Keeping validation here prevents an executor from silently weakening a
//! policy when the host lacks a required capability.

use std::{
    collections::BTreeSet,
    fmt,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};

/// How the caller expects a policy to be enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxMode {
    /// Runtime owns sandbox construction and must fail closed if unavailable.
    Managed,
    /// An external supervisor owns construction and reports its capabilities.
    External,
    /// No sandbox is requested. This must be an explicit choice by the caller.
    Disabled,
}

/// Network visibility granted to an action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkMode {
    /// No network interfaces are visible inside the sandbox.
    None,
    /// Only the loopback interface is available.
    Loopback,
    /// A later executor must enforce this host allowlist.
    Allowlist(Vec<String>),
}

/// Filesystem boundary presented to an action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilesystemPolicy {
    /// Absolute workspace path mounted into the sandbox.
    pub workspace: PathBuf,
    /// Whether the mounted workspace may be modified by the action.
    pub workspace_writable: bool,
    /// Absolute paths that must remain read-only even when nested in a
    /// writable workspace.
    #[serde(default)]
    pub read_only_paths: Vec<PathBuf>,
}

/// Namespace isolation requested from the executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamespacePolicy {
    pub mount: bool,
    pub user: bool,
    pub pid: bool,
    pub network: bool,
}

/// Bounded resources for one action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ResourceLimits {
    pub wall_time_ms: Option<u64>,
    pub cpu_time_ms: Option<u64>,
    pub memory_bytes: Option<u64>,
    pub process_count: Option<u32>,
    pub output_bytes: Option<u64>,
}

/// Complete sandbox request passed from runtime to an executor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxPolicy {
    pub mode: SandboxMode,
    pub filesystem: FilesystemPolicy,
    pub network: NetworkMode,
    pub namespaces: NamespacePolicy,
    pub resources: ResourceLimits,
}

impl SandboxPolicy {
    /// Builds a managed policy with all namespace boundaries enabled.
    pub fn managed(workspace: impl Into<PathBuf>) -> Self {
        Self {
            mode: SandboxMode::Managed,
            filesystem: FilesystemPolicy {
                workspace: workspace.into(),
                workspace_writable: true,
                read_only_paths: Vec::new(),
            },
            network: NetworkMode::None,
            namespaces: NamespacePolicy {
                mount: true,
                user: true,
                pid: true,
                network: true,
            },
            resources: ResourceLimits::default(),
        }
    }

    /// Builds an explicitly unsandboxed policy for controlled environments.
    pub fn disabled(workspace: impl Into<PathBuf>) -> Self {
        Self {
            mode: SandboxMode::Disabled,
            filesystem: FilesystemPolicy {
                workspace: workspace.into(),
                workspace_writable: true,
                read_only_paths: Vec::new(),
            },
            network: NetworkMode::None,
            namespaces: NamespacePolicy {
                mount: false,
                user: false,
                pid: false,
                network: false,
            },
            resources: ResourceLimits::default(),
        }
    }

    /// Validates invariants that must hold before capability probing.
    pub fn validate(&self) -> Result<(), PolicyError> {
        validate_absolute_path("workspace", &self.filesystem.workspace)?;
        for path in &self.filesystem.read_only_paths {
            validate_absolute_path("read-only path", path)?;
        }

        let mut paths = BTreeSet::new();
        for path in &self.filesystem.read_only_paths {
            if !paths.insert(path) {
                return Err(PolicyError::DuplicateReadOnlyPath(path.clone()));
            }
        }

        if self.mode == SandboxMode::Managed
            && (!self.namespaces.mount || !self.namespaces.user || !self.namespaces.pid)
        {
            return Err(PolicyError::ManagedNamespacesRequired);
        }
        if self.mode == SandboxMode::Managed && !self.namespaces.network {
            return Err(PolicyError::ManagedNetworkNamespaceRequired);
        }
        validate_network(&self.network)?;
        validate_limits(self.resources)
    }
}

/// Validation failures that must be reported before an executor is selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    RelativePath { field: &'static str, path: PathBuf },
    ParentTraversal { field: &'static str, path: PathBuf },
    DuplicateReadOnlyPath(PathBuf),
    ManagedNamespacesRequired,
    ManagedNetworkNamespaceRequired,
    EmptyNetworkAllowlist,
    InvalidNetworkHost(String),
    DuplicateNetworkHost(String),
    NonPositiveLimit { field: &'static str },
}

impl fmt::Display for PolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RelativePath { field, path } => {
                write!(
                    formatter,
                    "sandbox {field} must be absolute: {}",
                    path.display()
                )
            }
            Self::ParentTraversal { field, path } => write!(
                formatter,
                "sandbox {field} must not contain parent traversal: {}",
                path.display()
            ),
            Self::DuplicateReadOnlyPath(path) => write!(
                formatter,
                "sandbox read-only path is duplicated: {}",
                path.display()
            ),
            Self::ManagedNamespacesRequired => write!(
                formatter,
                "managed sandbox requires mount, user, and pid namespaces"
            ),
            Self::ManagedNetworkNamespaceRequired => {
                write!(formatter, "managed sandbox requires a network namespace")
            }
            Self::EmptyNetworkAllowlist => {
                formatter.write_str("sandbox network allowlist must not be empty")
            }
            Self::InvalidNetworkHost(host) => {
                write!(formatter, "sandbox network host is invalid: {host:?}")
            }
            Self::DuplicateNetworkHost(host) => {
                write!(formatter, "sandbox network host is duplicated: {host:?}")
            }
            Self::NonPositiveLimit { field } => {
                write!(formatter, "sandbox resource limit {field} must be positive")
            }
        }
    }
}

impl std::error::Error for PolicyError {}

fn validate_absolute_path(field: &'static str, path: &Path) -> Result<(), PolicyError> {
    if !path.is_absolute() {
        return Err(PolicyError::RelativePath {
            field,
            path: path.to_path_buf(),
        });
    }
    if path
        .components()
        .any(|component| component == Component::ParentDir)
    {
        return Err(PolicyError::ParentTraversal {
            field,
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn validate_limits(limits: ResourceLimits) -> Result<(), PolicyError> {
    for (field, value) in [
        ("wall_time_ms", limits.wall_time_ms),
        ("cpu_time_ms", limits.cpu_time_ms),
        ("memory_bytes", limits.memory_bytes),
        ("output_bytes", limits.output_bytes),
    ] {
        if value == Some(0) {
            return Err(PolicyError::NonPositiveLimit { field });
        }
    }
    if limits.process_count == Some(0) {
        return Err(PolicyError::NonPositiveLimit {
            field: "process_count",
        });
    }
    Ok(())
}

fn validate_network(network: &NetworkMode) -> Result<(), PolicyError> {
    let NetworkMode::Allowlist(hosts) = network else {
        return Ok(());
    };
    if hosts.is_empty() {
        return Err(PolicyError::EmptyNetworkAllowlist);
    }
    let mut unique = BTreeSet::new();
    for host in hosts {
        if host.trim().is_empty() || host.trim() != host || host.chars().any(char::is_whitespace) {
            return Err(PolicyError::InvalidNetworkHost(host.clone()));
        }
        if !unique.insert(host) {
            return Err(PolicyError::DuplicateNetworkHost(host.clone()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_policy_requires_all_isolation_namespaces() {
        let mut policy = SandboxPolicy::managed("/workspace");
        assert!(policy.validate().is_ok());

        policy.namespaces.pid = false;
        assert_eq!(
            policy.validate(),
            Err(PolicyError::ManagedNamespacesRequired)
        );
    }

    #[test]
    fn relative_and_parent_paths_are_rejected() {
        let mut policy = SandboxPolicy::managed("workspace");
        assert!(matches!(
            policy.validate(),
            Err(PolicyError::RelativePath {
                field: "workspace",
                ..
            })
        ));

        policy.filesystem.workspace = PathBuf::from("/safe/../workspace");
        assert!(matches!(
            policy.validate(),
            Err(PolicyError::ParentTraversal {
                field: "workspace",
                ..
            })
        ));
    }

    #[test]
    fn invalid_limits_and_unsupported_allowlist_fail_closed() {
        let mut policy = SandboxPolicy::managed("/workspace");
        policy.resources.output_bytes = Some(0);
        assert_eq!(
            policy.validate(),
            Err(PolicyError::NonPositiveLimit {
                field: "output_bytes"
            })
        );

        policy.resources.output_bytes = None;
        policy.network = NetworkMode::Allowlist(Vec::new());
        assert_eq!(policy.validate(), Err(PolicyError::EmptyNetworkAllowlist));

        policy.network = NetworkMode::Allowlist(vec!["api.example".to_owned()]);
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn disabled_mode_is_explicit_and_serializable() {
        let policy = SandboxPolicy::disabled("/workspace");
        assert_eq!(policy.mode, SandboxMode::Disabled);
        let restored: SandboxPolicy =
            serde_json::from_value(serde_json::to_value(&policy).expect("policy should serialize"))
                .expect("policy should deserialize");
        assert_eq!(restored, policy);
    }
}
