use std::{fs, path::Path, path::PathBuf};

use pi_agent_runtime::DeltaBoxConfig;
use serde_json::json;
use tempfile::TempDir;

use crate::ToolError;

use super::super::shell_error;

/// Write the exact policy used by the sandbox and attach it to the client
/// configuration. The control directory owns the policy file's lifetime.
pub(super) fn apply(
    mut config: DeltaBoxConfig,
    worker: &Path,
    control_dir: &TempDir,
) -> Result<DeltaBoxConfig, ToolError> {
    let policy_path: PathBuf = control_dir.path().join("policy.toml");
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
    Ok(config)
}
