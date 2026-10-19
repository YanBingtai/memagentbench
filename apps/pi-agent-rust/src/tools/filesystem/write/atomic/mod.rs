//! Workspace-bounded atomic file replacement.

#[cfg(not(unix))]
mod portable;
#[cfg(unix)]
mod unix;

use std::path::Path;

#[cfg(unix)]
use std::{ffi::OsString, io, path::PathBuf};

use super::super::path::resolve_workspace_write_path;
use crate::tools::errors::ToolError;

pub(in crate::tools::filesystem) async fn atomic_write_workspace_file(
    workspace: &Path,
    requested: &Path,
    content: &[u8],
) -> Result<(), ToolError> {
    let (target, temporary) = resolve_workspace_write_path(workspace, requested).await?;

    #[cfg(unix)]
    {
        let canonical_workspace = tokio::fs::canonicalize(workspace).await.map_err(|source| {
            ToolError::ResolveWorkspace {
                path: workspace.to_path_buf(),
                source,
            }
        })?;
        let parts = WriteParts::from_paths(&target, &temporary, requested)?;
        let content = content.to_vec();
        let parent = parts.parent.clone();
        let temporary_name = parts.temporary_name.clone();
        let target_name = parts.target_name.clone();
        let write_target = target.clone();
        tokio::task::spawn_blocking(move || {
            unix::write(
                &canonical_workspace,
                &parent,
                &temporary_name,
                &target_name,
                &content,
            )
        })
        .await
        .map_err(|error| ToolError::WriteFile {
            path: write_target.clone(),
            source: io::Error::other(format!("write worker failed: {error}")),
        })?
        .map_err(|source| ToolError::WriteFile {
            path: write_target,
            source,
        })?;
        Ok(())
    }

    #[cfg(not(unix))]
    {
        portable::write(&temporary, &target, content).await
    }
}

#[cfg(unix)]
struct WriteParts {
    parent: PathBuf,
    temporary_name: OsString,
    target_name: OsString,
}

#[cfg(unix)]
impl WriteParts {
    fn from_paths(target: &Path, temporary: &Path, requested: &Path) -> Result<Self, ToolError> {
        let parent = temporary
            .parent()
            .ok_or_else(|| invalid_path(requested))?
            .to_path_buf();
        let temporary_name = temporary
            .file_name()
            .ok_or_else(|| invalid_path(requested))?
            .to_os_string();
        let target_name = target
            .file_name()
            .ok_or_else(|| invalid_path(requested))?
            .to_os_string();
        Ok(Self {
            parent,
            temporary_name,
            target_name,
        })
    }
}

#[cfg(unix)]
fn invalid_path(requested: &Path) -> ToolError {
    ToolError::InvalidPath {
        tool: "write_file".to_string(),
        path: requested.to_path_buf(),
    }
}
