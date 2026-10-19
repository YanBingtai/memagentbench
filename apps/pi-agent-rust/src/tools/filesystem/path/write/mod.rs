//! Workspace-safe parent resolution for atomic writes.

mod parent;
mod target;

use std::path::{Component, Path, PathBuf};

use crate::tools::errors::ToolError;

/// Resolve an atomic-write target and a unique temporary path in its parent.
///
/// The parent is created one component at a time so a symlink cannot redirect
/// a write outside the canonical workspace between validation and creation.
pub(in crate::tools::filesystem) async fn resolve_workspace_write_path(
    workspace: &Path,
    requested: &Path,
) -> Result<(PathBuf, PathBuf), ToolError> {
    validate_requested_path(requested)?;

    let workspace =
        tokio::fs::canonicalize(workspace)
            .await
            .map_err(|source| ToolError::ResolveWorkspace {
                path: workspace.to_path_buf(),
                source,
            })?;
    let candidate = workspace.join(requested);
    let parent = candidate.parent().ok_or_else(|| invalid_path(requested))?;
    let parent = parent::ensure_workspace_write_parent(&workspace, parent, requested).await?;
    let file_name = candidate
        .file_name()
        .ok_or_else(|| invalid_path(requested))?;
    let target = target::validate_target(&workspace, &parent, file_name, requested).await?;
    let temporary = target::temporary_path(&parent, file_name);

    Ok((target, temporary))
}

fn validate_requested_path(requested: &Path) -> Result<(), ToolError> {
    if requested.as_os_str().is_empty()
        || requested.is_absolute()
        || requested
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(invalid_path(requested));
    }
    Ok(())
}

fn invalid_path(path: &Path) -> ToolError {
    ToolError::InvalidPath {
        tool: "write_file".to_string(),
        path: path.to_path_buf(),
    }
}

fn path_outside_workspace(path: &Path) -> ToolError {
    ToolError::PathOutsideWorkspace {
        tool: "write_file".to_string(),
        path: path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests;
