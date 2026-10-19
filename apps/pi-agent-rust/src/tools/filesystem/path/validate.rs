//! Workspace-relative path validation for existing files and directories.

use std::path::{Component, Path, PathBuf};

use crate::tools::errors::ToolError;

pub(in crate::tools::filesystem) fn display_relative_path(path: &Path) -> PathBuf {
    path.components()
        .filter(|component| !matches!(component, Component::CurDir))
        .fold(PathBuf::new(), |mut result, component| {
            result.push(component.as_os_str());
            result
        })
}

pub(in crate::tools::filesystem) async fn resolve_workspace_file(
    workspace: &Path,
    requested: &Path,
    tool: &str,
) -> Result<PathBuf, ToolError> {
    if requested.as_os_str().is_empty()
        || requested.is_absolute()
        || requested
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(ToolError::InvalidPath {
            tool: tool.to_string(),
            path: requested.to_path_buf(),
        });
    }

    let workspace =
        tokio::fs::canonicalize(workspace)
            .await
            .map_err(|source| ToolError::ResolveWorkspace {
                path: workspace.to_path_buf(),
                source,
            })?;
    let candidate = workspace.join(requested);
    let canonical = tokio::fs::canonicalize(&candidate)
        .await
        .map_err(|source| ToolError::ResolveFile {
            path: candidate.clone(),
            source,
        })?;

    if !canonical.starts_with(&workspace) {
        return Err(ToolError::PathOutsideWorkspace {
            tool: tool.to_string(),
            path: requested.to_path_buf(),
        });
    }

    Ok(canonical)
}

pub(in crate::tools::filesystem) async fn resolve_workspace_directory(
    workspace: &Path,
    requested: &Path,
    tool: &str,
) -> Result<PathBuf, ToolError> {
    if requested.is_absolute()
        || requested
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(ToolError::InvalidPath {
            tool: tool.to_string(),
            path: requested.to_path_buf(),
        });
    }

    let workspace =
        tokio::fs::canonicalize(workspace)
            .await
            .map_err(|source| ToolError::ResolveWorkspace {
                path: workspace.to_path_buf(),
                source,
            })?;
    let candidate = if requested.as_os_str().is_empty() {
        workspace.clone()
    } else {
        workspace.join(requested)
    };
    let canonical = tokio::fs::canonicalize(&candidate)
        .await
        .map_err(|source| ToolError::ResolveFile {
            path: candidate.clone(),
            source,
        })?;
    if !canonical.starts_with(&workspace) {
        return Err(ToolError::PathOutsideWorkspace {
            tool: tool.to_string(),
            path: requested.to_path_buf(),
        });
    }
    let metadata =
        tokio::fs::metadata(&canonical)
            .await
            .map_err(|source| ToolError::InspectFile {
                path: canonical.clone(),
                source,
            })?;
    if !metadata.is_dir() {
        return Err(ToolError::InvalidPath {
            tool: tool.to_string(),
            path: requested.to_path_buf(),
        });
    }
    Ok(canonical)
}
