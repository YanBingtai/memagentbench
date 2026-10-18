//! Shared workspace path validation and directory traversal helpers.

use std::path::{Component, Path, PathBuf};

use uuid::Uuid;

use super::super::errors::ToolError;

pub(super) fn display_relative_path(path: &Path) -> PathBuf {
    path.components()
        .filter(|component| !matches!(component, Component::CurDir))
        .fold(PathBuf::new(), |mut result, component| {
            result.push(component.as_os_str());
            result
        })
}

pub(super) async fn resolve_workspace_file(
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

pub(super) async fn resolve_workspace_directory(
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

pub(super) async fn collect_directory_entries(
    workspace: &Path,
    root: &Path,
    prefix: &Path,
    max_depth: usize,
    max_entries: usize,
) -> Result<(Vec<String>, bool), ToolError> {
    let mut pending = vec![(root.to_path_buf(), PathBuf::new(), 0usize)];
    let mut entries = Vec::with_capacity(max_entries.min(64));

    while let Some((directory, relative_directory, depth)) = pending.pop() {
        let mut reader =
            tokio::fs::read_dir(&directory)
                .await
                .map_err(|source| ToolError::ReadDirectory {
                    path: directory.clone(),
                    source,
                })?;
        let mut children = Vec::new();
        while let Some(entry) =
            reader
                .next_entry()
                .await
                .map_err(|source| ToolError::ReadDirectory {
                    path: directory.clone(),
                    source,
                })?
        {
            children.push(entry);
        }
        children.sort_by_key(|entry| entry.file_name());

        let mut child_directories = Vec::new();
        for entry in children {
            if entries.len() == max_entries {
                return Ok((entries, true));
            }

            let name = entry.file_name();
            let relative = if relative_directory.as_os_str().is_empty() {
                PathBuf::from(&name)
            } else {
                relative_directory.join(&name)
            };
            let file_type = entry
                .file_type()
                .await
                .map_err(|source| ToolError::InspectFile {
                    path: entry.path(),
                    source,
                })?;
            let is_directory = file_type.is_dir();
            let mut display = if prefix.as_os_str().is_empty() {
                relative.clone()
            } else {
                prefix.join(&relative)
            }
            .to_string_lossy()
            .into_owned();
            if is_directory {
                display.push('/');
            }
            entries.push(display);

            if is_directory && depth < max_depth {
                let canonical = tokio::fs::canonicalize(entry.path())
                    .await
                    .map_err(|source| ToolError::InspectFile {
                        path: entry.path(),
                        source,
                    })?;
                if !canonical.starts_with(workspace) {
                    return Err(ToolError::PathOutsideWorkspace {
                        tool: "list_files".to_string(),
                        path: entry.path(),
                    });
                }
                child_directories.push((canonical, relative, depth + 1));
            }
        }

        for child in child_directories.into_iter().rev() {
            pending.push(child);
        }
    }

    Ok((entries, false))
}

pub(super) async fn resolve_workspace_write_path(
    workspace: &Path,
    requested: &Path,
) -> Result<(PathBuf, PathBuf), ToolError> {
    if requested.as_os_str().is_empty()
        || requested.is_absolute()
        || requested
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(ToolError::InvalidPath {
            tool: "write_file".to_string(),
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
    let parent = candidate.parent().ok_or_else(|| ToolError::InvalidPath {
        tool: "write_file".to_string(),
        path: requested.to_path_buf(),
    })?;
    let parent = ensure_workspace_write_parent(&workspace, parent, requested).await?;

    let file_name = candidate
        .file_name()
        .ok_or_else(|| ToolError::InvalidPath {
            tool: "write_file".to_string(),
            path: requested.to_path_buf(),
        })?;
    let target = parent.join(file_name);
    match tokio::fs::symlink_metadata(&target).await {
        Ok(metadata) => {
            let canonical = tokio::fs::canonicalize(&target).await.map_err(|source| {
                ToolError::InspectFile {
                    path: target.clone(),
                    source,
                }
            })?;
            if !canonical.starts_with(&workspace) {
                return Err(ToolError::PathOutsideWorkspace {
                    tool: "write_file".to_string(),
                    path: requested.to_path_buf(),
                });
            }
            if !metadata.is_file() {
                return Err(ToolError::InvalidPath {
                    tool: "write_file".to_string(),
                    path: target,
                });
            }
        }
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(ToolError::InspectFile {
                path: target,
                source,
            });
        }
    }

    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        file_name.to_string_lossy(),
        Uuid::new_v4()
    ));
    Ok((target, temporary))
}

/// Resolve a write target's parent while creating missing directories.
///
/// Each component is opened relative to the validated workspace directory
/// without following symlinks. `create_dir_all` is intentionally avoided
/// because it would follow an unvalidated symlink while creating descendants.
async fn ensure_workspace_write_parent(
    workspace: &Path,
    parent: &Path,
    requested: &Path,
) -> Result<PathBuf, ToolError> {
    let relative_parent =
        parent
            .strip_prefix(workspace)
            .map_err(|_| ToolError::PathOutsideWorkspace {
                tool: "write_file".to_string(),
                path: requested.to_path_buf(),
            })?;
    let mut current = workspace.to_path_buf();

    for component in relative_parent.components() {
        let name = match component {
            Component::Normal(name) => name,
            Component::CurDir => continue,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(ToolError::InvalidPath {
                    tool: "write_file".to_string(),
                    path: requested.to_path_buf(),
                });
            }
        };
        current.push(name);

        match tokio::fs::symlink_metadata(&current).await {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(ToolError::InvalidPath {
                    tool: "write_file".to_string(),
                    path: current,
                });
            }
            Ok(metadata) if metadata.is_dir() => {
                validate_workspace_directory(workspace, &current, requested).await?;
            }
            Ok(_) => {
                return Err(ToolError::InvalidPath {
                    tool: "write_file".to_string(),
                    path: current,
                });
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                tokio::fs::create_dir(&current)
                    .await
                    .map_err(|source| ToolError::ResolveFile {
                        path: current.clone(),
                        source,
                    })?;
                validate_workspace_directory(workspace, &current, requested).await?;
            }
            Err(source) => {
                return Err(ToolError::ResolveFile {
                    path: current,
                    source,
                });
            }
        }
    }

    tokio::fs::canonicalize(parent)
        .await
        .map_err(|source| ToolError::ResolveFile {
            path: parent.to_path_buf(),
            source,
        })
}

async fn validate_workspace_directory(
    workspace: &Path,
    path: &Path,
    requested: &Path,
) -> Result<(), ToolError> {
    let canonical =
        tokio::fs::canonicalize(path)
            .await
            .map_err(|source| ToolError::ResolveFile {
                path: path.to_path_buf(),
                source,
            })?;
    if !canonical.starts_with(workspace) {
        return Err(ToolError::PathOutsideWorkspace {
            tool: "write_file".to_string(),
            path: requested.to_path_buf(),
        });
    }
    if !tokio::fs::metadata(path)
        .await
        .map_err(|source| ToolError::InspectFile {
            path: path.to_path_buf(),
            source,
        })?
        .is_dir()
    {
        return Err(ToolError::InvalidPath {
            tool: "write_file".to_string(),
            path: path.to_path_buf(),
        });
    }
    Ok(())
}
