//! Safe creation and validation of a write target's parent directory.

use std::path::{Component, Path, PathBuf};

use crate::tools::errors::ToolError;

use super::{invalid_path, path_outside_workspace};

/// Resolve a write parent while creating missing directories.
///
/// `create_dir_all` is intentionally avoided: it may follow an unvalidated
/// symlink while creating a descendant. Each component is inspected before it
/// is used, then the resulting directory is canonicalized and checked again.
pub(super) async fn ensure_workspace_write_parent(
    workspace: &Path,
    parent: &Path,
    requested: &Path,
) -> Result<PathBuf, ToolError> {
    let relative_parent = parent
        .strip_prefix(workspace)
        .map_err(|_| path_outside_workspace(requested))?;
    let mut current = workspace.to_path_buf();

    for component in relative_parent.components() {
        let name = match component {
            Component::Normal(name) => name,
            Component::CurDir => continue,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(invalid_path(requested));
            }
        };
        current.push(name);

        match tokio::fs::symlink_metadata(&current).await {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(invalid_path(&current));
            }
            Ok(metadata) if metadata.is_dir() => {
                validate_workspace_directory(workspace, &current, requested).await?;
            }
            Ok(_) => return Err(invalid_path(&current)),
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
        return Err(path_outside_workspace(requested));
    }
    if !tokio::fs::metadata(path)
        .await
        .map_err(|source| ToolError::InspectFile {
            path: path.to_path_buf(),
            source,
        })?
        .is_dir()
    {
        return Err(invalid_path(path));
    }
    Ok(())
}
