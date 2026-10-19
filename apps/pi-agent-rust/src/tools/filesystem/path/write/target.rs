//! Existing target validation and temporary-path generation for atomic writes.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use uuid::Uuid;

use crate::tools::errors::ToolError;

use super::{invalid_path, path_outside_workspace};

pub(super) async fn validate_target(
    workspace: &Path,
    parent: &Path,
    file_name: &OsStr,
    requested: &Path,
) -> Result<PathBuf, ToolError> {
    let target = parent.join(file_name);
    match tokio::fs::symlink_metadata(&target).await {
        Ok(metadata) => {
            let canonical = tokio::fs::canonicalize(&target).await.map_err(|source| {
                ToolError::InspectFile {
                    path: target.clone(),
                    source,
                }
            })?;
            if !canonical.starts_with(workspace) {
                return Err(path_outside_workspace(requested));
            }
            if !metadata.is_file() {
                return Err(invalid_path(&target));
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

    Ok(target)
}

pub(super) fn temporary_path(parent: &Path, file_name: &OsStr) -> PathBuf {
    parent.join(format!(
        ".{}.{}.tmp",
        file_name.to_string_lossy(),
        Uuid::new_v4()
    ))
}
