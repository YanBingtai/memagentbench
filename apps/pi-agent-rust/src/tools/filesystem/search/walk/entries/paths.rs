//! Relative paths and workspace-boundary checks for directory entries.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use crate::tools::errors::ToolError;

pub(super) fn relative_path(relative_directory: &Path, name: &OsStr) -> PathBuf {
    if relative_directory.as_os_str().is_empty() {
        PathBuf::from(name)
    } else {
        relative_directory.join(name)
    }
}

pub(super) async fn validate_directory(
    workspace: &Path,
    entry: &tokio::fs::DirEntry,
) -> Result<PathBuf, ToolError> {
    let path = entry.path();
    let canonical =
        tokio::fs::canonicalize(&path)
            .await
            .map_err(|source| ToolError::InspectFile {
                path: path.clone(),
                source,
            })?;
    if !canonical.starts_with(workspace) {
        return Err(ToolError::PathOutsideWorkspace {
            tool: "search_files".to_string(),
            path,
        });
    }
    Ok(canonical)
}

pub(super) fn should_skip_search_directory(name: &OsStr) -> bool {
    matches!(
        name.to_str(),
        Some(".git") | Some("target") | Some("node_modules")
    )
}
