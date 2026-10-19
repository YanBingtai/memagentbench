//! File and child-directory collection for the directory traversal.

use std::path::{Path, PathBuf};

use crate::tools::errors::ToolError;

use super::limits::inspect_file;
use super::paths::{relative_path, should_skip_search_directory, validate_directory};
use super::{SearchOptions, SearchState};

pub(in crate::tools::filesystem::search::walk) async fn collect_entries(
    options: &SearchOptions<'_>,
    relative_directory: &Path,
    children: Vec<tokio::fs::DirEntry>,
    state: &mut SearchState<'_>,
) -> Result<Vec<(PathBuf, PathBuf)>, ToolError> {
    let mut child_directories = Vec::new();
    for entry in children {
        let name = entry.file_name();
        let relative = relative_path(relative_directory, &name);
        let file_type = entry
            .file_type()
            .await
            .map_err(|source| ToolError::InspectFile {
                path: entry.path(),
                source,
            })?;

        if file_type.is_dir() {
            if should_skip_search_directory(&name) {
                continue;
            }
            child_directories.push((
                validate_directory(options.workspace, &entry).await?,
                relative,
            ));
            continue;
        }
        if !file_type.is_file() {
            continue;
        }

        if inspect_file(&entry.path(), &relative, options, state).await {
            return Ok(child_directories);
        }
    }

    Ok(child_directories)
}
