//! Deterministic, bounded directory traversal for workspace listing.

use std::path::{Path, PathBuf};

use crate::tools::errors::ToolError;

pub(in crate::tools::filesystem) async fn collect_directory_entries(
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
