//! Deterministic directory reads for `search_files`.

use std::path::Path;

use crate::tools::errors::ToolError;

pub(in crate::tools::filesystem::search::walk) async fn read_children(
    directory: &Path,
) -> Result<Vec<tokio::fs::DirEntry>, ToolError> {
    let mut reader =
        tokio::fs::read_dir(directory)
            .await
            .map_err(|source| ToolError::ReadDirectory {
                path: directory.to_path_buf(),
                source,
            })?;
    let mut children = Vec::new();
    while let Some(entry) =
        reader
            .next_entry()
            .await
            .map_err(|source| ToolError::ReadDirectory {
                path: directory.to_path_buf(),
                source,
            })?
    {
        children.push(entry);
    }
    children.sort_by_key(|entry| entry.file_name());
    Ok(children)
}
