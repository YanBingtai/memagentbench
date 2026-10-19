//! Session append file preparation.

use tokio::{fs, fs::OpenOptions};

use super::super::super::error::SessionError;

pub(super) async fn create_parent_directory(path: &std::path::Path) -> Result<(), SessionError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    fs::create_dir_all(parent)
        .await
        .map_err(|source| SessionError::CreateDirectory {
            path: parent.to_path_buf(),
            source,
        })
}

pub(super) async fn open_append_file(
    path: &std::path::Path,
) -> Result<tokio::fs::File, SessionError> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .read(true)
        .open(path)
        .await
        .map_err(|source| SessionError::Open {
            path: path.to_path_buf(),
            source,
        })
}
