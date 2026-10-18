//! OS-level lock for one session file operation.

use std::{fs::OpenOptions as StdOpenOptions, io, path::Path};

use tokio::fs;

use super::{error::SessionError, types::SessionFileLock};

pub(crate) async fn acquire_session_lock(
    path: &Path,
    exclusive: bool,
) -> Result<SessionFileLock, SessionError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .await
            .map_err(|source| SessionError::CreateDirectory {
                path: parent.to_path_buf(),
                source,
            })?;
    }

    let lock_path = path.to_path_buf();
    let task_path = lock_path.clone();
    tokio::task::spawn_blocking(move || {
        let file = StdOpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&task_path)
            .map_err(|source| SessionError::Lock {
                path: task_path.clone(),
                source,
            })?;
        let result = if exclusive {
            file.lock()
        } else {
            file.lock_shared()
        };
        result.map_err(|source| SessionError::Lock {
            path: task_path,
            source,
        })?;
        Ok(SessionFileLock { _file: file })
    })
    .await
    .map_err(|source| SessionError::Lock {
        path: lock_path,
        source: io::Error::other(source.to_string()),
    })?
}
