use std::{
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use tokio::fs::OpenOptions;

use crate::session::error::SessionError;

/// Remove a final record that was interrupted while being appended.
///
/// Complete malformed lines still return `SessionError::Decode`; only a
/// malformed final line without a trailing newline is treated as a crash tail.
pub(crate) async fn truncate_incomplete_tail(
    path: &Path,
    length: usize,
) -> Result<(), SessionError> {
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .await
        .map_err(|source| SessionError::Open {
            path: path.to_path_buf(),
            source,
        })?;
    file.set_len(length as u64)
        .await
        .map_err(|source| SessionError::Write {
            path: path.to_path_buf(),
            source,
        })?;
    file.sync_data()
        .await
        .map_err(|source| SessionError::Write {
            path: path.to_path_buf(),
            source,
        })
}

pub(crate) fn unix_timestamp_ms() -> Result<u64, SessionError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(SessionError::Clock)?;
    Ok(duration.as_millis().min(u64::MAX as u128) as u64)
}
