//! Session append file writes.

use std::io::SeekFrom;

use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use super::super::super::error::SessionError;

pub(super) async fn ensure_line_separator(
    file: &mut tokio::fs::File,
    path: &std::path::Path,
) -> Result<(), SessionError> {
    if file
        .metadata()
        .await
        .map_err(|source| SessionError::Open {
            path: path.to_path_buf(),
            source,
        })?
        .len()
        == 0
    {
        return Ok(());
    }

    file.seek(SeekFrom::End(-1))
        .await
        .map_err(|source| SessionError::Write {
            path: path.to_path_buf(),
            source,
        })?;
    let mut last_byte = [0_u8; 1];
    file.read_exact(&mut last_byte)
        .await
        .map_err(|source| SessionError::Write {
            path: path.to_path_buf(),
            source,
        })?;
    if last_byte[0] != b'\n' {
        file.write_all(b"\n")
            .await
            .map_err(|source| SessionError::Write {
                path: path.to_path_buf(),
                source,
            })?;
    }
    Ok(())
}

pub(super) async fn write_record(
    file: &mut tokio::fs::File,
    path: &std::path::Path,
    line: &[u8],
) -> Result<(), SessionError> {
    file.write_all(line)
        .await
        .map_err(|source| SessionError::Write {
            path: path.to_path_buf(),
            source,
        })?;
    file.flush().await.map_err(|source| SessionError::Write {
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
