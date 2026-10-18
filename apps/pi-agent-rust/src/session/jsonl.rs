//! Bounded JSONL framing and crash-tail repair.

use std::{
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use tokio::{
    fs::{File, OpenOptions},
    io::{AsyncBufRead, AsyncBufReadExt, BufReader},
};

use super::{error::SessionError, limits::MAX_SESSION_RECORD_BYTES, types::SessionRecord};

pub(crate) async fn read_records(path: &Path) -> Result<Vec<SessionRecord>, SessionError> {
    let file = match File::open(path).await {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(SessionError::Open {
                path: path.to_path_buf(),
                source,
            });
        }
    };

    let mut reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut line_start = 0usize;
    let mut line_number = 0usize;
    let mut line = Vec::new();
    loop {
        let Some((bytes_read, has_newline)) =
            read_record_line(&mut reader, &mut line, path, line_number + 1).await?
        else {
            break;
        };

        line_number += 1;
        if has_newline {
            line.pop();
        }
        let current_line_start = line_start;
        line_start += bytes_read;

        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }

        match serde_json::from_slice::<SessionRecord>(&line) {
            Ok(record) => records.push(record),
            Err(_source) if !has_newline => {
                truncate_incomplete_tail(path, current_line_start).await?;
                break;
            }
            Err(source) => {
                return Err(SessionError::Decode {
                    path: path.to_path_buf(),
                    line: line_number,
                    source,
                });
            }
        }
    }

    Ok(records)
}

pub(crate) async fn read_record_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    line: &mut Vec<u8>,
    path: &Path,
    line_number: usize,
) -> Result<Option<(usize, bool)>, SessionError> {
    line.clear();
    let mut bytes_read = 0usize;
    loop {
        let buffer = reader
            .fill_buf()
            .await
            .map_err(|source| SessionError::Read {
                path: path.to_path_buf(),
                line: line_number,
                source,
            })?;
        if buffer.is_empty() {
            return if bytes_read == 0 {
                Ok(None)
            } else {
                Ok(Some((bytes_read, false)))
            };
        }

        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let chunk_len = newline.map_or(buffer.len(), |index| index + 1);
        if line.len().saturating_add(chunk_len) > MAX_SESSION_RECORD_BYTES {
            return Err(SessionError::RecordTooLarge {
                path: path.to_path_buf(),
                line: line_number,
                limit: MAX_SESSION_RECORD_BYTES,
            });
        }
        line.extend_from_slice(&buffer[..chunk_len]);
        reader.consume(chunk_len);
        bytes_read = bytes_read.saturating_add(chunk_len);
        if newline.is_some() {
            return Ok(Some((bytes_read, true)));
        }
    }
}

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
