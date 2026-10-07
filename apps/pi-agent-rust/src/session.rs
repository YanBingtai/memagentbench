use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{
    fs::{self, File, OpenOptions},
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::Mutex,
};
use uuid::Uuid;

use crate::message::Message;

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("failed to create session directory {path}: {source}")]
    CreateDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to open session file {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to read session file {path} at line {line}: {source}")]
    Read {
        path: PathBuf,
        line: usize,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid JSON in session file {path} at line {line}: {source}")]
    Decode {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to encode session record: {0}")]
    Encode(#[source] serde_json::Error),

    #[error("failed to append session file {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("system clock is before the Unix epoch: {0}")]
    Clock(#[source] std::time::SystemTimeError),
}

/// One append-only JSONL entry in a session transcript.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionRecord {
    pub session_id: Uuid,
    pub sequence: u64,
    pub timestamp_ms: u64,
    pub message: Message,
}

#[derive(Debug, Default)]
struct SessionState {
    next_sequence: u64,
}

/// Durable storage for one conversation's message transcript.
///
/// The file is append-only JSONL. A mutex serializes writes made by this
/// process; each record remains independently recoverable after a restart.
#[derive(Clone, Debug)]
pub struct SessionStore {
    path: PathBuf,
    session_id: Uuid,
    state: Arc<Mutex<SessionState>>,
}

impl SessionStore {
    /// Open an existing session file or prepare a new one.
    pub async fn open(path: impl Into<PathBuf>, session_id: Uuid) -> Result<Self, SessionError> {
        let path = path.into();
        let records = read_records(&path).await?;
        let next_sequence = records
            .iter()
            .filter(|record| record.session_id == session_id)
            .map(|record| record.sequence.saturating_add(1))
            .max()
            .unwrap_or(0);

        Ok(Self {
            path,
            session_id,
            state: Arc::new(Mutex::new(SessionState { next_sequence })),
        })
    }

    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one message and return the durable record that was written.
    pub async fn append(&self, message: &Message) -> Result<SessionRecord, SessionError> {
        let mut state = self.state.lock().await;
        let record = SessionRecord {
            session_id: self.session_id,
            sequence: state.next_sequence,
            timestamp_ms: unix_timestamp_ms()?,
            message: message.clone(),
        };
        let mut line = serde_json::to_vec(&record).map_err(SessionError::Encode)?;
        line.push(b'\n');

        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|source| SessionError::CreateDirectory {
                    path: parent.to_path_buf(),
                    source,
                })?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await
            .map_err(|source| SessionError::Open {
                path: self.path.clone(),
                source,
            })?;
        file.write_all(&line)
            .await
            .map_err(|source| SessionError::Write {
                path: self.path.clone(),
                source,
            })?;
        file.flush().await.map_err(|source| SessionError::Write {
            path: self.path.clone(),
            source,
        })?;
        file.sync_data()
            .await
            .map_err(|source| SessionError::Write {
                path: self.path.clone(),
                source,
            })?;
        state.next_sequence = state.next_sequence.saturating_add(1);

        Ok(record)
    }

    /// Load this session's records, ignoring records belonging to other IDs.
    pub async fn load(&self) -> Result<Vec<SessionRecord>, SessionError> {
        // Keep repair and sequence allocation mutually exclusive for clones of
        // this store; otherwise a reader could truncate an active append.
        let _state = self.state.lock().await;
        Ok(read_records(&self.path)
            .await?
            .into_iter()
            .filter(|record| record.session_id == self.session_id)
            .collect())
    }
}

async fn read_records(path: &Path) -> Result<Vec<SessionRecord>, SessionError> {
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
        line.clear();
        let bytes_read = reader
            .read_until(b'\n', &mut line)
            .await
            .map_err(|source| SessionError::Read {
                path: path.to_path_buf(),
                line: line_number.saturating_add(1),
                source,
            })?;
        if bytes_read == 0 {
            break;
        }

        line_number += 1;
        let has_newline = line.last() == Some(&b'\n');
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
            Err(_source) if !has_newline && !records.is_empty() => {
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

/// Remove a final record that was interrupted while being appended.
///
/// Complete malformed lines still return `SessionError::Decode`; only a
/// malformed final line without a trailing newline is treated as a crash tail.
async fn truncate_incomplete_tail(path: &Path, length: usize) -> Result<(), SessionError> {
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

fn unix_timestamp_ms() -> Result<u64, SessionError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(SessionError::Clock)?;
    Ok(duration.as_millis().min(u64::MAX as u128) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn appends_and_reloads_messages_in_sequence() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("sessions/conversation.jsonl");
        let session_id = Uuid::new_v4();
        let store = SessionStore::open(&path, session_id)
            .await
            .expect("session should open");

        let first = store
            .append(&Message::user("hello"))
            .await
            .expect("first message should append");
        let second = store
            .append(&Message::assistant(Some("hi".to_string()), None))
            .await
            .expect("second message should append");
        let records = store.load().await.expect("session should load");

        assert_eq!(first.sequence, 0);
        assert_eq!(second.sequence, 1);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].message, Message::user("hello"));
        assert_eq!(records[1].message.role, "assistant");
    }

    #[tokio::test]
    async fn separate_sessions_share_a_file_without_mixing_records() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("conversation.jsonl");
        let first_id = Uuid::new_v4();
        let second_id = Uuid::new_v4();
        let first = SessionStore::open(&path, first_id)
            .await
            .expect("first session should open");
        let second = SessionStore::open(&path, second_id)
            .await
            .expect("second session should open");

        first
            .append(&Message::user("first"))
            .await
            .expect("first message should append");
        second
            .append(&Message::user("second"))
            .await
            .expect("second message should append");

        assert_eq!(first.load().await.expect("first should load").len(), 1);
        assert_eq!(second.load().await.expect("second should load").len(), 1);
    }

    #[tokio::test]
    async fn malformed_json_reports_the_line_number() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("broken.jsonl");
        fs::write(&path, "\nnot-json\n")
            .await
            .expect("fixture should be written");

        let error = SessionStore::open(&path, Uuid::new_v4())
            .await
            .expect_err("malformed JSON should fail");

        assert!(matches!(error, SessionError::Decode { line: 2, .. }));
    }

    #[tokio::test]
    async fn recovers_from_an_incomplete_final_record() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("recoverable.jsonl");
        let session_id = Uuid::new_v4();
        let store = SessionStore::open(&path, session_id)
            .await
            .expect("session should open");
        store
            .append(&Message::user("complete"))
            .await
            .expect("complete message should append");

        let mut file = OpenOptions::new()
            .append(true)
            .open(&path)
            .await
            .expect("session file should open");
        file.write_all(b"{\"session_id\":")
            .await
            .expect("partial record should append");
        file.flush().await.expect("partial record should flush");

        let reopened = SessionStore::open(&path, session_id)
            .await
            .expect("incomplete tail should be recoverable");
        assert_eq!(reopened.load().await.expect("session should load").len(), 1);

        reopened
            .append(&Message::assistant(Some("recovered".to_string()), None))
            .await
            .expect("new message should append after recovery");
        assert_eq!(
            reopened.load().await.expect("session should reload").len(),
            2
        );
    }
}
