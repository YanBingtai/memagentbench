use std::{
    collections::BTreeSet,
    fs::{File as StdFile, OpenOptions as StdOpenOptions},
    io,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{
    fs::{self, File, OpenOptions},
    io::{
        AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader,
        SeekFrom,
    },
    sync::Mutex,
};
use uuid::Uuid;

use crate::message::{Message, ToolResultMetadata};

const MAX_SESSION_RECORD_BYTES: usize = 8 * 1024 * 1024;

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

    #[error("failed to lock session file {path}: {source}")]
    Lock {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("session record at {path} line {line} exceeds the {limit}-byte limit")]
    RecordTooLarge {
        path: PathBuf,
        line: usize,
        limit: usize,
    },

    #[error("session contains incomplete tool calls outside the transcript tail: {call_ids:?}")]
    OrphanToolCall { call_ids: Vec<String> },

    #[error("tool result is missing its tool_call_id")]
    ToolResultMissingCallId,

    #[error("tool result refers to unknown tool call {call_id:?}")]
    UnexpectedToolResult { call_id: String },

    #[error("tool call {call_id:?} has more than one result")]
    DuplicateToolResult { call_id: String },

    #[error("tool call id {call_id:?} is declared more than once")]
    DuplicateToolCall { call_id: String },

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

/// Holds an OS-level lock for the lifetime of one session operation.
///
/// The lock is attached to the session file itself, so the kernel releases it
/// when the process exits unexpectedly. This protects independent
/// `SessionStore::open` instances as well as separate processes.
struct SessionFileLock {
    _file: StdFile,
}

/// Durable storage for one conversation's message transcript.
///
/// The file is append-only JSONL. An in-process mutex and an OS-level file
/// lock serialize reads, repairs, and writes across clones, independent opens,
/// and separate processes; each record remains independently recoverable after
/// a restart.
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
        let _lock = acquire_session_lock(&path, true).await?;
        let records = read_records(&path).await?;
        let next_sequence = next_sequence(&records, session_id);

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
        let _lock = acquire_session_lock(&self.path, true).await?;
        self.refresh_sequence(&mut state).await?;
        self.append_locked(&mut state, message).await
    }

    async fn refresh_sequence(&self, state: &mut SessionState) -> Result<(), SessionError> {
        state.next_sequence = next_sequence(&read_records(&self.path).await?, self.session_id);
        Ok(())
    }

    async fn append_locked(
        &self,
        state: &mut SessionState,
        message: &Message,
    ) -> Result<SessionRecord, SessionError> {
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
            .read(true)
            .open(&self.path)
            .await
            .map_err(|source| SessionError::Open {
                path: self.path.clone(),
                source,
            })?;
        if file
            .metadata()
            .await
            .map_err(|source| SessionError::Open {
                path: self.path.clone(),
                source,
            })?
            .len()
            > 0
        {
            file.seek(SeekFrom::End(-1))
                .await
                .map_err(|source| SessionError::Write {
                    path: self.path.clone(),
                    source,
                })?;
            let mut last_byte = [0_u8; 1];
            file.read_exact(&mut last_byte)
                .await
                .map_err(|source| SessionError::Write {
                    path: self.path.clone(),
                    source,
                })?;
            if last_byte[0] != b'\n' {
                file.write_all(b"\n")
                    .await
                    .map_err(|source| SessionError::Write {
                        path: self.path.clone(),
                        source,
                    })?;
            }
        }
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
        let _lock = acquire_session_lock(&self.path, true).await?;
        Ok(read_records(&self.path)
            .await?
            .into_iter()
            .filter(|record| record.session_id == self.session_id)
            .collect())
    }

    /// Load the transcript and repair an interrupted final tool batch.
    ///
    /// Appending a synthetic result is safe only when the incomplete batch is
    /// at the transcript tail. An orphan in the middle cannot be repaired by
    /// appending, so it is reported instead of silently changing message
    /// order.
    pub async fn load_repaired(&self) -> Result<Vec<SessionRecord>, SessionError> {
        let mut state = self.state.lock().await;
        let _lock = acquire_session_lock(&self.path, true).await?;
        let records = read_records(&self.path)
            .await?
            .into_iter()
            .filter(|record| record.session_id == self.session_id)
            .collect::<Vec<_>>();
        state.next_sequence = next_sequence(&records, self.session_id);
        let missing = find_tail_orphans(&records)?;
        if missing.is_empty() {
            return Ok(records);
        }

        let mut repaired = records;
        for call_id in missing {
            let record = self
                .append_locked(
                    &mut state,
                    &Message::tool_with_result(
                        call_id,
                        "[agent recovery] tool execution was interrupted; outcome is unknown",
                        ToolResultMetadata::new(
                            serde_json::json!({"status": "unknown"}),
                            true,
                            false,
                        ),
                    ),
                )
                .await?;
            repaired.push(record);
        }
        Ok(repaired)
    }
}

fn find_tail_orphans(records: &[SessionRecord]) -> Result<Vec<String>, SessionError> {
    validate_tool_result_identity(records)?;
    for (index, record) in records.iter().enumerate() {
        let Some(tool_calls) = record.message.tool_calls.as_ref() else {
            continue;
        };
        if record.message.role != "assistant" || tool_calls.is_empty() {
            continue;
        }

        let expected: BTreeSet<&str> = tool_calls.iter().map(|call| call.id.as_str()).collect();
        let mut found = BTreeSet::new();
        let mut has_non_tool_after_batch = false;
        for following in records.iter().skip(index + 1) {
            if following.message.role == "tool" {
                if let Some(call_id) = following.message.tool_call_id.as_deref() {
                    if expected.contains(call_id) {
                        found.insert(call_id);
                    }
                }
            } else {
                has_non_tool_after_batch = true;
                break;
            }
        }

        let missing: Vec<String> = tool_calls
            .iter()
            .filter(|call| !found.contains(call.id.as_str()))
            .map(|call| call.id.clone())
            .collect();
        if missing.is_empty() {
            continue;
        }
        if has_non_tool_after_batch {
            return Err(SessionError::OrphanToolCall { call_ids: missing });
        }
        return Ok(missing);
    }
    Ok(Vec::new())
}

fn validate_tool_result_identity(records: &[SessionRecord]) -> Result<(), SessionError> {
    let mut declared = BTreeSet::new();
    let mut results = BTreeSet::new();
    for record in records {
        if record.message.role == "assistant" {
            let Some(tool_calls) = record.message.tool_calls.as_ref() else {
                continue;
            };
            for call in tool_calls {
                if call.id.trim().is_empty() {
                    return Err(SessionError::DuplicateToolCall {
                        call_id: call.id.clone(),
                    });
                }
                if !declared.insert(call.id.as_str()) {
                    return Err(SessionError::DuplicateToolCall {
                        call_id: call.id.clone(),
                    });
                }
            }
        } else if record.message.role == "tool" {
            let Some(call_id) = record.message.tool_call_id.as_deref() else {
                return Err(SessionError::ToolResultMissingCallId);
            };
            if !declared.contains(call_id) {
                return Err(SessionError::UnexpectedToolResult {
                    call_id: call_id.to_string(),
                });
            }
            if !results.insert(call_id) {
                return Err(SessionError::DuplicateToolResult {
                    call_id: call_id.to_string(),
                });
            }
        }
    }
    Ok(())
}

fn next_sequence(records: &[SessionRecord], session_id: Uuid) -> u64 {
    records
        .iter()
        .filter(|record| record.session_id == session_id)
        .map(|record| record.sequence.saturating_add(1))
        .max()
        .unwrap_or(0)
}

async fn acquire_session_lock(
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

async fn read_record_line<R: AsyncBufRead + Unpin>(
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
    use crate::message::{FunctionCall, ToolCall};
    use tempfile::tempdir;

    fn tool_call(id: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: r#"{"path":"note.txt"}"#.to_string(),
            },
        }
    }

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
    async fn separate_stores_serialize_same_session_sequences() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("conversation.jsonl");
        let session_id = Uuid::new_v4();
        let first = SessionStore::open(&path, session_id)
            .await
            .expect("first session should open");
        let second = SessionStore::open(&path, session_id)
            .await
            .expect("second session should open");

        let first_message = Message::user("first");
        let second_message = Message::user("second");
        let (first_result, second_result) =
            tokio::join!(first.append(&first_message), second.append(&second_message),);
        let first_record = first_result.expect("first append should succeed");
        let second_record = second_result.expect("second append should succeed");
        let mut sequences = [first_record.sequence, second_record.sequence];
        sequences.sort_unstable();

        assert_eq!(sequences, [0, 1]);
        assert_eq!(
            SessionStore::open(&path, session_id)
                .await
                .expect("session should reopen")
                .load()
                .await
                .expect("session should load")
                .len(),
            2
        );
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
    async fn rejects_an_oversized_session_record() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("oversized.jsonl");
        fs::write(&path, vec![b'x'; MAX_SESSION_RECORD_BYTES + 1])
            .await
            .expect("oversized fixture should be written");

        let error = SessionStore::open(&path, Uuid::new_v4())
            .await
            .expect_err("oversized record should be rejected");

        assert!(matches!(
            error,
            SessionError::RecordTooLarge {
                line: 1,
                limit: MAX_SESSION_RECORD_BYTES,
                ..
            }
        ));
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

    #[tokio::test]
    async fn repairs_a_first_incomplete_record_before_append() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("first-incomplete.jsonl");
        fs::write(&path, b"{\"session_id\":")
            .await
            .expect("partial record should be written");

        let session_id = Uuid::new_v4();
        let store = SessionStore::open(&path, session_id)
            .await
            .expect("incomplete first record should be recoverable");
        assert!(store.load().await.expect("session should load").is_empty());

        let record = store
            .append(&Message::user("after repair"))
            .await
            .expect("append should succeed after repair");
        assert_eq!(record.sequence, 0);
        assert_eq!(store.load().await.expect("session should reload").len(), 1);
    }

    #[tokio::test]
    async fn appends_after_a_complete_record_without_trailing_newline() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("missing-newline.jsonl");
        let session_id = Uuid::new_v4();
        let store = SessionStore::open(&path, session_id)
            .await
            .expect("session should open");
        store
            .append(&Message::user("first"))
            .await
            .expect("first message should append");

        let file = OpenOptions::new()
            .write(true)
            .open(&path)
            .await
            .expect("session file should open");
        let length = file
            .metadata()
            .await
            .expect("session metadata should be readable")
            .len();
        file.set_len(length - 1)
            .await
            .expect("trailing newline should be removable");
        file.sync_data()
            .await
            .expect("session truncation should be durable");

        let reopened = SessionStore::open(&path, session_id)
            .await
            .expect("complete JSON without a newline should load");
        let second = reopened
            .append(&Message::assistant(Some("second".to_string()), None))
            .await
            .expect("append should restore the JSONL separator");
        let records = reopened.load().await.expect("session should reload");

        assert_eq!(second.sequence, 1);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].message.content.as_deref(), Some("first"));
        assert_eq!(records[1].message.content.as_deref(), Some("second"));
    }

    #[tokio::test]
    async fn repairs_all_tail_tool_calls_once_in_source_order() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("orphan.jsonl");
        let session_id = Uuid::new_v4();
        let store = SessionStore::open(&path, session_id)
            .await
            .expect("session should open");
        store
            .append(&Message::user("read notes"))
            .await
            .expect("user message should append");
        store
            .append(&Message::assistant(
                None,
                Some(vec![tool_call("call-b"), tool_call("call-a")]),
            ))
            .await
            .expect("assistant call should append");

        let repaired = store
            .load_repaired()
            .await
            .expect("tail calls should be repaired");
        assert_eq!(repaired.len(), 4);
        assert_eq!(repaired[2].message.tool_call_id.as_deref(), Some("call-b"));
        assert_eq!(repaired[3].message.tool_call_id.as_deref(), Some("call-a"));
        assert!(repaired[2]
            .message
            .content
            .as_deref()
            .is_some_and(|content| content.contains("outcome is unknown")));
        for record in &repaired[2..] {
            let result = record
                .message
                .tool_result
                .as_ref()
                .expect("recovered tool result should have metadata");
            assert_eq!(result.details, serde_json::json!({"status": "unknown"}));
            assert!(result.is_error);
            assert!(!result.terminate);
        }

        let second_load = store
            .load_repaired()
            .await
            .expect("repaired transcript should remain stable");
        assert_eq!(second_load.len(), 4);
    }

    #[tokio::test]
    async fn repairs_only_missing_tail_call_after_partial_execution() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("partial-orphan.jsonl");
        let session_id = Uuid::new_v4();
        let store = SessionStore::open(&path, session_id)
            .await
            .expect("session should open");
        store
            .append(&Message::assistant(
                None,
                Some(vec![tool_call("call-1"), tool_call("call-2")]),
            ))
            .await
            .expect("assistant call should append");
        store
            .append(&Message::tool("call-1", "done"))
            .await
            .expect("completed tool should append");

        let repaired = store
            .load_repaired()
            .await
            .expect("missing tail call should be repaired");
        assert_eq!(repaired.len(), 3);
        assert_eq!(repaired[2].message.tool_call_id.as_deref(), Some("call-2"));
        let result = repaired[2]
            .message
            .tool_result
            .as_ref()
            .expect("recovered tool result should have metadata");
        assert_eq!(result.details, serde_json::json!({"status": "unknown"}));
        assert!(result.is_error);
        assert!(!result.terminate);
    }

    #[tokio::test]
    async fn rejects_an_orphan_that_is_not_at_the_transcript_tail() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("middle-orphan.jsonl");
        let session_id = Uuid::new_v4();
        let store = SessionStore::open(&path, session_id)
            .await
            .expect("session should open");
        store
            .append(&Message::assistant(None, Some(vec![tool_call("call-1")])))
            .await
            .expect("assistant call should append");
        store
            .append(&Message::user("continue"))
            .await
            .expect("following user message should append");

        let error = store
            .load_repaired()
            .await
            .expect_err("middle orphan must not be appended out of order");
        assert!(matches!(
            error,
            SessionError::OrphanToolCall { call_ids } if call_ids == vec!["call-1"]
        ));
    }

    #[tokio::test]
    async fn rejects_duplicate_tool_results() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("duplicate-tool-result.jsonl");
        let session_id = Uuid::new_v4();
        let store = SessionStore::open(&path, session_id)
            .await
            .expect("session should open");
        store
            .append(&Message::assistant(None, Some(vec![tool_call("call-1")])))
            .await
            .expect("assistant call should append");
        store
            .append(&Message::tool("call-1", "first"))
            .await
            .expect("first result should append");
        store
            .append(&Message::tool("call-1", "duplicate"))
            .await
            .expect("duplicate fixture should append");

        let error = store
            .load_repaired()
            .await
            .expect_err("duplicate tool results must be rejected");
        assert!(matches!(
            error,
            SessionError::DuplicateToolResult { call_id } if call_id == "call-1"
        ));
    }

    #[tokio::test]
    async fn rejects_tool_results_without_a_known_call() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("unknown-tool-result.jsonl");
        let session_id = Uuid::new_v4();
        let store = SessionStore::open(&path, session_id)
            .await
            .expect("session should open");
        store
            .append(&Message::tool("unknown-call", "unexpected"))
            .await
            .expect("unexpected result fixture should append");

        let error = store
            .load_repaired()
            .await
            .expect_err("unknown tool results must be rejected");
        assert!(matches!(
            error,
            SessionError::UnexpectedToolResult { call_id } if call_id == "unknown-call"
        ));
    }

    #[tokio::test]
    async fn rejects_a_tool_result_before_its_assistant_call() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("out-of-order-tool-result.jsonl");
        let session_id = Uuid::new_v4();
        let store = SessionStore::open(&path, session_id)
            .await
            .expect("session should open");
        store
            .append(&Message::tool("call-1", "result before call"))
            .await
            .expect("out-of-order result fixture should append");
        store
            .append(&Message::assistant(None, Some(vec![tool_call("call-1")])))
            .await
            .expect("assistant call fixture should append");

        let error = store
            .load_repaired()
            .await
            .expect_err("out-of-order tool result must be rejected immediately");
        assert!(matches!(
            error,
            SessionError::UnexpectedToolResult { call_id } if call_id == "call-1"
        ));
    }

    #[tokio::test]
    async fn rejects_tool_results_without_call_ids() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("missing-tool-result-id.jsonl");
        let session_id = Uuid::new_v4();
        let store = SessionStore::open(&path, session_id)
            .await
            .expect("session should open");
        store
            .append(&Message {
                role: "tool".to_string(),
                content: Some("missing id".to_string()),
                reasoning_content: None,
                stop_reason: None,
                name: None,
                tool_call_id: None,
                tool_result: None,
                tool_calls: None,
            })
            .await
            .expect("malformed result fixture should append");

        let error = store
            .load_repaired()
            .await
            .expect_err("tool results without ids must be rejected");
        assert!(matches!(error, SessionError::ToolResultMissingCallId));
    }
}
