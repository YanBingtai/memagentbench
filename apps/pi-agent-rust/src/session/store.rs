//! SessionStore operations over the JSONL and repair backends.

use std::{
    io::SeekFrom,
    path::{Path, PathBuf},
    sync::Arc,
};

use tokio::{
    fs,
    fs::OpenOptions,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    sync::Mutex,
};
use uuid::Uuid;

use crate::message::{Message, ToolResultMetadata};

use super::{
    error::SessionError,
    jsonl::{read_records, unix_timestamp_ms},
    lock::acquire_session_lock,
    repair::{find_tail_orphans, next_sequence},
    types::{SessionRecord, SessionState, SessionStore},
};

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
