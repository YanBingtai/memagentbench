//! Session loading and interrupted-tool-batch repair.

use crate::message::{Message, ToolResultMetadata};

use super::super::{
    error::SessionError,
    jsonl::read_records,
    lock::acquire_session_lock,
    repair::{find_tail_orphans, next_sequence},
    types::{SessionRecord, SessionStore},
};

impl SessionStore {
    /// Load this session's records, ignoring records belonging to other IDs.
    pub async fn load(&self) -> Result<Vec<SessionRecord>, SessionError> {
        // Keep repair and sequence allocation mutually exclusive for clones of
        // this store; otherwise a reader could truncate an active append.
        let _state = self.state.lock().await;
        let _lock = acquire_session_lock(&self.path, true).await?;
        Ok(session_records(
            read_records(&self.path).await?,
            self.session_id,
        ))
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
        let records = session_records(read_records(&self.path).await?, self.session_id);
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

fn session_records(records: Vec<SessionRecord>, session_id: uuid::Uuid) -> Vec<SessionRecord> {
    records
        .into_iter()
        .filter(|record| record.session_id == session_id)
        .collect()
}
