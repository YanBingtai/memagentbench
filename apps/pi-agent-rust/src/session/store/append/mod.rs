//! Append operations for durable session records.

mod open;
mod write;

use crate::message::Message;

use super::super::{
    error::SessionError,
    jsonl::{read_records, unix_timestamp_ms},
    lock::acquire_session_lock,
    repair::next_sequence,
    types::{SessionRecord, SessionState, SessionStore},
};

impl SessionStore {
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

    pub(super) async fn append_locked(
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

        open::create_parent_directory(&self.path).await?;
        let mut file = open::open_append_file(&self.path).await?;
        write::ensure_line_separator(&mut file, &self.path).await?;
        write::write_record(&mut file, &self.path, &line).await?;
        state.next_sequence = state.next_sequence.saturating_add(1);

        Ok(record)
    }
}
