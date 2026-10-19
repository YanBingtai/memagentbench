//! Durable session storage facade.
//!
//! The public `SessionStore` API stays small while append and read/repair
//! operations live in focused modules.  Both operation paths share the
//! in-process state mutex and the OS-level session lock implemented by the
//! surrounding session modules.

mod append;
mod load;

use std::{path::PathBuf, sync::Arc};

use tokio::sync::Mutex;
use uuid::Uuid;

use super::{
    error::SessionError,
    jsonl::read_records,
    lock::acquire_session_lock,
    repair::next_sequence,
    types::{SessionState, SessionStore},
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

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}
