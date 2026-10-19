//! Session record and shared state types.

use std::{fs::File as StdFile, path::PathBuf, sync::Arc};

use serde::{Deserialize, Serialize};

use tokio::sync::Mutex;
use uuid::Uuid;

use crate::message::Message;

/// One append-only JSONL entry in a session transcript.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionRecord {
    pub session_id: Uuid,
    pub sequence: u64,
    pub timestamp_ms: u64,
    pub message: Message,
}

#[derive(Debug, Default)]
pub(crate) struct SessionState {
    pub(crate) next_sequence: u64,
}

/// Holds an OS-level lock for the lifetime of one session operation.
///
/// The lock is attached to the session file itself, so the kernel releases it
/// when the process exits unexpectedly. This protects independent
/// `SessionStore::open` instances as well as separate processes.
pub(crate) struct SessionFileLock {
    pub(crate) _file: StdFile,
}

/// Durable storage for one conversation's message transcript.
///
/// The file is append-only JSONL. An in-process mutex and an OS-level file
/// lock serialize reads, repairs, and writes across clones, independent opens,
/// and separate processes; each record remains independently recoverable after
/// a restart.
#[derive(Clone, Debug)]
pub struct SessionStore {
    pub(crate) path: PathBuf,
    pub(crate) session_id: Uuid,
    pub(crate) state: Arc<Mutex<SessionState>>,
}
