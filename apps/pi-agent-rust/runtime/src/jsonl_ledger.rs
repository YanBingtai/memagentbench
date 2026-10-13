//! Append-only JSONL storage for the runtime event ledger.
//!
//! This backend keeps one JSON object per line and rebuilds its sequence and
//! idempotency indexes when it opens a file. A trailing record without a final
//! newline is treated as an interrupted write and truncated; a malformed
//! record that already has a newline is reported as corruption instead of
//! being silently discarded.
//!
//! The lock is deliberately cooperative and process-local: a sidecar file is
//! created with `create_new`. A normal close removes it, while a process crash
//! can leave a stale lock that an operator or a future lease implementation
//! must handle explicitly.

use crate::ledger::{
    AppendReceipt, DurabilityClass, EventDraft, EventSequence, LedgerError, LedgerEvent,
};
use serde_json::Error as JsonError;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Errors raised while opening or operating a JSONL ledger.
#[derive(Debug)]
pub enum JsonlLedgerError {
    Io {
        operation: &'static str,
        source: io::Error,
    },
    Serialize(JsonError),
    CorruptRecord {
        offset: u64,
        source: JsonError,
    },
    SequenceMismatch {
        expected: EventSequence,
        actual: EventSequence,
    },
    LockHeld(PathBuf),
    Ledger(LedgerError),
}

impl fmt::Display for JsonlLedgerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, source } => {
                write!(formatter, "JSONL ledger {operation} failed: {source}")
            }
            Self::Serialize(source) => {
                write!(formatter, "failed to serialize ledger event: {source}")
            }
            Self::CorruptRecord { offset, source } => {
                write!(formatter, "corrupt JSONL record at byte {offset}: {source}")
            }
            Self::SequenceMismatch { expected, actual } => write!(
                formatter,
                "ledger sequence mismatch: expected {}, found {}",
                expected.value(),
                actual.value()
            ),
            Self::LockHeld(path) => {
                write!(formatter, "ledger lock is already held: {}", path.display())
            }
            Self::Ledger(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for JsonlLedgerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Serialize(source) | Self::CorruptRecord { source, .. } => Some(source),
            Self::Ledger(source) => Some(source),
            Self::SequenceMismatch { .. } | Self::LockHeld(_) => None,
        }
    }
}

impl From<LedgerError> for JsonlLedgerError {
    fn from(source: LedgerError) -> Self {
        Self::Ledger(source)
    }
}

fn io_error(operation: &'static str, source: io::Error) -> JsonlLedgerError {
    JsonlLedgerError::Io { operation, source }
}

fn lock_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.lock", path.display()))
}

/// Guard for the cooperative sidecar lock.
struct LockGuard {
    path: PathBuf,
    _file: File,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, Default)]
struct LoadedState {
    last_sequence: EventSequence,
    event_ids: HashSet<Uuid>,
    idempotency_keys: HashMap<String, EventSequence>,
}

#[derive(Debug)]
enum TailRepair {
    Truncate(u64),
    AppendNewline,
}

#[derive(Debug)]
struct ParsedLog {
    events: Vec<LedgerEvent>,
    repair: Option<TailRepair>,
}

/// A local append-only JSONL event ledger.
pub struct JsonlLedger {
    path: PathBuf,
    data_file: Option<File>,
    _lock: Option<LockGuard>,
    last_sequence: EventSequence,
    durable_through: EventSequence,
    event_ids: HashSet<Uuid>,
    idempotency_keys: HashMap<String, EventSequence>,
    closed: bool,
}

impl fmt::Debug for JsonlLedger {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JsonlLedger")
            .field("path", &self.path)
            .field("last_sequence", &self.last_sequence)
            .field("durable_through", &self.durable_through)
            .field("closed", &self.closed)
            .finish()
    }
}

impl JsonlLedger {
    /// Opens or creates a ledger and rebuilds its indexes from existing lines.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, JsonlLedgerError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|source| io_error("create parent", source))?;
            }
        }

        let lock = acquire_lock(&path)?;
        let mut data_file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)
            .map_err(|source| io_error("open data file", source))?;

        let state = match load_state(&path, &mut data_file) {
            Ok(state) => state,
            Err(error) => {
                drop(data_file);
                drop(lock);
                return Err(error);
            }
        };

        Ok(Self {
            path,
            data_file: Some(data_file),
            _lock: Some(lock),
            last_sequence: state.last_sequence,
            durable_through: state.last_sequence,
            event_ids: state.event_ids,
            idempotency_keys: state.idempotency_keys,
            closed: false,
        })
    }

    /// Returns the data file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the last assigned event sequence.
    pub fn last_sequence(&self) -> EventSequence {
        self.last_sequence
    }

    /// Returns the sequence through which `flush` has completed.
    pub fn durable_through(&self) -> EventSequence {
        self.durable_through
    }

    /// Returns whether this handle has stopped accepting writes.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Appends canonical events and writes one JSON object per line.
    pub fn append(&mut self, drafts: Vec<EventDraft>) -> Result<AppendReceipt, JsonlLedgerError> {
        if self.closed {
            return Err(LedgerError::Closed.into());
        }
        if drafts.is_empty() {
            return Err(LedgerError::EmptyBatch.into());
        }

        let events = self.prepare_events(drafts)?;
        let encoded = events
            .iter()
            .map(|event| {
                let mut line = serde_json::to_vec(event).map_err(JsonlLedgerError::Serialize)?;
                line.push(b'\n');
                Ok(line)
            })
            .collect::<Result<Vec<_>, JsonlLedgerError>>()?;

        let count = events.len();
        let first_sequence = events
            .first()
            .map(|event| event.sequence)
            .ok_or_else(|| LedgerError::EmptyBatch)?;
        let last_sequence = events
            .last()
            .map(|event| event.sequence)
            .ok_or_else(|| LedgerError::EmptyBatch)?;

        for (event, line) in events.iter().zip(encoded.iter()) {
            {
                let file = self.data_file.as_mut().ok_or_else(|| LedgerError::Closed)?;
                file.write_all(line)
                    .map_err(|source| io_error("append event", source))?;
            }
            self.event_ids.insert(event.draft.event_id);
            if let Some(key) = &event.draft.idempotency_key {
                self.idempotency_keys.insert(key.clone(), event.sequence);
            }
            self.last_sequence = event.sequence;
        }

        self.data_file
            .as_mut()
            .ok_or_else(|| LedgerError::Closed)?
            .flush()
            .map_err(|source| io_error("flush userspace buffer", source))?;

        Ok(AppendReceipt {
            first_sequence,
            last_sequence,
            count,
        })
    }

    /// Syncs all bytes through `sequence` to the underlying file system.
    pub fn flush(&mut self, sequence: EventSequence) -> Result<(), JsonlLedgerError> {
        if self.closed {
            return Err(LedgerError::Closed.into());
        }
        if sequence > self.last_sequence {
            return Err(LedgerError::CursorOutOfRange {
                requested: sequence,
                last: self.last_sequence,
            }
            .into());
        }

        self.data_file
            .as_ref()
            .ok_or_else(|| LedgerError::Closed)?
            .sync_data()
            .map_err(|source| io_error("sync data", source))?;
        if sequence > self.durable_through {
            self.durable_through = sequence;
        }
        Ok(())
    }

    /// Reads and validates all events after a sequence cursor.
    pub fn scan_after(&self, cursor: EventSequence) -> Result<Vec<LedgerEvent>, JsonlLedgerError> {
        let data = fs::read(&self.path).map_err(|source| io_error("read data file", source))?;
        let parsed = parse_log(&data, false)?;
        if cursor > self.last_sequence {
            return Err(LedgerError::CursorOutOfRange {
                requested: cursor,
                last: self.last_sequence,
            }
            .into());
        }

        Ok(parsed
            .events
            .into_iter()
            .filter(|event| event.sequence > cursor)
            .collect())
    }

    /// Flushes the file and releases the cooperative lock.
    pub fn close(&mut self) -> Result<(), JsonlLedgerError> {
        if self.closed {
            return Ok(());
        }

        if let Some(file) = self.data_file.as_ref() {
            file.sync_all()
                .map_err(|source| io_error("sync on close", source))?;
        }
        self.data_file.take();
        self._lock.take();
        self.closed = true;
        Ok(())
    }

    fn prepare_events(
        &self,
        drafts: Vec<EventDraft>,
    ) -> Result<Vec<LedgerEvent>, JsonlLedgerError> {
        let mut event_ids = self.event_ids.clone();
        let mut idempotency_keys = self.idempotency_keys.clone();
        let mut next_sequence = self
            .last_sequence
            .checked_next()
            .ok_or(LedgerError::SequenceExhausted)?;
        let mut events = Vec::with_capacity(drafts.len());

        let count = drafts.len();
        for (index, draft) in drafts.into_iter().enumerate() {
            if draft.durability != DurabilityClass::Canonical {
                return Err(LedgerError::NonCanonicalEvent {
                    event_id: draft.event_id,
                    durability: draft.durability,
                }
                .into());
            }
            if !event_ids.insert(draft.event_id) {
                return Err(LedgerError::DuplicateEventId(draft.event_id).into());
            }
            if let Some(key) = &draft.idempotency_key {
                if idempotency_keys
                    .insert(key.clone(), next_sequence)
                    .is_some()
                {
                    return Err(LedgerError::DuplicateIdempotencyKey(key.clone()).into());
                }
            }

            events.push(LedgerEvent {
                sequence: next_sequence,
                draft,
            });
            if index + 1 < count {
                next_sequence = next_sequence
                    .checked_next()
                    .ok_or(LedgerError::SequenceExhausted)?;
            }
        }

        Ok(events)
    }
}

impl Drop for JsonlLedger {
    fn drop(&mut self) {
        self.data_file.take();
        self._lock.take();
    }
}

fn acquire_lock(path: &Path) -> Result<LockGuard, JsonlLedgerError> {
    let path = lock_path(path);
    let mut file = match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
            return Err(JsonlLedgerError::LockHeld(path));
        }
        Err(source) => return Err(io_error("create lock", source)),
    };

    if let Err(source) = writeln!(file, "pid={}", std::process::id()) {
        let _ = fs::remove_file(&path);
        return Err(io_error("write lock", source));
    }

    Ok(LockGuard { path, _file: file })
}

fn load_state(path: &Path, data_file: &mut File) -> Result<LoadedState, JsonlLedgerError> {
    let data = fs::read(path).map_err(|source| io_error("read existing data", source))?;
    let parsed = parse_log(&data, true)?;
    if let Some(repair) = parsed.repair {
        match repair {
            TailRepair::Truncate(length) => data_file
                .set_len(length)
                .map_err(|source| io_error("truncate incomplete tail", source))?,
            TailRepair::AppendNewline => data_file
                .write_all(b"\n")
                .map_err(|source| io_error("complete final record", source))?,
        }
        data_file
            .sync_data()
            .map_err(|source| io_error("sync repaired tail", source))?;
    }

    let mut state = LoadedState::default();
    let mut expected = EventSequence::INITIAL;
    for event in parsed.events {
        expected = expected
            .checked_next()
            .ok_or(LedgerError::SequenceExhausted)?;
        if event.sequence != expected {
            return Err(JsonlLedgerError::SequenceMismatch {
                expected,
                actual: event.sequence,
            });
        }
        if event.draft.durability != DurabilityClass::Canonical {
            return Err(LedgerError::NonCanonicalEvent {
                event_id: event.draft.event_id,
                durability: event.draft.durability,
            }
            .into());
        }
        if !state.event_ids.insert(event.draft.event_id) {
            return Err(LedgerError::DuplicateEventId(event.draft.event_id).into());
        }
        if let Some(key) = event.draft.idempotency_key {
            if state
                .idempotency_keys
                .insert(key.clone(), event.sequence)
                .is_some()
            {
                return Err(LedgerError::DuplicateIdempotencyKey(key).into());
            }
        }
        state.last_sequence = event.sequence;
    }

    Ok(state)
}

fn parse_log(data: &[u8], recover_tail: bool) -> Result<ParsedLog, JsonlLedgerError> {
    let mut events = Vec::new();
    let mut offset = 0_u64;
    let mut repair = None;

    for chunk in data.split_inclusive(|byte| *byte == b'\n') {
        let has_newline = chunk.ends_with(b"\n");
        let is_last = offset as usize + chunk.len() == data.len();
        let mut line = chunk;
        if has_newline {
            line = &line[..line.len() - 1];
        }
        if line.ends_with(b"\r") {
            line = &line[..line.len() - 1];
        }

        match serde_json::from_slice::<LedgerEvent>(line) {
            Ok(event) => {
                events.push(event);
                if is_last && !has_newline {
                    repair = Some(TailRepair::AppendNewline);
                }
            }
            Err(_source) if recover_tail && is_last && !has_newline => {
                repair = Some(TailRepair::Truncate(offset));
                break;
            }
            Err(source) => {
                return Err(JsonlLedgerError::CorruptRecord { offset, source });
            }
        }
        offset += chunk.len() as u64;
    }

    Ok(ParsedLog { events, repair })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EventKind, Revision, RunId};

    fn path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("pi-agent-runtime-{name}-{}.jsonl", Uuid::new_v4()))
    }

    fn draft(kind: &str) -> EventDraft {
        EventDraft::new(
            RunId::new(),
            Revision::new(1),
            EventKind::new(kind).expect("test event kind should be valid"),
            serde_json::json!({"kind": kind}),
            42,
        )
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(lock_path(path));
    }

    #[test]
    fn reopens_and_replays_jsonl_events() {
        let file_path = path("replay");
        let mut ledger = JsonlLedger::open(&file_path).expect("ledger should open");
        ledger
            .append(vec![draft("run_started"), draft("run_finished")])
            .expect("events should append");
        ledger
            .flush(EventSequence::new(2))
            .expect("flush should succeed");
        ledger.close().expect("close should succeed");

        let ledger = JsonlLedger::open(&file_path).expect("ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("events should replay");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sequence, EventSequence::new(1));
        assert_eq!(events[1].sequence, EventSequence::new(2));
        cleanup(&file_path);
    }

    #[test]
    fn repairs_an_incomplete_trailing_record() {
        let file_path = path("repair");
        let mut ledger = JsonlLedger::open(&file_path).expect("ledger should open");
        ledger
            .append(vec![draft("run_started")])
            .expect("event should append");
        ledger.close().expect("close should succeed");

        OpenOptions::new()
            .append(true)
            .open(&file_path)
            .expect("data file should open")
            .write_all(br#"{"sequence":"#)
            .expect("partial record should write");

        let ledger = JsonlLedger::open(&file_path).expect("incomplete tail should be repaired");
        assert_eq!(ledger.last_sequence(), EventSequence::new(1));
        assert_eq!(ledger.scan_after(EventSequence::INITIAL).unwrap().len(), 1);
        cleanup(&file_path);
    }

    #[test]
    fn rejects_a_complete_corrupt_record() {
        let file_path = path("corrupt");
        fs::write(&file_path, b"not-json\n").expect("corrupt fixture should write");

        let error = JsonlLedger::open(&file_path).expect_err("corruption should be visible");
        assert!(matches!(
            error,
            JsonlLedgerError::CorruptRecord { offset: 0, .. }
        ));
        cleanup(&file_path);
    }

    #[test]
    fn lock_prevents_two_writers_and_is_released_on_close() {
        let file_path = path("lock");
        let mut first = JsonlLedger::open(&file_path).expect("first writer should open");
        let error = JsonlLedger::open(&file_path).expect_err("second writer must be rejected");
        assert!(matches!(error, JsonlLedgerError::LockHeld(_)));

        first.close().expect("first writer should close");
        let mut second = JsonlLedger::open(&file_path).expect("lock should be released");
        second.close().expect("second writer should close");
        cleanup(&file_path);
    }
}
