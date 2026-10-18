//! Append-only JSONL storage for the runtime event ledger.
//!
//! This backend keeps one JSON object per line and rebuilds its sequence and
//! idempotency indexes when it opens a file. Each append writes event records
//! followed by a batch commit marker. A batch without that marker is treated
//! as an interrupted write and truncated; a malformed record that already has
//! a newline is reported as corruption instead of being silently discarded.
//! New files begin with a format header. Older event-only files are validated
//! and atomically migrated before they accept new writes, so the first new
//! batch also has an unambiguous recovery boundary.
//!
//! The lock is deliberately cooperative and process-local: a sidecar file is
//! created with `create_new`. A normal close removes it, while a process crash
//! can leave a stale lock. On Linux, a lock that names a dead PID is reclaimed;
//! malformed locks remain visible instead of being guessed away.

use crate::ledger::{
    AppendReceipt, DurabilityClass, EventDraft, EventLedger, EventSequence, LedgerError,
    LedgerEvent,
};
use serde::{Deserialize, Serialize};
use serde_json::{error::Category, Error as JsonError};
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
    InvalidCommit {
        offset: u64,
        message: String,
    },
    InvalidHeader {
        offset: u64,
        message: String,
    },
    IncompleteBatch {
        offset: u64,
    },
    SequenceMismatch {
        expected: EventSequence,
        actual: EventSequence,
    },
    LockHeld(PathBuf),
    InvalidLock {
        path: PathBuf,
        message: String,
    },
    WriteFailed,
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
            Self::InvalidCommit { offset, message } => {
                write!(
                    formatter,
                    "invalid JSONL batch commit at byte {offset}: {message}"
                )
            }
            Self::InvalidHeader { offset, message } => {
                write!(
                    formatter,
                    "invalid JSONL ledger header at byte {offset}: {message}"
                )
            }
            Self::IncompleteBatch { offset } => {
                write!(formatter, "incomplete JSONL batch begins at byte {offset}")
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
            Self::InvalidLock { path, message } => {
                write!(
                    formatter,
                    "invalid ledger lock {}: {message}",
                    path.display()
                )
            }
            Self::WriteFailed => {
                formatter.write_str("ledger handle requires reopen after an interrupted append")
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
            Self::SequenceMismatch { .. }
            | Self::InvalidCommit { .. }
            | Self::InvalidHeader { .. }
            | Self::IncompleteBatch { .. }
            | Self::LockHeld(_)
            | Self::InvalidLock { .. }
            | Self::WriteFailed => None,
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
    format: LogFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogFormat {
    Framed,
    Legacy,
}

/// A marker written after every event batch.
///
/// Event lines before this marker are provisional. On recovery, a batch with
/// no marker is discarded so a process crash cannot expose a partial action as
/// canonical history. Older JSONL files without markers remain readable as a
/// legacy format.
#[derive(Debug, Serialize, Deserialize)]
struct BatchCommit {
    record_type: String,
    batch_id: Uuid,
    first_sequence: EventSequence,
    last_sequence: EventSequence,
    count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct LedgerHeader {
    record_type: String,
    format_version: u8,
}

impl LedgerHeader {
    fn current() -> Self {
        Self {
            record_type: "ledger_header".to_owned(),
            format_version: 1,
        }
    }
}

impl BatchCommit {
    fn new(first_sequence: EventSequence, last_sequence: EventSequence, count: usize) -> Self {
        Self {
            record_type: "batch_commit".to_owned(),
            batch_id: Uuid::new_v4(),
            first_sequence,
            last_sequence,
            count,
        }
    }
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
    write_failed: bool,
    closed: bool,
}

impl fmt::Debug for JsonlLedger {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JsonlLedger")
            .field("path", &self.path)
            .field("last_sequence", &self.last_sequence)
            .field("durable_through", &self.durable_through)
            .field("write_failed", &self.write_failed)
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

        let legacy_events = if data_file
            .metadata()
            .map_err(|source| io_error("inspect data file", source))?
            .len()
            == 0
        {
            write_header(&mut data_file)?;
            None
        } else {
            let data = fs::read(&path).map_err(|source| io_error("read existing data", source))?;
            let parsed = parse_log(&data, true)?;
            if parsed.format == LogFormat::Legacy && !parsed.events.is_empty() {
                Some(parsed.events)
            } else {
                None
            }
        };

        let state = match load_state(&path, &mut data_file) {
            Ok(state) => state,
            Err(error) => {
                drop(data_file);
                drop(lock);
                return Err(error);
            }
        };

        if let Some(events) = legacy_events {
            drop(data_file);
            migrate_legacy(&path, &events)?;
            data_file = OpenOptions::new()
                .read(true)
                .append(true)
                .open(&path)
                .map_err(|source| io_error("reopen migrated data file", source))?;
        }

        Ok(Self {
            path,
            data_file: Some(data_file),
            _lock: Some(lock),
            last_sequence: state.last_sequence,
            durable_through: state.last_sequence,
            event_ids: state.event_ids,
            idempotency_keys: state.idempotency_keys,
            write_failed: false,
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
        if self.write_failed {
            return Err(JsonlLedgerError::WriteFailed);
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
            .ok_or(LedgerError::EmptyBatch)?;
        let last_sequence = events
            .last()
            .map(|event| event.sequence)
            .ok_or(LedgerError::EmptyBatch)?;

        let mut commit_line =
            serde_json::to_vec(&BatchCommit::new(first_sequence, last_sequence, count))
                .map_err(JsonlLedgerError::Serialize)?;
        commit_line.push(b'\n');

        let file = self.data_file.as_mut().ok_or(LedgerError::Closed)?;
        for line in &encoded {
            if let Err(source) = file.write_all(line) {
                self.write_failed = true;
                return Err(io_error("append event", source));
            }
        }
        if let Err(source) = file.write_all(&commit_line) {
            self.write_failed = true;
            return Err(io_error("append batch commit", source));
        }

        if let Err(source) = file.flush() {
            self.write_failed = true;
            return Err(io_error("flush userspace buffer", source));
        }

        for event in events {
            self.event_ids.insert(event.draft.event_id);
            if let Some(key) = event.draft.idempotency_key {
                self.idempotency_keys.insert(key, event.sequence);
            }
            self.last_sequence = event.sequence;
        }

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
        if self.write_failed {
            return Err(JsonlLedgerError::WriteFailed);
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
            .ok_or(LedgerError::Closed)?
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

fn into_ledger_error(error: JsonlLedgerError) -> LedgerError {
    match error {
        JsonlLedgerError::Ledger(error) => error,
        other => LedgerError::Storage(other.to_string()),
    }
}

impl EventLedger for JsonlLedger {
    fn append(&mut self, drafts: Vec<EventDraft>) -> Result<AppendReceipt, LedgerError> {
        JsonlLedger::append(self, drafts).map_err(into_ledger_error)
    }

    fn flush(&mut self, sequence: EventSequence) -> Result<(), LedgerError> {
        JsonlLedger::flush(self, sequence).map_err(into_ledger_error)
    }

    fn scan_after(&self, cursor: EventSequence) -> Result<Vec<LedgerEvent>, LedgerError> {
        JsonlLedger::scan_after(self, cursor).map_err(into_ledger_error)
    }

    fn close(&mut self) -> Result<(), LedgerError> {
        JsonlLedger::close(self).map_err(into_ledger_error)
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
    loop {
        let mut file = match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => file,
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                if reclaim_dead_lock(&path)? {
                    continue;
                }
                return Err(JsonlLedgerError::LockHeld(path));
            }
            Err(source) => return Err(io_error("create lock", source)),
        };

        if let Err(source) = writeln!(file, "pid={}", std::process::id()) {
            let _ = fs::remove_file(&path);
            return Err(io_error("write lock", source));
        }
        if let Err(source) = file.sync_data() {
            let _ = fs::remove_file(&path);
            return Err(io_error("sync lock", source));
        }

        return Ok(LockGuard { path, _file: file });
    }
}

fn reclaim_dead_lock(path: &Path) -> Result<bool, JsonlLedgerError> {
    let content = fs::read_to_string(path).map_err(|source| io_error("read lock", source))?;
    let pid = content
        .lines()
        .find_map(|line| {
            line.strip_prefix("pid=")
                .and_then(|value| value.parse::<u32>().ok())
        })
        .ok_or_else(|| JsonlLedgerError::InvalidLock {
            path: path.to_path_buf(),
            message: "expected a numeric pid= entry".to_owned(),
        })?;

    if process_is_alive(pid) {
        return Ok(false);
    }

    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(true),
        Err(source) => Err(io_error("remove stale lock", source)),
    }
}

#[cfg(target_os = "linux")]
fn process_is_alive(pid: u32) -> bool {
    Path::new("/proc").join(pid.to_string()).exists()
}

#[cfg(not(target_os = "linux"))]
fn process_is_alive(pid: u32) -> bool {
    // Without a portable process-liveness API, fail closed and only recognize
    // this process. Operators can remove a stale sidecar on other platforms.
    pid == std::process::id()
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

fn write_header(data_file: &mut File) -> Result<(), JsonlLedgerError> {
    let mut line =
        serde_json::to_vec(&LedgerHeader::current()).map_err(JsonlLedgerError::Serialize)?;
    line.push(b'\n');
    data_file
        .write_all(&line)
        .map_err(|source| io_error("write ledger header", source))?;
    data_file
        .flush()
        .map_err(|source| io_error("flush ledger header", source))?;
    data_file
        .sync_data()
        .map_err(|source| io_error("sync ledger header", source))
}

fn migrate_legacy(path: &Path, events: &[LedgerEvent]) -> Result<(), JsonlLedgerError> {
    let temporary_path = PathBuf::from(format!("{}.migrate-{}", path.display(), Uuid::new_v4()));
    let result = (|| {
        let mut temporary = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
            .map_err(|source| io_error("create migration file", source))?;
        write_header(&mut temporary)?;

        for event in events {
            let mut line = serde_json::to_vec(event).map_err(JsonlLedgerError::Serialize)?;
            line.push(b'\n');
            temporary
                .write_all(&line)
                .map_err(|source| io_error("write migrated event", source))?;
        }

        let first = events
            .first()
            .expect("legacy migration requires at least one event")
            .sequence;
        let last = events
            .last()
            .expect("legacy migration requires at least one event")
            .sequence;
        let mut commit_line = serde_json::to_vec(&BatchCommit::new(first, last, events.len()))
            .map_err(JsonlLedgerError::Serialize)?;
        commit_line.push(b'\n');
        temporary
            .write_all(&commit_line)
            .map_err(|source| io_error("write migrated batch commit", source))?;
        temporary
            .sync_all()
            .map_err(|source| io_error("sync migration file", source))?;
        drop(temporary);

        fs::rename(&temporary_path, path)
            .map_err(|source| io_error("replace legacy ledger", source))
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

fn parse_log(data: &[u8], recover_tail: bool) -> Result<ParsedLog, JsonlLedgerError> {
    let mut events = Vec::new();
    let mut pending = Vec::new();
    let mut pending_offset = None;
    let mut framed = false;
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
                pending_offset.get_or_insert(offset);
                pending.push(event);
            }
            Err(event_error) => match serde_json::from_slice::<BatchCommit>(line) {
                Ok(commit) => {
                    let legacy_count = if framed {
                        0
                    } else {
                        pending
                            .iter()
                            .take_while(|event| event.sequence < commit.first_sequence)
                            .count()
                    };
                    events.extend(pending.drain(..legacy_count));

                    validate_commit(offset, &commit, &pending)?;
                    events.append(&mut pending);
                    pending_offset = None;
                    framed = true;
                    if recover_tail && is_last && !has_newline {
                        repair = Some(TailRepair::AppendNewline);
                    }
                }
                Err(_) => match serde_json::from_slice::<LedgerHeader>(line) {
                    Ok(header) => {
                        if header.record_type != "ledger_header" || header.format_version != 1 {
                            return Err(JsonlLedgerError::InvalidHeader {
                                offset,
                                message: format!(
                                    "expected ledger_header version 1, found type {:?} version {}",
                                    header.record_type, header.format_version
                                ),
                            });
                        }
                        if offset != 0 || !events.is_empty() || !pending.is_empty() || framed {
                            return Err(JsonlLedgerError::InvalidHeader {
                                offset,
                                message: "ledger header must be the first record".to_owned(),
                            });
                        }
                        framed = true;
                        if recover_tail && is_last && !has_newline {
                            repair = Some(TailRepair::AppendNewline);
                        }
                    }
                    Err(_)
                        if recover_tail
                            && is_last
                            && !has_newline
                            && is_truncated_json(&event_error) =>
                    {
                        let truncate_at = if framed {
                            pending_offset.unwrap_or(offset)
                        } else {
                            offset
                        };
                        repair = Some(TailRepair::Truncate(truncate_at));
                        break;
                    }
                    Err(_) => {
                        return Err(JsonlLedgerError::CorruptRecord {
                            offset,
                            source: event_error,
                        });
                    }
                },
            },
        }
        offset += chunk.len() as u64;
    }

    if framed {
        if !pending.is_empty() {
            let batch_offset = pending_offset.expect("pending events have an offset");
            if recover_tail {
                repair = Some(TailRepair::Truncate(batch_offset));
            } else {
                return Err(JsonlLedgerError::IncompleteBatch {
                    offset: batch_offset,
                });
            }
        }
    } else {
        events.append(&mut pending);
        if recover_tail && data.last().is_some_and(|byte| *byte != b'\n') && repair.is_none() {
            repair = Some(TailRepair::AppendNewline);
        }
    }

    Ok(ParsedLog {
        events,
        repair,
        format: if framed {
            LogFormat::Framed
        } else {
            LogFormat::Legacy
        },
    })
}

fn is_truncated_json(error: &JsonError) -> bool {
    matches!(error.classify(), Category::Eof)
}

fn validate_commit(
    offset: u64,
    commit: &BatchCommit,
    events: &[LedgerEvent],
) -> Result<(), JsonlLedgerError> {
    if commit.record_type != "batch_commit" {
        return Err(JsonlLedgerError::InvalidCommit {
            offset,
            message: format!("unknown record type {:?}", commit.record_type),
        });
    }
    if commit.count == 0 || commit.count != events.len() {
        return Err(JsonlLedgerError::InvalidCommit {
            offset,
            message: format!(
                "commit count {} does not match pending event count {}",
                commit.count,
                events.len()
            ),
        });
    }

    let first = events
        .first()
        .expect("a non-empty commit must have pending events");
    let last = events
        .last()
        .expect("a non-empty commit must have pending events");
    if first.sequence != commit.first_sequence || last.sequence != commit.last_sequence {
        return Err(JsonlLedgerError::InvalidCommit {
            offset,
            message: format!(
                "commit range {}..={} does not match event range {}..={}",
                commit.first_sequence.value(),
                commit.last_sequence.value(),
                first.sequence.value(),
                last.sequence.value()
            ),
        });
    }
    for pair in events.windows(2) {
        if pair[0].sequence.checked_next() != Some(pair[1].sequence) {
            return Err(JsonlLedgerError::InvalidCommit {
                offset,
                message: "pending event sequences are not contiguous".to_owned(),
            });
        }
    }
    Ok(())
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
    fn writes_a_commit_marker_after_each_batch() {
        let file_path = path("commit-marker");
        let mut ledger = JsonlLedger::open(&file_path).expect("ledger should open");
        ledger
            .append(vec![draft("run_started")])
            .expect("event should append");
        ledger.close().expect("close should succeed");

        let lines = fs::read_to_string(&file_path).expect("JSONL should be readable");
        assert!(lines.lines().last().unwrap().contains("\"batch_id\""));
        assert!(lines.lines().last().unwrap().contains("batch_commit"));
        cleanup(&file_path);
    }

    #[test]
    fn discards_an_uncommitted_batch_during_reopen() {
        let file_path = path("uncommitted-batch");
        let mut ledger = JsonlLedger::open(&file_path).expect("ledger should open");
        ledger
            .append(vec![draft("run_started")])
            .expect("committed event should append");
        ledger.close().expect("close should succeed");

        let uncommitted = LedgerEvent {
            sequence: EventSequence::new(2),
            draft: draft("run_finished"),
        };
        let mut line = serde_json::to_vec(&uncommitted).expect("event should serialize");
        line.push(b'\n');
        OpenOptions::new()
            .append(true)
            .open(&file_path)
            .expect("data file should open")
            .write_all(&line)
            .expect("uncommitted event should write");

        let ledger = JsonlLedger::open(&file_path).expect("tail batch should be repaired");
        assert_eq!(ledger.last_sequence(), EventSequence::new(1));
        assert_eq!(ledger.scan_after(EventSequence::INITIAL).unwrap().len(), 1);
        cleanup(&file_path);
    }

    #[test]
    fn discards_the_first_uncommitted_batch_after_a_header() {
        let file_path = path("first-uncommitted-batch");
        let header = serde_json::to_vec(&LedgerHeader::current()).expect("header should serialize");
        let event = LedgerEvent {
            sequence: EventSequence::new(1),
            draft: draft("run_started"),
        };
        let mut data = header;
        data.push(b'\n');
        data.extend(serde_json::to_vec(&event).expect("event should serialize"));
        data.push(b'\n');
        fs::write(&file_path, data).expect("crash fixture should write");

        let ledger = JsonlLedger::open(&file_path).expect("first incomplete batch should repair");
        assert_eq!(ledger.last_sequence(), EventSequence::INITIAL);
        assert!(ledger
            .scan_after(EventSequence::INITIAL)
            .expect("repaired ledger should scan")
            .is_empty());
        cleanup(&file_path);
    }

    #[test]
    fn reads_legacy_event_only_jsonl_files() {
        let file_path = path("legacy");
        let event = LedgerEvent {
            sequence: EventSequence::new(1),
            draft: draft("legacy_event"),
        };
        let mut line = serde_json::to_vec(&event).expect("event should serialize");
        line.push(b'\n');
        fs::write(&file_path, line).expect("legacy fixture should write");

        let ledger = JsonlLedger::open(&file_path).expect("legacy file should remain readable");
        assert_eq!(ledger.last_sequence(), EventSequence::new(1));
        assert_eq!(ledger.scan_after(EventSequence::INITIAL).unwrap().len(), 1);
        let migrated = fs::read_to_string(&file_path).expect("migrated file should be readable");
        assert!(migrated.lines().next().unwrap().contains("ledger_header"));
        assert!(migrated.lines().any(|line| line.contains("batch_commit")));
        cleanup(&file_path);
    }

    #[test]
    fn migrates_legacy_files_before_accepting_new_batches() {
        let file_path = path("legacy-append");
        let legacy_event = LedgerEvent {
            sequence: EventSequence::new(1),
            draft: draft("legacy_event"),
        };
        let mut legacy_line =
            serde_json::to_vec(&legacy_event).expect("legacy event should serialize");
        legacy_line.push(b'\n');
        fs::write(&file_path, legacy_line).expect("legacy fixture should write");

        let mut ledger = JsonlLedger::open(&file_path).expect("legacy file should migrate");
        ledger
            .append(vec![draft("new_event")])
            .expect("new batch should append after migration");
        ledger.close().expect("close should succeed");

        let uncommitted = LedgerEvent {
            sequence: EventSequence::new(3),
            draft: draft("crashed_event"),
        };
        let mut line = serde_json::to_vec(&uncommitted).expect("event should serialize");
        line.push(b'\n');
        OpenOptions::new()
            .append(true)
            .open(&file_path)
            .expect("data file should open")
            .write_all(&line)
            .expect("uncommitted event should write");

        let ledger = JsonlLedger::open(&file_path).expect("uncommitted tail should repair");
        assert_eq!(ledger.last_sequence(), EventSequence::new(2));
        assert_eq!(ledger.scan_after(EventSequence::INITIAL).unwrap().len(), 2);
        cleanup(&file_path);
    }

    #[test]
    fn rejects_a_complete_schema_error_without_a_newline() {
        let file_path = path("schema-error");
        fs::write(&file_path, br#"{"sequence":1}"#).expect("invalid fixture should write");

        let error = JsonlLedger::open(&file_path).expect_err("schema error must be visible");
        assert!(matches!(
            error,
            JsonlLedgerError::CorruptRecord { offset: 0, .. }
        ));
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

    #[test]
    fn reclaims_a_lock_owned_by_a_dead_process() {
        let file_path = path("stale-lock");
        fs::write(lock_path(&file_path), b"pid=4294967295\n")
            .expect("stale lock fixture should write");

        let mut ledger = JsonlLedger::open(&file_path).expect("dead lock should be reclaimed");
        ledger
            .append(vec![draft("run_started")])
            .expect("event should append after stale lock recovery");
        ledger.close().expect("close should succeed");
        cleanup(&file_path);
    }

    #[test]
    fn refuses_to_guess_about_a_malformed_lock() {
        let file_path = path("malformed-lock");
        fs::write(lock_path(&file_path), b"owner=unknown\n")
            .expect("malformed lock fixture should write");

        let error = JsonlLedger::open(&file_path).expect_err("malformed lock must be visible");
        assert!(matches!(error, JsonlLedgerError::InvalidLock { .. }));
        cleanup(&file_path);
    }
}
