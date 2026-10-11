//! Canonical runtime events and their ordering semantics.
//!
//! The ledger is the runtime's factual boundary. It records durable events in
//! one global sequence, while the event payload remains independent of any
//! concrete storage backend. A file or database implementation can later use
//! the same drafts and receipts without changing the action or memory APIs.
//!
//! [`InMemoryLedger`] is deliberately a deterministic test double. Its
//! `flush` method advances a logical durability watermark; it does not claim
//! that bytes survived a process crash. A durable backend must preserve the
//! same ordering and flush contract before it is used by production runs.

use crate::{Revision, RunId};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fmt;
use uuid::Uuid;

/// A monotonically increasing position in the canonical event stream.
///
/// Sequence zero is the cursor before the first event. The first appended
/// event receives sequence one. Sequences are global to one ledger, so a
/// reader can replay several runs without merging per-run counters.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct EventSequence(u64);

impl EventSequence {
    /// Cursor before the first event.
    pub const INITIAL: Self = Self(0);

    /// Creates a sequence from a persisted value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric value used by storage cursors.
    pub const fn value(self) -> u64 {
        self.0
    }

    /// Returns the next sequence, or `None` when the counter is exhausted.
    pub const fn checked_next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

/// Identifies the stable event vocabulary without constraining provider code.
///
/// Runtime-owned event names use snake_case, while adapters may reserve a
/// namespaced value such as `memory.provider_name.observation_recorded`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EventKind(String);

impl EventKind {
    /// Creates an event kind and rejects an empty name.
    pub fn new(value: impl Into<String>) -> Result<Self, EventDraftError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(EventDraftError::EmptyKind);
        }

        Ok(Self(value))
    }

    /// Returns the stable serialized name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EventKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Persistence policy applied before an event enters the canonical ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DurabilityClass {
    /// Replayed as part of the authoritative runtime history.
    Canonical,
    /// Stored only in a derived query or metrics projection.
    ProjectionOnly,
    /// Kept in memory or telemetry and discarded from the canonical history.
    Ephemeral,
}

/// A validated event that has not yet received a ledger sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventDraft {
    /// Stable event identity used for duplicate detection and tracing.
    pub event_id: Uuid,
    pub run_id: RunId,
    pub revision: Revision,
    pub kind: EventKind,
    pub durability: DurabilityClass,
    /// Optional key for retry-safe writes from an action or provider adapter.
    pub idempotency_key: Option<String>,
    /// Runtime-provided wall-clock timestamp in milliseconds.
    pub created_at_ms: u64,
    pub payload: Value,
    /// Digest supplied by the caller after canonical payload serialization.
    pub payload_digest: Option<String>,
}

impl EventDraft {
    /// Creates a canonical event draft with a new event id.
    pub fn new(
        run_id: RunId,
        revision: Revision,
        kind: EventKind,
        payload: Value,
        created_at_ms: u64,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4(),
            run_id,
            revision,
            kind,
            durability: DurabilityClass::Canonical,
            idempotency_key: None,
            created_at_ms,
            payload,
            payload_digest: None,
        }
    }

    /// Adds an idempotency key used to reject duplicate external operations.
    pub fn with_idempotency_key(mut self, key: impl Into<String>) -> Result<Self, EventDraftError> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(EventDraftError::EmptyIdempotencyKey);
        }

        self.idempotency_key = Some(key);
        Ok(self)
    }

    /// Attaches a caller-computed digest without choosing a hash algorithm.
    pub fn with_payload_digest(
        mut self,
        digest: impl Into<String>,
    ) -> Result<Self, EventDraftError> {
        let digest = digest.into();
        if digest.trim().is_empty() {
            return Err(EventDraftError::EmptyPayloadDigest);
        }

        self.payload_digest = Some(digest);
        Ok(self)
    }

    /// Changes the persistence class before the draft is submitted.
    ///
    /// The canonical ledger intentionally rejects non-canonical drafts. This
    /// setter exists so the same envelope can be validated by a projection or
    /// telemetry pipeline without inventing a second event representation.
    pub fn with_durability(mut self, durability: DurabilityClass) -> Self {
        self.durability = durability;
        self
    }
}

/// An event after the ledger has assigned its global sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerEvent {
    pub sequence: EventSequence,
    #[serde(flatten)]
    pub draft: EventDraft,
}

/// Receipt returned after an entire batch has been accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendReceipt {
    pub first_sequence: EventSequence,
    pub last_sequence: EventSequence,
    pub count: usize,
}

/// Errors that can be found before a batch changes the ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventDraftError {
    EmptyKind,
    EmptyIdempotencyKey,
    EmptyPayloadDigest,
}

impl fmt::Display for EventDraftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyKind => "event kind must not be empty",
            Self::EmptyIdempotencyKey => "idempotency key must not be empty",
            Self::EmptyPayloadDigest => "payload digest must not be empty",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for EventDraftError {}

/// Errors returned by a ledger operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    EmptyBatch,
    Closed,
    SequenceExhausted,
    DuplicateEventId(Uuid),
    DuplicateIdempotencyKey(String),
    NonCanonicalEvent {
        event_id: Uuid,
        durability: DurabilityClass,
    },
    CursorOutOfRange {
        requested: EventSequence,
        last: EventSequence,
    },
}

impl fmt::Display for LedgerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => formatter.write_str("event batch must not be empty"),
            Self::Closed => formatter.write_str("ledger is closed"),
            Self::SequenceExhausted => formatter.write_str("event sequence is exhausted"),
            Self::DuplicateEventId(event_id) => {
                write!(formatter, "event id {event_id} was already appended")
            }
            Self::DuplicateIdempotencyKey(key) => {
                write!(formatter, "idempotency key {key:?} was already appended")
            }
            Self::NonCanonicalEvent {
                event_id,
                durability,
            } => write!(
                formatter,
                "event {event_id} has non-canonical durability class {durability:?}"
            ),
            Self::CursorOutOfRange { requested, last } => write!(
                formatter,
                "cursor {} is after the ledger tail {}",
                requested.value(),
                last.value()
            ),
        }
    }
}

impl std::error::Error for LedgerError {}

/// Contract implemented by the runtime's canonical event stores.
pub trait EventLedger {
    /// Appends a batch atomically and returns its assigned sequence range.
    fn append(&mut self, drafts: Vec<EventDraft>) -> Result<AppendReceipt, LedgerError>;

    /// Makes every event through `sequence` visible at the backend's durable boundary.
    fn flush(&mut self, sequence: EventSequence) -> Result<(), LedgerError>;

    /// Replays events strictly after a cursor.
    fn scan_after(&self, cursor: EventSequence) -> Result<Vec<LedgerEvent>, LedgerError>;

    /// Stops accepting new events while keeping already-written events readable.
    fn close(&mut self) -> Result<(), LedgerError>;
}

/// Deterministic in-memory ledger used by unit tests and later conformance tests.
#[derive(Debug, Default)]
pub struct InMemoryLedger {
    events: Vec<LedgerEvent>,
    event_ids: HashSet<Uuid>,
    idempotency_keys: HashMap<String, EventSequence>,
    durable_through: EventSequence,
    closed: bool,
}

impl InMemoryLedger {
    /// Creates an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the last assigned sequence, or the initial cursor when empty.
    pub fn last_sequence(&self) -> EventSequence {
        self.events
            .last()
            .map_or(EventSequence::INITIAL, |event| event.sequence)
    }

    /// Returns the logical flush watermark.
    pub fn durable_through(&self) -> EventSequence {
        self.durable_through
    }

    /// Returns whether this ledger has stopped accepting writes.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    fn validate_batch(&self, drafts: &[EventDraft]) -> Result<(), LedgerError> {
        if drafts.is_empty() {
            return Err(LedgerError::EmptyBatch);
        }

        let mut event_ids = self.event_ids.clone();
        let mut idempotency_keys = self.idempotency_keys.clone();

        for draft in drafts {
            if draft.durability != DurabilityClass::Canonical {
                return Err(LedgerError::NonCanonicalEvent {
                    event_id: draft.event_id,
                    durability: draft.durability,
                });
            }

            if !event_ids.insert(draft.event_id) {
                return Err(LedgerError::DuplicateEventId(draft.event_id));
            }

            if let Some(key) = &draft.idempotency_key {
                if idempotency_keys
                    .insert(key.clone(), EventSequence::INITIAL)
                    .is_some()
                {
                    return Err(LedgerError::DuplicateIdempotencyKey(key.clone()));
                }
            }
        }

        let mut sequence = self.last_sequence();
        for _ in drafts {
            sequence = sequence
                .checked_next()
                .ok_or(LedgerError::SequenceExhausted)?;
        }

        Ok(())
    }
}

impl EventLedger for InMemoryLedger {
    fn append(&mut self, drafts: Vec<EventDraft>) -> Result<AppendReceipt, LedgerError> {
        if self.closed {
            return Err(LedgerError::Closed);
        }

        self.validate_batch(&drafts)?;

        let first_sequence = self
            .last_sequence()
            .checked_next()
            .ok_or(LedgerError::SequenceExhausted)?;
        let count = drafts.len();
        let mut next_sequence = first_sequence;
        let mut last_sequence = first_sequence;

        for (index, draft) in drafts.into_iter().enumerate() {
            if let Some(key) = &draft.idempotency_key {
                self.idempotency_keys.insert(key.clone(), next_sequence);
            }
            self.event_ids.insert(draft.event_id);
            self.events.push(LedgerEvent {
                sequence: next_sequence,
                draft,
            });
            last_sequence = next_sequence;

            if index + 1 < count {
                next_sequence = next_sequence
                    .checked_next()
                    .ok_or(LedgerError::SequenceExhausted)?;
            }
        }

        Ok(AppendReceipt {
            first_sequence,
            last_sequence,
            count,
        })
    }

    fn flush(&mut self, sequence: EventSequence) -> Result<(), LedgerError> {
        if self.closed {
            return Err(LedgerError::Closed);
        }

        let last_sequence = self.last_sequence();
        if sequence > last_sequence {
            return Err(LedgerError::CursorOutOfRange {
                requested: sequence,
                last: last_sequence,
            });
        }

        if sequence > self.durable_through {
            self.durable_through = sequence;
        }
        Ok(())
    }

    fn scan_after(&self, cursor: EventSequence) -> Result<Vec<LedgerEvent>, LedgerError> {
        let last_sequence = self.last_sequence();
        if cursor > last_sequence {
            return Err(LedgerError::CursorOutOfRange {
                requested: cursor,
                last: last_sequence,
            });
        }

        Ok(self
            .events
            .iter()
            .filter(|event| event.sequence > cursor)
            .cloned()
            .collect())
    }

    fn close(&mut self) -> Result<(), LedgerError> {
        self.closed = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(kind: &str, payload: Value) -> EventDraft {
        EventDraft::new(
            RunId::new(),
            Revision::new(1),
            EventKind::new(kind).expect("test event kind should be valid"),
            payload,
            42,
        )
    }

    #[test]
    fn append_assigns_global_sequences_and_returns_batch_range() {
        let mut ledger = InMemoryLedger::new();

        let receipt = ledger
            .append(vec![
                draft("run_started", serde_json::json!({"name": "first"})),
                draft("action_proposed", serde_json::json!({"tool": "read_file"})),
            ])
            .expect("canonical batch should append");

        assert_eq!(receipt.first_sequence, EventSequence::new(1));
        assert_eq!(receipt.last_sequence, EventSequence::new(2));
        assert_eq!(receipt.count, 2);
        assert_eq!(ledger.last_sequence(), EventSequence::new(2));
    }

    #[test]
    fn failed_batch_does_not_partially_change_the_ledger() {
        let mut ledger = InMemoryLedger::new();
        let projection_event = draft("projection_only", serde_json::json!({}))
            .with_durability(DurabilityClass::ProjectionOnly);

        let error = ledger
            .append(vec![
                draft("run_started", serde_json::json!({})),
                projection_event,
            ])
            .expect_err("projection events must not enter the canonical ledger");

        assert!(matches!(error, LedgerError::NonCanonicalEvent { .. }));
        assert_eq!(ledger.last_sequence(), EventSequence::INITIAL);
        assert!(ledger
            .scan_after(EventSequence::INITIAL)
            .expect("initial cursor should be readable")
            .is_empty());
    }

    #[test]
    fn scan_is_cursor_based_and_flush_is_monotonic() {
        let mut ledger = InMemoryLedger::new();
        ledger
            .append(vec![draft("run_started", serde_json::json!({}))])
            .expect("event should append");
        ledger
            .append(vec![draft("run_finished", serde_json::json!({}))])
            .expect("event should append");

        let events = ledger
            .scan_after(EventSequence::new(1))
            .expect("cursor should be valid");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].sequence, EventSequence::new(2));

        ledger
            .flush(EventSequence::new(2))
            .expect("tail should be flushable");
        ledger
            .flush(EventSequence::new(1))
            .expect("older flushes should be harmless");
        assert_eq!(ledger.durable_through(), EventSequence::new(2));
    }

    #[test]
    fn duplicate_idempotency_keys_are_rejected_before_append() {
        let mut ledger = InMemoryLedger::new();
        let first = draft("action_committed", serde_json::json!({}))
            .with_idempotency_key("operation-1")
            .expect("key should be valid");
        ledger
            .append(vec![first])
            .expect("first write should append");

        let duplicate = draft("action_committed", serde_json::json!({}))
            .with_idempotency_key("operation-1")
            .expect("key should be valid");
        let error = ledger
            .append(vec![duplicate])
            .expect_err("duplicate operation must be rejected");

        assert_eq!(
            error,
            LedgerError::DuplicateIdempotencyKey("operation-1".to_string())
        );
        assert_eq!(ledger.last_sequence(), EventSequence::new(1));
    }

    #[test]
    fn close_stops_writes_but_keeps_reads_available() {
        let mut ledger = InMemoryLedger::new();
        ledger
            .append(vec![draft("run_started", serde_json::json!({}))])
            .expect("event should append");
        ledger.close().expect("close should succeed");

        assert!(ledger.is_closed());
        assert_eq!(
            ledger
                .append(vec![draft("run_finished", serde_json::json!({}))])
                .expect_err("closed ledger must reject writes"),
            LedgerError::Closed
        );
        assert_eq!(ledger.scan_after(EventSequence::INITIAL).unwrap().len(), 1);
    }
}
