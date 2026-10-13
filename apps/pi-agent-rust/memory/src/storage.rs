//! Persistent projection storage for memory records and graph relations.
//!
//! The runtime event ledger remains the source of truth. This module stores a
//! rebuildable projection that memory methods can query without embedding a
//! database or a retrieval algorithm into the agent runtime. The first local
//! backend is JSONL: each newline-terminated line is one complete write batch,
//! and the in-memory indexes are rebuilt when the file is opened. A trailing
//! fragment without a newline is discarded, even if it is valid JSON.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    error::Error,
    fmt::{self, Display, Formatter},
    fs::{self, File, OpenOptions},
    future::Future,
    io::{self, Write},
    path::{Path, PathBuf},
    pin::Pin,
    sync::Mutex,
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    MemoryKind, MemoryRecord, MemoryRelation, MemoryScope, Modality, RelationType, SubjectRef,
    TimeRange, TraversalDirection,
};

const STORAGE_SCHEMA_VERSION: u32 = 1;

/// Boxed asynchronous result used by projection-store adapters.
pub type StorageFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Errors at the projection-store boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    InvalidRequest(String),
    Conflict(String),
    NotFound(Uuid),
    UnsupportedCapability(String),
    Unauthorized,
    Timeout,
    SnapshotUnavailable(String),
    CorruptRecord { line: u64, message: String },
    LockHeld(PathBuf),
    Closed,
    WriteFailed,
    Backend(String),
}

impl Display for StorageError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(message) => {
                write!(formatter, "invalid storage request: {message}")
            }
            Self::Conflict(message) => write!(formatter, "storage conflict: {message}"),
            Self::NotFound(id) => write!(formatter, "stored memory record not found: {id}"),
            Self::UnsupportedCapability(capability) => {
                write!(formatter, "storage capability is unsupported: {capability}")
            }
            Self::Unauthorized => formatter.write_str("storage operation is unauthorized"),
            Self::Timeout => formatter.write_str("storage operation timed out"),
            Self::SnapshotUnavailable(snapshot_id) => {
                write!(formatter, "storage snapshot unavailable: {snapshot_id}")
            }
            Self::CorruptRecord { line, message } => {
                write!(
                    formatter,
                    "corrupt storage record at line {line}: {message}"
                )
            }
            Self::LockHeld(path) => {
                write!(
                    formatter,
                    "storage lock is already held: {}",
                    path.display()
                )
            }
            Self::Closed => formatter.write_str("storage handle is closed"),
            Self::WriteFailed => {
                formatter.write_str("storage handle requires reopen after a failed write")
            }
            Self::Backend(message) => write!(formatter, "storage backend error: {message}"),
        }
    }
}

impl Error for StorageError {}

/// Capabilities declared by one concrete projection backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageCapabilities {
    pub persistent: bool,
    pub supports_versions: bool,
    pub supports_relations: bool,
    /// Batch writes; this flag does not imply crash-atomic or multi-request transactions.
    pub supports_transactions: bool,
    /// Historical snapshots are not implemented by the JSONL projection yet.
    pub supports_snapshots: bool,
}

/// Stable identity and capability metadata for one projection backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageManifest {
    pub backend_id: String,
    pub version: String,
    pub schema_version: u32,
    pub capabilities: StorageCapabilities,
}

/// Request-scoped isolation and reproducibility information.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageContext {
    /// Isolation boundary. Independent benchmark runs should use distinct namespaces.
    pub namespace: String,
    /// Caller correlation metadata; this is not an additional isolation boundary.
    pub run_id: Uuid,
    /// Requires the current generation; historical snapshot reads are unsupported.
    pub snapshot_id: Option<String>,
    /// Caller budget hint. The synchronous JSONL backend cannot interrupt file I/O.
    pub timeout_ms: Option<u64>,
}

/// One append batch for memory records and graph edges.
///
/// The JSONL backend does not provide crash-atomic transactions; callers that
/// need that guarantee must use a backend whose manifest advertises it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoreWriteRequest {
    pub records: Vec<MemoryRecord>,
    pub relations: Vec<MemoryRelation>,
    pub idempotency_key: String,
    /// Cursor of the canonical event log that produced this projection batch.
    /// Equal revisions are allowed; only backwards movement is rejected.
    pub source_revision: Option<u64>,
}

/// Receipt returned after a successful write or an idempotent retry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreReceipt {
    pub record_ids: Vec<Uuid>,
    pub relation_ids: Vec<Uuid>,
    pub deduplicated: bool,
    /// Current-generation marker, not a historical snapshot handle.
    pub snapshot_id: String,
}

/// Selects the latest version or one exact version of a record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreGetRequest {
    pub record_id: Uuid,
    pub version: Option<u64>,
}

/// Ordering supported by structural projection queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreOrder {
    Id,
    IngestTimeAsc,
    IngestTimeDesc,
}

/// Structural filters that a projection backend can execute without a memory
/// algorithm or relevance score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoreQuery {
    pub namespace: String,
    pub ids: Vec<Uuid>,
    pub scope: Option<MemoryScope>,
    pub subject: Option<SubjectRef>,
    pub episode_id: Option<Uuid>,
    pub kinds: Vec<MemoryKind>,
    pub modalities: Vec<Modality>,
    pub event_time: Option<TimeRange>,
    pub visibility_cutoff_ms: Option<u64>,
    pub order: StoreOrder,
    pub limit: usize,
    pub cursor: Option<String>,
    /// Current-generation marker; historical snapshots are unsupported.
    pub snapshot_id: Option<String>,
}

/// One bounded relation lookup for graph-aware memory methods.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationQuery {
    pub start_ids: Vec<Uuid>,
    pub relation_types: Vec<RelationType>,
    pub direction: TraversalDirection,
    pub limit: usize,
    pub cursor: Option<String>,
    /// Current-generation marker; historical snapshots are unsupported.
    pub snapshot_id: Option<String>,
}

/// Page of projected memory records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoreQueryResponse {
    pub records: Vec<MemoryRecord>,
    pub next_cursor: Option<String>,
    /// Current-generation marker; historical reads are unsupported.
    pub snapshot_id: String,
}

/// Page of projected graph relations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationQueryResponse {
    pub relations: Vec<MemoryRelation>,
    pub next_cursor: Option<String>,
    /// Current-generation marker; historical reads are unsupported.
    pub snapshot_id: String,
}

/// Persistence interface used by memory methods and projection adapters.
pub trait ProjectionStore: Send + Sync {
    fn manifest(&self) -> StorageManifest;

    fn write<'a>(
        &'a self,
        request: StoreWriteRequest,
        context: StorageContext,
    ) -> StorageFuture<'a, Result<StoreReceipt, StorageError>>;

    fn get<'a>(
        &'a self,
        request: StoreGetRequest,
        context: StorageContext,
    ) -> StorageFuture<'a, Result<MemoryRecord, StorageError>>;

    fn query<'a>(
        &'a self,
        request: StoreQuery,
        context: StorageContext,
    ) -> StorageFuture<'a, Result<StoreQueryResponse, StorageError>>;

    fn relations<'a>(
        &'a self,
        request: RelationQuery,
        context: StorageContext,
    ) -> StorageFuture<'a, Result<RelationQueryResponse, StorageError>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredBatch {
    record_type: String,
    schema_version: u32,
    generation: u64,
    batch_id: Uuid,
    namespace: String,
    source_revision: Option<u64>,
    idempotency_key: String,
    request_signature: String,
    records: Vec<MemoryRecord>,
    relations: Vec<MemoryRelation>,
}

#[derive(Debug, Clone)]
struct AppliedBatch {
    signature: String,
    receipt: StoreReceipt,
}

#[derive(Debug, Default)]
struct ProjectionState {
    records: BTreeMap<Uuid, BTreeMap<u64, MemoryRecord>>,
    relations: BTreeMap<Uuid, MemoryRelation>,
    idempotency: BTreeMap<(String, String), AppliedBatch>,
    source_revisions: BTreeMap<String, u64>,
    generation: u64,
}

struct StoreInner {
    file: Option<File>,
    state: ProjectionState,
    write_failed: bool,
    closed: bool,
}

/// A persistent local projection backed by one append-only JSONL file.
pub struct JsonlProjectionStore {
    path: PathBuf,
    inner: Mutex<StoreInner>,
    _lock: LockGuard,
}

impl fmt::Debug for JsonlProjectionStore {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let state = self.inner.lock().map_err(|_| fmt::Error)?;
        formatter
            .debug_struct("JsonlProjectionStore")
            .field("path", &self.path)
            .field("generation", &state.state.generation)
            .field("closed", &state.closed)
            .field("write_failed", &state.write_failed)
            .finish()
    }
}

impl JsonlProjectionStore {
    /// Opens or creates a projection and rebuilds its indexes from JSONL.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|source| io_error("create parent", source))?;
            }
        }

        let lock = acquire_lock(&path)?;
        let mut file = match OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)
        {
            Ok(file) => file,
            Err(source) => {
                drop(lock);
                return Err(io_error("open data file", source));
            }
        };

        let state = match load_state(&path, &mut file) {
            Ok(state) => state,
            Err(error) => {
                drop(file);
                drop(lock);
                return Err(error);
            }
        };

        Ok(Self {
            path,
            inner: Mutex::new(StoreInner {
                file: Some(file),
                state,
                write_failed: false,
                closed: false,
            }),
            _lock: lock,
        })
    }

    /// Returns the JSONL data path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the current projection snapshot marker.
    pub fn snapshot_id(&self) -> Result<String, StorageError> {
        let inner = self.lock_inner()?;
        ensure_open(&inner)?;
        Ok(snapshot_id(inner.state.generation))
    }

    /// Synchronous write convenience used by local callers and tests.
    pub fn write_sync(
        &self,
        request: StoreWriteRequest,
        context: StorageContext,
    ) -> Result<StoreReceipt, StorageError> {
        validate_context(&context)?;
        if request.idempotency_key.trim().is_empty() {
            return Err(StorageError::InvalidRequest(
                "idempotency key must be non-empty".to_string(),
            ));
        }
        if request.records.is_empty() && request.relations.is_empty() {
            return Err(StorageError::InvalidRequest(
                "write batch must contain a record or relation".to_string(),
            ));
        }

        let signature = request_signature(&request, &context.namespace)?;
        let mut inner = self.lock_inner()?;
        ensure_writable(&inner)?;

        let idempotency_key = (context.namespace.clone(), request.idempotency_key.clone());
        if let Some(applied) = inner.state.idempotency.get(&idempotency_key) {
            if applied.signature == signature {
                let mut receipt = applied.receipt.clone();
                receipt.deduplicated = true;
                return Ok(receipt);
            }
            return Err(StorageError::Conflict(format!(
                "idempotency key {:?} was used with a different request",
                request.idempotency_key
            )));
        }

        validate_snapshot(&inner.state, context.snapshot_id.as_deref())?;
        validate_request(&inner.state, &request, &context.namespace)?;
        validate_source_revision(&inner.state, &context.namespace, request.source_revision)?;
        let generation =
            inner.state.generation.checked_add(1).ok_or_else(|| {
                StorageError::Backend("projection generation exhausted".to_string())
            })?;
        let batch = StoredBatch {
            record_type: "projection_batch".to_string(),
            schema_version: STORAGE_SCHEMA_VERSION,
            generation,
            batch_id: Uuid::new_v4(),
            namespace: context.namespace,
            source_revision: request.source_revision,
            idempotency_key: request.idempotency_key,
            request_signature: signature,
            records: request.records,
            relations: request.relations,
        };

        let mut line = serde_json::to_vec(&batch)
            .map_err(|source| StorageError::Backend(format!("serialize write batch: {source}")))?;
        line.push(b'\n');
        let file = inner.file.as_mut().ok_or(StorageError::Closed)?;
        if let Err(source) = file.write_all(&line) {
            inner.write_failed = true;
            return Err(io_error("append projection batch", source));
        }
        if let Err(source) = file.flush() {
            inner.write_failed = true;
            return Err(io_error("flush projection batch", source));
        }
        if let Err(source) = file.sync_data() {
            inner.write_failed = true;
            return Err(io_error("sync projection batch", source));
        }

        apply_batch(&mut inner.state, &batch)
    }

    /// Synchronous lookup convenience.
    pub fn get_sync(
        &self,
        request: StoreGetRequest,
        context: StorageContext,
    ) -> Result<MemoryRecord, StorageError> {
        validate_context(&context)?;
        let inner = self.lock_inner()?;
        ensure_open(&inner)?;
        validate_snapshot(&inner.state, context.snapshot_id.as_deref())?;
        let versions = inner
            .state
            .records
            .get(&request.record_id)
            .ok_or(StorageError::NotFound(request.record_id))?;
        let record = match request.version {
            Some(version) => versions.get(&version),
            None => versions.values().next_back(),
        }
        .ok_or(StorageError::NotFound(request.record_id))?;
        if record.item.namespace != context.namespace {
            return Err(StorageError::NotFound(request.record_id));
        }
        Ok(record.clone())
    }

    /// Synchronous structural query convenience.
    pub fn query_sync(
        &self,
        request: StoreQuery,
        context: StorageContext,
    ) -> Result<StoreQueryResponse, StorageError> {
        validate_context(&context)?;
        if request.limit == 0 {
            return Err(StorageError::InvalidRequest(
                "query limit must be greater than zero".to_string(),
            ));
        }
        if request.namespace.trim().is_empty() {
            return Err(StorageError::InvalidRequest(
                "query namespace must be non-empty".to_string(),
            ));
        }
        if request.namespace != context.namespace {
            return Err(StorageError::InvalidRequest(
                "query namespace must match storage context".to_string(),
            ));
        }

        let inner = self.lock_inner()?;
        ensure_open(&inner)?;
        validate_snapshot(&inner.state, context.snapshot_id.as_deref())?;
        validate_snapshot(&inner.state, request.snapshot_id.as_deref())?;
        if let (Some(context_snapshot), Some(query_snapshot)) = (
            context.snapshot_id.as_deref(),
            request.snapshot_id.as_deref(),
        ) {
            if context_snapshot != query_snapshot {
                return Err(StorageError::SnapshotUnavailable(
                    query_snapshot.to_string(),
                ));
            }
        }

        let snapshot = snapshot_id(inner.state.generation);
        let binding = query_cursor_binding(&request)?;
        let offset = decode_cursor(request.cursor.as_deref(), &snapshot, &binding)?;
        let mut records = latest_visible_records(&inner.state, &request);
        records.sort_by(|left, right| match request.order {
            StoreOrder::Id => left.item.id.cmp(&right.item.id),
            StoreOrder::IngestTimeAsc => left
                .item
                .ingest_time_ms
                .cmp(&right.item.ingest_time_ms)
                .then_with(|| left.item.id.cmp(&right.item.id)),
            StoreOrder::IngestTimeDesc => right
                .item
                .ingest_time_ms
                .cmp(&left.item.ingest_time_ms)
                .then_with(|| left.item.id.cmp(&right.item.id)),
        });

        let page = page(records, offset, request.limit)?;
        let next_cursor = page
            .next_offset
            .map(|next| encode_cursor(&snapshot, &binding, next));
        Ok(StoreQueryResponse {
            records: page.values,
            next_cursor,
            snapshot_id: snapshot,
        })
    }

    /// Synchronous relation query convenience.
    pub fn relations_sync(
        &self,
        request: RelationQuery,
        context: StorageContext,
    ) -> Result<RelationQueryResponse, StorageError> {
        validate_context(&context)?;
        if request.limit == 0 {
            return Err(StorageError::InvalidRequest(
                "relation query limit must be greater than zero".to_string(),
            ));
        }

        let inner = self.lock_inner()?;
        ensure_open(&inner)?;
        validate_snapshot(&inner.state, context.snapshot_id.as_deref())?;
        validate_snapshot(&inner.state, request.snapshot_id.as_deref())?;
        let snapshot = snapshot_id(inner.state.generation);
        let binding = relation_cursor_binding(&request, &context.namespace)?;
        let offset = decode_cursor(request.cursor.as_deref(), &snapshot, &binding)?;
        let visible_ids = visible_record_ids(&inner.state, &context.namespace);
        let mut relations = inner
            .state
            .relations
            .values()
            .filter(|relation| {
                visible_ids.contains(&relation.from) && visible_ids.contains(&relation.to)
            })
            .filter(|relation| {
                request.start_ids.is_empty()
                    || request
                        .start_ids
                        .iter()
                        .any(|start| match request.direction {
                            TraversalDirection::Outgoing => relation.from == *start,
                            TraversalDirection::Incoming => relation.to == *start,
                            TraversalDirection::Both => {
                                relation.from == *start || relation.to == *start
                            }
                        })
            })
            .filter(|relation| {
                request.relation_types.is_empty()
                    || request
                        .relation_types
                        .iter()
                        .any(|relation_type| relation_type == &relation.relation_type)
            })
            .cloned()
            .collect::<Vec<_>>();
        relations.sort_by_key(|relation| relation.id);

        let page = page(relations, offset, request.limit)?;
        let next_cursor = page
            .next_offset
            .map(|next| encode_cursor(&snapshot, &binding, next));
        Ok(RelationQueryResponse {
            relations: page.values,
            next_cursor,
            snapshot_id: snapshot,
        })
    }

    fn lock_inner(&self) -> Result<std::sync::MutexGuard<'_, StoreInner>, StorageError> {
        self.inner
            .lock()
            .map_err(|_| StorageError::Backend("projection state lock poisoned".to_string()))
    }

    /// Flushes and closes this handle. The lock is released when the handle is
    /// dropped after this call.
    pub fn close(&mut self) -> Result<(), StorageError> {
        let mut inner = self.lock_inner()?;
        if inner.closed {
            return Ok(());
        }
        if let Some(file) = inner.file.as_ref() {
            file.sync_all()
                .map_err(|source| io_error("close projection", source))?;
        }
        inner.file.take();
        inner.closed = true;
        Ok(())
    }
}

impl ProjectionStore for JsonlProjectionStore {
    fn manifest(&self) -> StorageManifest {
        StorageManifest {
            backend_id: "jsonl-projection".to_string(),
            version: "0.1.0".to_string(),
            schema_version: STORAGE_SCHEMA_VERSION,
            capabilities: StorageCapabilities {
                persistent: true,
                supports_versions: true,
                supports_relations: true,
                supports_transactions: false,
                supports_snapshots: false,
            },
        }
    }

    fn write<'a>(
        &'a self,
        request: StoreWriteRequest,
        context: StorageContext,
    ) -> StorageFuture<'a, Result<StoreReceipt, StorageError>> {
        Box::pin(async move { self.write_sync(request, context) })
    }

    fn get<'a>(
        &'a self,
        request: StoreGetRequest,
        context: StorageContext,
    ) -> StorageFuture<'a, Result<MemoryRecord, StorageError>> {
        Box::pin(async move { self.get_sync(request, context) })
    }

    fn query<'a>(
        &'a self,
        request: StoreQuery,
        context: StorageContext,
    ) -> StorageFuture<'a, Result<StoreQueryResponse, StorageError>> {
        Box::pin(async move { self.query_sync(request, context) })
    }

    fn relations<'a>(
        &'a self,
        request: RelationQuery,
        context: StorageContext,
    ) -> StorageFuture<'a, Result<RelationQueryResponse, StorageError>> {
        Box::pin(async move { self.relations_sync(request, context) })
    }
}

impl Drop for JsonlProjectionStore {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.file.take();
            inner.closed = true;
        }
    }
}

#[derive(Debug, Serialize)]
struct WritePayload<'a> {
    namespace: &'a str,
    records: &'a [MemoryRecord],
    relations: &'a [MemoryRelation],
    source_revision: Option<u64>,
}

fn request_signature(request: &StoreWriteRequest, namespace: &str) -> Result<String, StorageError> {
    serde_json::to_string(&WritePayload {
        namespace,
        records: &request.records,
        relations: &request.relations,
        source_revision: request.source_revision,
    })
    .map_err(|source| StorageError::Backend(format!("serialize request signature: {source}")))
}

fn validate_context(context: &StorageContext) -> Result<(), StorageError> {
    if context.namespace.trim().is_empty() {
        return Err(StorageError::InvalidRequest(
            "storage namespace must be non-empty".to_string(),
        ));
    }
    if context.timeout_ms == Some(0) {
        return Err(StorageError::Timeout);
    }
    Ok(())
}

fn validate_request(
    state: &ProjectionState,
    request: &StoreWriteRequest,
    namespace: &str,
) -> Result<(), StorageError> {
    let mut seen_versions = HashMap::new();
    let mut available_ids = state.records.keys().copied().collect::<HashSet<_>>();
    let mut namespaces = HashMap::new();
    for (id, versions) in &state.records {
        if let Some(record) = versions.values().next_back() {
            namespaces.insert(*id, record.item.namespace.clone());
        }
    }

    for record in &request.records {
        if record.item.namespace != namespace {
            return Err(StorageError::InvalidRequest(format!(
                "record {} belongs to namespace {:?}, expected {:?}",
                record.item.id, record.item.namespace, namespace
            )));
        }
        if record.item.version == 0 {
            return Err(StorageError::InvalidRequest(format!(
                "record {} must have a non-zero version",
                record.item.id
            )));
        }
        let previous = state
            .records
            .get(&record.item.id)
            .and_then(|versions| versions.keys().next_back().copied());
        if let Some(existing) = state
            .records
            .get(&record.item.id)
            .and_then(|versions| versions.values().next_back())
        {
            if existing.item.namespace != namespace {
                return Err(StorageError::Conflict(format!(
                    "record {} cannot move from namespace {:?} to {:?}",
                    record.item.id, existing.item.namespace, namespace
                )));
            }
        }
        if seen_versions
            .insert(record.item.id, record.item.version)
            .is_some()
        {
            return Err(StorageError::Conflict(format!(
                "write batch contains duplicate record id {}",
                record.item.id
            )));
        }
        if previous.is_some_and(|version| version >= record.item.version) {
            return Err(StorageError::Conflict(format!(
                "record {} version {} is not newer than stored version {}",
                record.item.id,
                record.item.version,
                previous.expect("checked above")
            )));
        }
        available_ids.insert(record.item.id);
        namespaces.insert(record.item.id, record.item.namespace.clone());
    }

    let mut relation_ids = HashSet::new();
    for relation in &request.relations {
        if !relation_ids.insert(relation.id) || state.relations.contains_key(&relation.id) {
            return Err(StorageError::Conflict(format!(
                "relation {} already exists",
                relation.id
            )));
        }
        if !available_ids.contains(&relation.from) || !available_ids.contains(&relation.to) {
            return Err(StorageError::InvalidRequest(format!(
                "relation {} references a missing record",
                relation.id
            )));
        }
        if namespaces.get(&relation.from).map(String::as_str) != Some(namespace)
            || namespaces.get(&relation.to).map(String::as_str) != Some(namespace)
        {
            return Err(StorageError::InvalidRequest(format!(
                "relation {} crosses storage namespaces",
                relation.id
            )));
        }
    }
    Ok(())
}

fn validate_source_revision(
    state: &ProjectionState,
    namespace: &str,
    source_revision: Option<u64>,
) -> Result<(), StorageError> {
    if let Some(source_revision) = source_revision {
        if state
            .source_revisions
            .get(namespace)
            .is_some_and(|previous| source_revision < *previous)
        {
            return Err(StorageError::Conflict(format!(
                "source revision {} is older than the stored revision for namespace {:?}",
                source_revision, namespace
            )));
        }
    }
    Ok(())
}

fn apply_batch(
    state: &mut ProjectionState,
    batch: &StoredBatch,
) -> Result<StoreReceipt, StorageError> {
    if batch.record_type != "projection_batch" {
        return Err(StorageError::InvalidRequest(format!(
            "unknown projection record type {:?}",
            batch.record_type
        )));
    }
    if batch.schema_version != STORAGE_SCHEMA_VERSION {
        return Err(StorageError::UnsupportedCapability(format!(
            "projection schema version {}",
            batch.schema_version
        )));
    }
    if batch.generation != state.generation.saturating_add(1) {
        return Err(StorageError::Conflict(format!(
            "projection generation must advance from {} to {}, found {}",
            state.generation,
            state.generation.saturating_add(1),
            batch.generation
        )));
    }
    if batch.namespace.trim().is_empty() {
        return Err(StorageError::InvalidRequest(
            "persisted namespace must be non-empty".to_string(),
        ));
    }
    if batch.idempotency_key.trim().is_empty() {
        return Err(StorageError::InvalidRequest(
            "persisted idempotency key must be non-empty".to_string(),
        ));
    }
    let idempotency_key = (batch.namespace.clone(), batch.idempotency_key.clone());
    if state.idempotency.contains_key(&idempotency_key) {
        return Err(StorageError::Conflict(format!(
            "duplicate persisted idempotency key {:?}",
            batch.idempotency_key
        )));
    }

    let request = StoreWriteRequest {
        records: batch.records.clone(),
        relations: batch.relations.clone(),
        idempotency_key: batch.idempotency_key.clone(),
        source_revision: batch.source_revision,
    };
    let signature = request_signature(&request, &batch.namespace)?;
    if signature != batch.request_signature {
        return Err(StorageError::Conflict(
            "persisted request signature does not match its payload".to_string(),
        ));
    }
    validate_request(state, &request, &batch.namespace)?;
    validate_source_revision(state, &batch.namespace, batch.source_revision)?;

    for record in &batch.records {
        state
            .records
            .entry(record.item.id)
            .or_default()
            .insert(record.item.version, record.clone());
    }
    for relation in &batch.relations {
        state.relations.insert(relation.id, relation.clone());
    }
    state.generation = batch.generation;
    if let Some(source_revision) = batch.source_revision {
        state
            .source_revisions
            .insert(batch.namespace.clone(), source_revision);
    }
    let receipt = StoreReceipt {
        record_ids: batch.records.iter().map(|record| record.item.id).collect(),
        relation_ids: batch.relations.iter().map(|relation| relation.id).collect(),
        deduplicated: false,
        snapshot_id: snapshot_id(batch.generation),
    };
    state.idempotency.insert(
        idempotency_key,
        AppliedBatch {
            signature,
            receipt: receipt.clone(),
        },
    );
    Ok(receipt)
}

fn latest_visible_records(state: &ProjectionState, query: &StoreQuery) -> Vec<MemoryRecord> {
    state
        .records
        .values()
        .filter_map(|versions| {
            let latest_visible = versions.values().rev().find(|record| {
                query
                    .visibility_cutoff_ms
                    .is_none_or(|cutoff| record.item.ingest_time_ms <= cutoff)
            })?;
            matches_query_attributes(latest_visible, query).then(|| latest_visible.clone())
        })
        .collect()
}

fn matches_query_attributes(record: &MemoryRecord, query: &StoreQuery) -> bool {
    let item = &record.item;
    (query.ids.is_empty() || query.ids.contains(&item.id))
        && query
            .scope
            .as_ref()
            .is_none_or(|scope| scope == &item.scope)
        && query
            .subject
            .as_ref()
            .is_none_or(|subject| item.subject.as_ref() == Some(subject))
        && (query.episode_id.is_none() || query.episode_id == item.episode_id)
        && (query.kinds.is_empty() || query.kinds.contains(&item.kind))
        && (query.modalities.is_empty() || query.modalities.contains(&item.modality))
        && query
            .event_time
            .as_ref()
            .is_none_or(|requested| time_ranges_overlap(item.event_time.as_ref(), requested))
        && item.namespace == query.namespace
}

fn time_ranges_overlap(candidate: Option<&TimeRange>, requested: &TimeRange) -> bool {
    let Some(candidate) = candidate else {
        return false;
    };
    if candidate.clock_domain != requested.clock_domain {
        return false;
    }
    let candidate_end = candidate.end_ms.unwrap_or(i64::MAX);
    let requested_end = requested.end_ms.unwrap_or(i64::MAX);
    candidate.start_ms <= requested_end && requested.start_ms <= candidate_end
}

fn visible_record_ids(state: &ProjectionState, namespace: &str) -> HashSet<Uuid> {
    state
        .records
        .iter()
        .filter_map(|(id, versions)| {
            versions
                .values()
                .next_back()
                .filter(|record| record.item.namespace == namespace)
                .map(|_| *id)
        })
        .collect()
}

struct Page<T> {
    values: Vec<T>,
    next_offset: Option<usize>,
}

fn page<T>(values: Vec<T>, offset: usize, limit: usize) -> Result<Page<T>, StorageError> {
    let total = values.len();
    if offset > total {
        return Err(StorageError::InvalidRequest(
            "storage cursor is past the end of the result set".to_string(),
        ));
    }
    let end = offset.saturating_add(limit).min(total);
    Ok(Page {
        values: values.into_iter().skip(offset).take(end - offset).collect(),
        next_offset: (end < total).then_some(end),
    })
}

#[derive(Debug, Serialize, Deserialize)]
struct QueryCursor {
    snapshot_id: String,
    binding: String,
    offset: usize,
}

fn encode_cursor(snapshot_id: &str, binding: &str, offset: usize) -> String {
    serde_json::to_string(&QueryCursor {
        snapshot_id: snapshot_id.to_string(),
        binding: binding.to_string(),
        offset,
    })
    .expect("query cursor serialization cannot fail")
}

fn decode_cursor(
    cursor: Option<&str>,
    snapshot_id: &str,
    binding: &str,
) -> Result<usize, StorageError> {
    let Some(cursor) = cursor else {
        return Ok(0);
    };
    let cursor: QueryCursor = serde_json::from_str(cursor).map_err(|source| {
        StorageError::InvalidRequest(format!("invalid storage cursor: {source}"))
    })?;
    if cursor.snapshot_id != snapshot_id {
        return Err(StorageError::SnapshotUnavailable(cursor.snapshot_id));
    }
    if cursor.binding != binding {
        return Err(StorageError::InvalidRequest(
            "storage cursor does not match the query".to_string(),
        ));
    }
    Ok(cursor.offset)
}

fn query_cursor_binding(request: &StoreQuery) -> Result<String, StorageError> {
    let mut request = request.clone();
    request.cursor = None;
    request.snapshot_id = None;
    serde_json::to_string(&request)
        .map_err(|source| StorageError::Backend(format!("serialize query cursor: {source}")))
}

fn relation_cursor_binding(
    request: &RelationQuery,
    namespace: &str,
) -> Result<String, StorageError> {
    let mut request = request.clone();
    request.cursor = None;
    request.snapshot_id = None;
    serde_json::to_string(&(namespace, request))
        .map_err(|source| StorageError::Backend(format!("serialize relation cursor: {source}")))
}

fn validate_snapshot(state: &ProjectionState, requested: Option<&str>) -> Result<(), StorageError> {
    if let Some(requested) = requested {
        let current = snapshot_id(state.generation);
        if requested != current {
            return Err(StorageError::SnapshotUnavailable(requested.to_string()));
        }
    }
    Ok(())
}

fn snapshot_id(generation: u64) -> String {
    format!("projection-{generation}")
}

fn ensure_open(inner: &StoreInner) -> Result<(), StorageError> {
    if inner.closed || inner.file.is_none() {
        Err(StorageError::Closed)
    } else {
        Ok(())
    }
}

fn ensure_writable(inner: &StoreInner) -> Result<(), StorageError> {
    ensure_open(inner)?;
    if inner.write_failed {
        return Err(StorageError::WriteFailed);
    }
    Ok(())
}

fn io_error(operation: &str, source: io::Error) -> StorageError {
    StorageError::Backend(format!("{operation} failed: {source}"))
}

struct LockGuard {
    path: PathBuf,
    _file: File,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn lock_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.lock", path.display()))
}

fn acquire_lock(path: &Path) -> Result<LockGuard, StorageError> {
    let path = lock_path(path);
    let mut file = match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
            return Err(StorageError::LockHeld(path));
        }
        Err(source) => return Err(io_error("create storage lock", source)),
    };
    if let Err(source) = writeln!(file, "pid={}", std::process::id()) {
        let _ = fs::remove_file(&path);
        return Err(io_error("write storage lock", source));
    }
    Ok(LockGuard { path, _file: file })
}

fn load_state(path: &Path, file: &mut File) -> Result<ProjectionState, StorageError> {
    let data = fs::read(path).map_err(|source| io_error("read projection", source))?;
    let mut state = ProjectionState::default();
    let mut offset = 0_u64;

    for (line_index, chunk) in data.split_inclusive(|byte| *byte == b'\n').enumerate() {
        let line_number = line_index as u64 + 1;
        let has_newline = chunk.ends_with(b"\n");
        if !has_newline {
            file.set_len(offset)
                .map_err(|source| io_error("truncate incomplete projection", source))?;
            file.sync_data()
                .map_err(|source| io_error("sync truncated projection", source))?;
            break;
        }
        let mut line = chunk;
        if has_newline {
            line = &line[..line.len() - 1];
        }
        if line.ends_with(b"\r") {
            line = &line[..line.len() - 1];
        }

        if line.is_empty() {
            offset += chunk.len() as u64;
            continue;
        }

        match serde_json::from_slice::<StoredBatch>(line) {
            Ok(batch) => {
                apply_batch(&mut state, &batch).map_err(|error| StorageError::CorruptRecord {
                    line: line_number,
                    message: error.to_string(),
                })?;
            }
            Err(source) => {
                return Err(StorageError::CorruptRecord {
                    line: line_number,
                    message: source.to_string(),
                });
            }
        }
        offset += chunk.len() as u64;
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentRef, MemoryItem, MemoryProvenance};
    use std::collections::BTreeMap;

    fn path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("pi-agent-memory-{name}-{}.jsonl", Uuid::new_v4()))
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(lock_path(path));
    }

    fn context(namespace: &str) -> StorageContext {
        StorageContext {
            namespace: namespace.to_string(),
            run_id: Uuid::new_v4(),
            snapshot_id: None,
            timeout_ms: Some(500),
        }
    }

    fn record(id: u128, version: u64, namespace: &str, ingest_time_ms: u64) -> MemoryRecord {
        MemoryRecord {
            item: MemoryItem {
                id: Uuid::from_u128(id),
                version,
                kind: MemoryKind::Semantic,
                scope: MemoryScope::Personalized,
                modality: Modality::Text,
                content: ContentRef::Text {
                    text: format!("memory-{id}-v{version}"),
                },
                namespace: namespace.to_string(),
                episode_id: None,
                subject: None,
                source: "test".to_string(),
                metadata: BTreeMap::new(),
                abstraction_level: 0,
                event_time: None,
                ingest_time_ms,
                schema_version: 1,
            },
            episode: None,
            evidence: Vec::new(),
            provenance: MemoryProvenance {
                producer: "test".to_string(),
                source_id: "test".to_string(),
                source_event_id: None,
                parent_record_ids: Vec::new(),
                method_version: Some("test-1".to_string()),
            },
        }
    }

    fn write_request(
        key: &str,
        records: Vec<MemoryRecord>,
        relations: Vec<MemoryRelation>,
    ) -> StoreWriteRequest {
        StoreWriteRequest {
            records,
            relations,
            idempotency_key: key.to_string(),
            source_revision: Some(7),
        }
    }

    #[test]
    fn writes_reopens_queries_and_deduplicates_retries() {
        let file_path = path("replay");
        let store = JsonlProjectionStore::open(&file_path).expect("store should open");
        let request = write_request("batch-1", vec![record(1, 1, "tenant-a", 10)], Vec::new());
        let first = store
            .write_sync(request.clone(), context("tenant-a"))
            .expect("write should succeed");
        assert!(!first.deduplicated);

        let retry = store
            .write_sync(request, context("tenant-a"))
            .expect("retry should succeed");
        assert!(retry.deduplicated);
        assert_eq!(retry.snapshot_id, first.snapshot_id);

        let response = store
            .query_sync(
                StoreQuery {
                    namespace: "tenant-a".to_string(),
                    ids: Vec::new(),
                    scope: Some(MemoryScope::Personalized),
                    subject: None,
                    episode_id: None,
                    kinds: vec![MemoryKind::Semantic],
                    modalities: vec![Modality::Text],
                    event_time: None,
                    visibility_cutoff_ms: None,
                    order: StoreOrder::Id,
                    limit: 8,
                    cursor: None,
                    snapshot_id: Some(first.snapshot_id.clone()),
                },
                context("tenant-a"),
            )
            .expect("query should succeed");
        assert_eq!(response.records.len(), 1);
        assert_eq!(response.records[0].item.id, Uuid::from_u128(1));

        drop(store);
        let reopened = JsonlProjectionStore::open(&file_path).expect("store should reopen");
        let loaded = reopened
            .get_sync(
                StoreGetRequest {
                    record_id: Uuid::from_u128(1),
                    version: None,
                },
                context("tenant-a"),
            )
            .expect("record should be replayed");
        assert_eq!(loaded.item.version, 1);
        drop(reopened);
        cleanup(&file_path);
    }

    #[test]
    fn latest_query_respects_version_and_visibility_cutoff() {
        let file_path = path("versions");
        let store = JsonlProjectionStore::open(&file_path).expect("store should open");
        store
            .write_sync(
                write_request("v1", vec![record(2, 1, "tenant-a", 10)], Vec::new()),
                context("tenant-a"),
            )
            .expect("v1 should write");
        store
            .write_sync(
                write_request("v2", vec![record(2, 2, "tenant-a", 20)], Vec::new()),
                context("tenant-a"),
            )
            .expect("v2 should write");

        let response = store
            .query_sync(
                StoreQuery {
                    namespace: "tenant-a".to_string(),
                    ids: vec![Uuid::from_u128(2)],
                    scope: None,
                    subject: None,
                    episode_id: None,
                    kinds: Vec::new(),
                    modalities: Vec::new(),
                    event_time: None,
                    visibility_cutoff_ms: Some(15),
                    order: StoreOrder::Id,
                    limit: 1,
                    cursor: None,
                    snapshot_id: None,
                },
                context("tenant-a"),
            )
            .expect("cutoff query should succeed");
        assert_eq!(response.records[0].item.version, 1);
        drop(store);
        cleanup(&file_path);
    }

    #[test]
    fn latest_version_is_filtered_after_visibility_selection() {
        let file_path = path("latest-filter");
        let store = JsonlProjectionStore::open(&file_path).expect("store should open");
        store
            .write_sync(
                write_request("filter-v1", vec![record(20, 1, "tenant-a", 10)], Vec::new()),
                context("tenant-a"),
            )
            .expect("v1 should write");
        let mut newer = record(20, 2, "tenant-a", 20);
        newer.item.modality = Modality::Image;
        store
            .write_sync(
                write_request("filter-v2", vec![newer], Vec::new()),
                context("tenant-a"),
            )
            .expect("v2 should write");

        let response = store
            .query_sync(
                StoreQuery {
                    namespace: "tenant-a".to_string(),
                    ids: vec![Uuid::from_u128(20)],
                    scope: None,
                    subject: None,
                    episode_id: None,
                    kinds: Vec::new(),
                    modalities: vec![Modality::Text],
                    event_time: None,
                    visibility_cutoff_ms: None,
                    order: StoreOrder::Id,
                    limit: 1,
                    cursor: None,
                    snapshot_id: None,
                },
                context("tenant-a"),
            )
            .expect("filtered query should succeed");
        assert!(response.records.is_empty());
        drop(store);
        cleanup(&file_path);
    }

    #[test]
    fn idempotency_and_source_revision_are_namespace_scoped() {
        let file_path = path("isolation");
        let store = JsonlProjectionStore::open(&file_path).expect("store should open");
        store
            .write_sync(
                write_request("same-key", vec![record(21, 1, "tenant-a", 1)], Vec::new()),
                context("tenant-a"),
            )
            .expect("tenant-a should write");
        store
            .write_sync(
                write_request("same-key", vec![record(22, 1, "tenant-b", 1)], Vec::new()),
                context("tenant-b"),
            )
            .expect("tenant-b should not reuse tenant-a idempotency state");

        let mut stale = write_request(
            "stale-revision",
            vec![record(23, 1, "tenant-a", 2)],
            Vec::new(),
        );
        stale.source_revision = Some(6);
        let error = store
            .write_sync(stale, context("tenant-a"))
            .expect_err("source revisions must not move backwards");
        assert!(
            matches!(error, StorageError::Conflict(message) if message.contains("source revision"))
        );

        let mut current = write_request(
            "current-revision",
            vec![record(23, 1, "tenant-a", 2)],
            Vec::new(),
        );
        current.source_revision = Some(8);
        store
            .write_sync(current, context("tenant-a"))
            .expect("a newer source revision should still write");
        drop(store);
        cleanup(&file_path);
    }

    #[test]
    fn cursors_are_bound_to_their_query() {
        let file_path = path("cursor-binding");
        let store = JsonlProjectionStore::open(&file_path).expect("store should open");
        store
            .write_sync(
                write_request(
                    "cursor-batch",
                    vec![record(30, 1, "tenant-a", 1), record(31, 1, "tenant-a", 2)],
                    Vec::new(),
                ),
                context("tenant-a"),
            )
            .expect("records should write");
        let first = store
            .query_sync(
                StoreQuery {
                    namespace: "tenant-a".to_string(),
                    ids: Vec::new(),
                    scope: None,
                    subject: None,
                    episode_id: None,
                    kinds: Vec::new(),
                    modalities: Vec::new(),
                    event_time: None,
                    visibility_cutoff_ms: None,
                    order: StoreOrder::Id,
                    limit: 1,
                    cursor: None,
                    snapshot_id: None,
                },
                context("tenant-a"),
            )
            .expect("first page should succeed");
        let error = store
            .query_sync(
                StoreQuery {
                    namespace: "tenant-a".to_string(),
                    ids: vec![Uuid::from_u128(31)],
                    scope: None,
                    subject: None,
                    episode_id: None,
                    kinds: Vec::new(),
                    modalities: Vec::new(),
                    event_time: None,
                    visibility_cutoff_ms: None,
                    order: StoreOrder::Id,
                    limit: 1,
                    cursor: first.next_cursor,
                    snapshot_id: None,
                },
                context("tenant-a"),
            )
            .expect_err("a cursor cannot be reused for another query");
        assert!(
            matches!(error, StorageError::InvalidRequest(message) if message.contains("does not match"))
        );
        drop(store);
        cleanup(&file_path);
    }

    #[test]
    fn relations_are_namespace_scoped_and_directional() {
        let file_path = path("relations");
        let store = JsonlProjectionStore::open(&file_path).expect("store should open");
        let relation = MemoryRelation {
            id: Uuid::from_u128(100),
            from: Uuid::from_u128(3),
            to: Uuid::from_u128(4),
            relation_type: RelationType::DerivedFrom,
            weight: None,
            confidence: Some(0.9),
            valid_time: None,
            source: "test".to_string(),
            metadata: BTreeMap::new(),
            schema_version: 1,
        };
        store
            .write_sync(
                write_request(
                    "graph-1",
                    vec![record(3, 1, "tenant-a", 1), record(4, 1, "tenant-a", 1)],
                    vec![relation],
                ),
                context("tenant-a"),
            )
            .expect("graph batch should write");

        let response = store
            .relations_sync(
                RelationQuery {
                    start_ids: vec![Uuid::from_u128(3)],
                    relation_types: vec![RelationType::DerivedFrom],
                    direction: TraversalDirection::Outgoing,
                    limit: 8,
                    cursor: None,
                    snapshot_id: None,
                },
                context("tenant-a"),
            )
            .expect("relation query should succeed");
        assert_eq!(response.relations.len(), 1);
        assert_eq!(response.relations[0].to, Uuid::from_u128(4));
        drop(store);
        cleanup(&file_path);
    }

    #[test]
    fn a_record_cannot_move_between_namespaces() {
        let file_path = path("namespace-move");
        let store = JsonlProjectionStore::open(&file_path).expect("store should open");
        store
            .write_sync(
                write_request("tenant-a-v1", vec![record(5, 1, "tenant-a", 1)], Vec::new()),
                context("tenant-a"),
            )
            .expect("initial record should write");

        let error = store
            .write_sync(
                write_request("tenant-b-v2", vec![record(5, 2, "tenant-b", 2)], Vec::new()),
                context("tenant-b"),
            )
            .expect_err("record namespace changes must be rejected");
        assert!(
            matches!(error, StorageError::Conflict(message) if message.contains("cannot move"))
        );
        drop(store);
        cleanup(&file_path);
    }

    #[test]
    fn lock_and_incomplete_tail_are_handled_explicitly() {
        let file_path = path("recovery");
        let first = JsonlProjectionStore::open(&file_path).expect("first store should open");
        let error = JsonlProjectionStore::open(&file_path).expect_err("lock should be held");
        assert!(matches!(error, StorageError::LockHeld(_)));
        drop(first);

        fs::write(&file_path, b"{\"record_type\":").expect("partial tail should write");
        let reopened = JsonlProjectionStore::open(&file_path).expect("tail should be repaired");
        assert_eq!(reopened.snapshot_id().unwrap(), "projection-0");
        drop(reopened);
        cleanup(&file_path);

        let valid_tail_path = path("valid-tail");
        let valid = JsonlProjectionStore::open(&valid_tail_path).expect("store should open");
        valid
            .write_sync(
                write_request(
                    "valid-tail-batch",
                    vec![record(40, 1, "tenant-a", 1)],
                    Vec::new(),
                ),
                context("tenant-a"),
            )
            .expect("valid batch should write");
        drop(valid);
        let mut bytes = fs::read(&valid_tail_path).expect("valid batch should be readable");
        assert_eq!(bytes.pop(), Some(b'\n'));
        fs::write(&valid_tail_path, bytes).expect("newline should be removed");
        let reopened =
            JsonlProjectionStore::open(&valid_tail_path).expect("valid fragment should truncate");
        assert_eq!(reopened.snapshot_id().unwrap(), "projection-0");
        assert!(matches!(
            reopened.get_sync(
                StoreGetRequest {
                    record_id: Uuid::from_u128(40),
                    version: None,
                },
                context("tenant-a")
            ),
            Err(StorageError::NotFound(_))
        ));
        drop(reopened);
        cleanup(&valid_tail_path);
    }
}
