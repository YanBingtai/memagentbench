//! Persistence contracts for canonical memory records.
//!
//! A storage backend keeps `MemoryItem` records and typed relations. It does
//! not embed content, rank results, or choose a retrieval strategy; those
//! responsibilities belong to memory methods and their optional indexes.

use std::{
    error::Error,
    fmt::{Display, Formatter},
    future::Future,
    pin::Pin,
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    query::TraversalDirection,
    types::{
        MemoryItem, MemoryKind, MemoryRelation, MemoryScope, Modality, RelationType, SubjectRef,
        TimeRange,
    },
};

/// Boxed asynchronous result used by storage backends.
pub type StorageFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Errors at the storage boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    InvalidRequest(String),
    Conflict(String),
    NotFound(Uuid),
    UnsupportedCapability(String),
    Unauthorized,
    Timeout,
    SnapshotUnavailable(String),
    Backend(String),
}

impl Display for StorageError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) => {
                write!(formatter, "invalid storage request: {message}")
            }
            Self::Conflict(message) => write!(formatter, "storage conflict: {message}"),
            Self::NotFound(id) => write!(formatter, "stored memory not found: {id}"),
            Self::UnsupportedCapability(capability) => {
                write!(formatter, "storage capability is unsupported: {capability}")
            }
            Self::Unauthorized => write!(formatter, "storage operation is unauthorized"),
            Self::Timeout => write!(formatter, "storage operation timed out"),
            Self::SnapshotUnavailable(snapshot_id) => {
                write!(formatter, "storage snapshot unavailable: {snapshot_id}")
            }
            Self::Backend(message) => write!(formatter, "storage backend error: {message}"),
        }
    }
}

impl Error for StorageError {}

/// Capabilities declared by one concrete persistence backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageCapabilities {
    pub persistent: bool,
    pub supports_versions: bool,
    pub supports_relations: bool,
    pub supports_transactions: bool,
    pub supports_snapshots: bool,
}

/// Stable identity and capability metadata for one storage backend.
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
    pub namespace: String,
    pub run_id: Uuid,
    pub snapshot_id: Option<String>,
    pub timeout_ms: Option<u64>,
}

/// An atomic append batch for canonical records and graph edges.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoreWriteRequest {
    pub items: Vec<MemoryItem>,
    pub relations: Vec<MemoryRelation>,
    pub idempotency_key: String,
}

/// Receipt returned after a successful write or an idempotent retry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreReceipt {
    pub item_ids: Vec<Uuid>,
    pub relation_ids: Vec<Uuid>,
    pub deduplicated: bool,
    pub snapshot_id: Option<String>,
}

/// Selects the latest version or one exact version of an item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreGetRequest {
    pub item_id: Uuid,
    pub version: Option<u64>,
}

/// Ordering supported by canonical record queries.
///
/// These orders are reproducible storage orders, not relevance scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreOrder {
    Id,
    IngestTimeAsc,
    IngestTimeDesc,
}

/// Structural filters that a backend can execute without a memory algorithm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoreQuery {
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
}

/// One bounded relation lookup for graph-aware memory methods.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationQuery {
    pub start_ids: Vec<Uuid>,
    pub relation_types: Vec<RelationType>,
    pub direction: TraversalDirection,
    pub limit: usize,
    pub cursor: Option<String>,
}

/// Page of canonical memory records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoreQueryResponse {
    pub items: Vec<MemoryItem>,
    pub next_cursor: Option<String>,
    pub snapshot_id: Option<String>,
}

/// Page of canonical graph relations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationQueryResponse {
    pub relations: Vec<MemoryRelation>,
    pub next_cursor: Option<String>,
    pub snapshot_id: Option<String>,
}

/// Persistence interface used by memory methods.
pub trait MemoryStore: Send + Sync {
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
    ) -> StorageFuture<'a, Result<MemoryItem, StorageError>>;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_manifest_serializes_backend_capabilities() {
        let manifest = StorageManifest {
            backend_id: "sqlite.records".to_string(),
            version: "0.1.0".to_string(),
            schema_version: 1,
            capabilities: StorageCapabilities {
                persistent: true,
                supports_versions: true,
                supports_relations: true,
                supports_transactions: true,
                supports_snapshots: false,
            },
        };

        let json = serde_json::to_value(manifest).expect("manifest should serialize");

        assert_eq!(json["backend_id"], "sqlite.records");
        assert_eq!(json["capabilities"]["supports_transactions"], true);
    }

    #[test]
    fn store_query_serializes_filters_without_semantic_input() {
        let query = StoreQuery {
            ids: vec![Uuid::new_v4()],
            scope: Some(MemoryScope::Personalized),
            subject: None,
            episode_id: None,
            kinds: vec![MemoryKind::Semantic],
            modalities: vec![Modality::Text],
            event_time: None,
            visibility_cutoff_ms: Some(50_000),
            order: StoreOrder::IngestTimeDesc,
            limit: 16,
            cursor: None,
        };

        let json = serde_json::to_value(query).expect("query should serialize");

        assert_eq!(json["order"], "ingest_time_desc");
        assert_eq!(json["limit"], 16);
        assert!(json.get("input").is_none());
    }

    #[test]
    fn storage_errors_keep_backend_categories() {
        let error = StorageError::SnapshotUnavailable("run-42".to_string());

        assert_eq!(error.to_string(), "storage snapshot unavailable: run-42");
    }
}
