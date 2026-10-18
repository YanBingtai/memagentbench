//! Provider-independent interface for pluggable memory methods.
//!
//! The trait describes the contract between the agent runtime and a memory
//! implementation. A provider may be an in-process Rust adapter or a worker
//! that forwards these requests to another language or service.

use std::{
    collections::BTreeMap,
    error::Error,
    fmt::{Display, Formatter},
    future::Future,
    pin::Pin,
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    types::{MemoryItem, MemoryKind, MemoryRelation, Modality},
    MemoryQuery, MemoryRecord, RetrieveResponse,
};

/// Boxed asynchronous result used by object-safe memory providers.
pub type MemoryFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Errors with stable categories that callers can handle or report separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    InvalidRequest(String),
    UnsupportedCapability(String),
    Unauthorized,
    NotFound(Uuid),
    Timeout,
    ResourceExhausted(String),
    Backend(String),
}

impl Display for MemoryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) => write!(formatter, "invalid memory request: {message}"),
            Self::UnsupportedCapability(capability) => {
                write!(formatter, "memory capability is unsupported: {capability}")
            }
            Self::Unauthorized => write!(formatter, "memory operation is unauthorized"),
            Self::NotFound(id) => write!(formatter, "memory item not found: {id}"),
            Self::Timeout => write!(formatter, "memory operation timed out"),
            Self::ResourceExhausted(message) => {
                write!(formatter, "memory resource exhausted: {message}")
            }
            Self::Backend(message) => write!(formatter, "memory backend error: {message}"),
        }
    }
}

impl Error for MemoryError {}

/// Capabilities exposed by one memory method.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryCapabilities {
    pub modalities: Vec<Modality>,
    pub memory_kinds: Vec<MemoryKind>,
    /// Whether `MemoryQuery::input` is interpreted as a search signal.
    #[serde(default)]
    pub supports_content_search: bool,
    /// Whether the provider can enforce `MemoryQuery::time_range`.
    #[serde(default)]
    pub supports_time_filter: bool,
    /// Whether the provider can enforce `MemoryQuery::visibility_cutoff_ms`.
    #[serde(default)]
    pub supports_visibility_cutoff: bool,
    pub supports_graph: bool,
    pub supports_cross_modal: bool,
    pub supports_update: bool,
    pub supports_delete: bool,
    pub supports_batch: bool,
    /// Whether the provider preserves complete [`MemoryRecord`] metadata on
    /// batch writes through [`MemoryMethod::ingest_records`].
    #[serde(default)]
    pub supports_record_ingest: bool,
    /// Whether the provider can return complete [`MemoryRecord`] values from
    /// [`MemoryMethod::get_record`].
    #[serde(default)]
    pub supports_record_get: bool,
    pub deterministic: bool,
    pub persistent: bool,
}

/// Identity and capability declaration for one memory method.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryManifest {
    pub method_id: String,
    pub version: String,
    pub schema_version: u32,
    pub capabilities: MemoryCapabilities,
}

/// Request-scoped identity and resource information.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationContext {
    pub namespace: String,
    pub run_id: Uuid,
    pub session_id: Option<Uuid>,
    pub episode_id: Option<Uuid>,
    /// Relative execution budget forwarded to remote providers.
    pub timeout_ms: Option<u64>,
    pub idempotency_key: Option<String>,
}

/// Items and graph relations to add to a memory method.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngestRequest {
    pub items: Vec<MemoryItem>,
    pub relations: Vec<MemoryRelation>,
    pub idempotency_key: String,
}

/// Lossless ingest request for providers that preserve episode, evidence, and
/// provenance metadata. The original [`IngestRequest`] remains available for
/// providers that only accept the older item-level contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordIngestRequest {
    pub records: Vec<MemoryRecord>,
    pub relations: Vec<MemoryRelation>,
    pub idempotency_key: String,
}

/// Durable identifiers returned after an ingest operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestReceipt {
    pub item_ids: Vec<Uuid>,
    pub relation_ids: Vec<Uuid>,
    pub deduplicated: bool,
    pub snapshot_id: Option<String>,
    /// Provider-side resource measurements for this operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<ProviderTelemetry>,
}

/// Provider-side measurements that remain useful across memory methods.
///
/// Counters and durations are intentionally named by the provider adapter so
/// new modalities can add measurements without changing the core contract.
/// Cost is stored in integer micro-dollars to keep receipts exactly
/// comparable and avoid floating-point equality in persisted traces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ProviderTelemetry {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub counters: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub durations_ms: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd_micros: Option<u64>,
}

/// The common interface implemented by every memory method adapter.
pub trait MemoryMethod: Send + Sync {
    fn manifest(&self) -> MemoryManifest;

    fn ingest<'a>(
        &'a self,
        request: IngestRequest,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>>;

    /// Ingests complete records without dropping episode, evidence, or
    /// provenance fields. Existing providers may keep the default explicit
    /// unsupported response until they implement this richer boundary.
    fn ingest_records<'a>(
        &'a self,
        _request: RecordIngestRequest,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        Box::pin(std::future::ready(Err(MemoryError::UnsupportedCapability(
            "record_ingest".to_string(),
        ))))
    }

    fn retrieve<'a>(
        &'a self,
        query: MemoryQuery,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>>;

    fn get<'a>(
        &'a self,
        item_id: Uuid,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>>;

    /// Reads a complete record without reducing it to [`MemoryItem`].
    /// Implementations that support this operation should advertise
    /// `MemoryCapabilities::supports_record_get`.
    fn get_record<'a>(
        &'a self,
        _item_id: Uuid,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryRecord, MemoryError>> {
        Box::pin(std::future::ready(Err(MemoryError::UnsupportedCapability(
            "record_get".to_string(),
        ))))
    }
}

/// Forwards the provider contract through a shared handle.
///
/// Registries intentionally return `Arc<dyn MemoryMethod>` so one provider can
/// be shared by hooks, benchmark runners, and background ingest tasks. Making
/// the handle implement the same trait keeps those consumers generic without
/// forcing every caller to unwrap or redesign its ownership model.
impl<T> MemoryMethod for Arc<T>
where
    T: MemoryMethod + ?Sized,
{
    fn manifest(&self) -> MemoryManifest {
        self.as_ref().manifest()
    }

    fn ingest<'a>(
        &'a self,
        request: IngestRequest,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        self.as_ref().ingest(request, context)
    }

    fn ingest_records<'a>(
        &'a self,
        request: RecordIngestRequest,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        self.as_ref().ingest_records(request, context)
    }

    fn retrieve<'a>(
        &'a self,
        query: MemoryQuery,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
        self.as_ref().retrieve(query, context)
    }

    fn get<'a>(
        &'a self,
        item_id: Uuid,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
        self.as_ref().get(item_id, context)
    }

    fn get_record<'a>(
        &'a self,
        item_id: Uuid,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryRecord, MemoryError>> {
        self.as_ref().get_record(item_id, context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_serializes_capabilities_for_registry_metadata() {
        let manifest = MemoryManifest {
            method_id: "reference-text".to_string(),
            version: "0.1.0".to_string(),
            schema_version: 1,
            capabilities: MemoryCapabilities {
                modalities: vec![Modality::Text],
                memory_kinds: vec![MemoryKind::Semantic],
                supports_content_search: true,
                supports_time_filter: false,
                supports_visibility_cutoff: false,
                supports_graph: false,
                supports_cross_modal: false,
                supports_update: true,
                supports_delete: true,
                supports_batch: true,
                supports_record_ingest: false,
                supports_record_get: false,
                deterministic: true,
                persistent: false,
            },
        };

        let json = serde_json::to_value(manifest).expect("manifest should serialize");

        assert_eq!(json["method_id"], "reference-text");
        assert_eq!(json["capabilities"]["modalities"][0], "text");
        assert_eq!(json["capabilities"]["deterministic"], true);
    }

    #[test]
    fn legacy_manifest_defaults_new_query_capabilities_to_false() {
        let legacy = serde_json::json!({
            "method_id": "legacy",
            "version": "1",
            "schema_version": 1,
            "capabilities": {
                "modalities": ["text"],
                "memory_kinds": ["semantic"],
                "supports_graph": false,
                "supports_cross_modal": false,
                "supports_update": false,
                "supports_delete": false,
                "supports_batch": true,
                "deterministic": true,
                "persistent": true
            }
        });
        let manifest: MemoryManifest =
            serde_json::from_value(legacy).expect("legacy manifest should remain readable");

        assert!(!manifest.capabilities.supports_content_search);
        assert!(!manifest.capabilities.supports_time_filter);
        assert!(!manifest.capabilities.supports_visibility_cutoff);
        assert!(!manifest.capabilities.supports_record_ingest);
        assert!(!manifest.capabilities.supports_record_get);
    }

    #[test]
    fn operation_context_preserves_isolation_and_retry_identity() {
        let context = OperationContext {
            namespace: "benchmark/text".to_string(),
            run_id: Uuid::new_v4(),
            session_id: None,
            episode_id: None,
            timeout_ms: Some(500),
            idempotency_key: Some("write-1".to_string()),
        };

        let json = serde_json::to_value(context).expect("context should serialize");

        assert_eq!(json["namespace"], "benchmark/text");
        assert_eq!(json["timeout_ms"], 500);
        assert_eq!(json["idempotency_key"], "write-1");
    }

    #[test]
    fn errors_keep_machine_readable_categories() {
        let error = MemoryError::UnsupportedCapability("graph_search".to_string());

        assert_eq!(
            error.to_string(),
            "memory capability is unsupported: graph_search"
        );
        assert!(matches!(
            error,
            MemoryError::UnsupportedCapability(capability) if capability == "graph_search"
        ));
    }

    #[test]
    fn record_ingest_request_round_trips_lossless_metadata() {
        let record = MemoryRecord {
            item: MemoryItem {
                id: Uuid::from_u128(7),
                version: 2,
                kind: MemoryKind::Episodic,
                scope: crate::MemoryScope::Personalized,
                modality: Modality::Video,
                content: crate::ContentRef::Text {
                    text: "抓取红色方块".to_string(),
                },
                namespace: "benchmark/robotics".to_string(),
                episode_id: Some("episode-7".to_string()),
                subject: None,
                source: "episode-summarizer".to_string(),
                metadata: std::collections::BTreeMap::new(),
                abstraction_level: 1,
                event_time: None,
                ingest_time_ms: 20_000,
                schema_version: 1,
            },
            episode: Some(crate::EpisodeRef {
                dataset_or_run_id: "robotics-v2".to_string(),
                episode_id: "episode-7".to_string(),
                task: "抓取红色方块".to_string(),
                time_range: None,
                dataset_revision: "v2".to_string(),
                source: "simulator".to_string(),
            }),
            evidence: Vec::new(),
            provenance: crate::MemoryProvenance {
                producer: "episode-summarizer".to_string(),
                source_id: "episode-7".to_string(),
                source_event_id: Some("event-101".to_string()),
                parent_record_ids: vec![Uuid::from_u128(8)],
                method_version: Some("summarizer-0.1".to_string()),
            },
        };
        let request = RecordIngestRequest {
            records: vec![record.clone()],
            relations: Vec::new(),
            idempotency_key: "episode-7-ingest".to_string(),
        };

        let encoded = serde_json::to_vec(&request).expect("record request should serialize");
        let decoded: RecordIngestRequest =
            serde_json::from_slice(&encoded).expect("record request should deserialize");

        assert_eq!(decoded.records, vec![record]);
        assert_eq!(decoded.idempotency_key, "episode-7-ingest");
    }
}
