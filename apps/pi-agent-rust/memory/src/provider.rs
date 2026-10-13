//! Provider-independent interface for pluggable memory methods.
//!
//! The trait describes the contract between the agent runtime and a memory
//! implementation. A provider may be an in-process Rust adapter or a worker
//! that forwards these requests to another language or service.

use std::{
    error::Error,
    fmt::{Display, Formatter},
    future::Future,
    pin::Pin,
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    types::{MemoryItem, MemoryKind, MemoryRelation, Modality},
    MemoryQuery, RetrieveResponse,
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
    pub supports_graph: bool,
    pub supports_cross_modal: bool,
    pub supports_update: bool,
    pub supports_delete: bool,
    pub supports_batch: bool,
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

/// Durable identifiers returned after an ingest operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestReceipt {
    pub item_ids: Vec<Uuid>,
    pub relation_ids: Vec<Uuid>,
    pub deduplicated: bool,
    pub snapshot_id: Option<String>,
}

/// The common interface implemented by every memory method adapter.
pub trait MemoryMethod: Send + Sync {
    fn manifest(&self) -> MemoryManifest;

    fn ingest<'a>(
        &'a self,
        request: IngestRequest,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>>;

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
                supports_graph: false,
                supports_cross_modal: false,
                supports_update: true,
                supports_delete: true,
                supports_batch: true,
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
}
