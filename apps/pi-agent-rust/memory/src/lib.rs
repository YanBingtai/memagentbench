//! Public entry point for the pluggable agent-memory crate.
//!
//! The crate exposes domain types first. Storage backends, retrieval methods,
//! and benchmark adapters will be added as separate modules so callers do not
//! depend on one concrete memory implementation.

pub mod episode;
pub mod projection_method;
pub mod provenance;
pub mod provider;
pub mod query;
pub mod record;
pub mod registry;
pub mod storage;
pub mod types;

pub use registry::MemoryRegistry;

pub use episode::{
    ActionRef, ActionSpace, EpisodeManifest, ObservationRef, Reward, SensorStream, Step,
    TerminalReason,
};

pub use provenance::{ArtifactRef, EpisodeRef, EvidenceLocator};

pub use projection_method::ProjectionMemoryMethod;

pub use provider::{
    IngestReceipt, IngestRequest, MemoryCapabilities, MemoryError, MemoryFuture, MemoryManifest,
    MemoryMethod, OperationContext, ProviderTelemetry, RecordIngestRequest,
};

pub use query::{
    EvidenceSpan, GraphTraversal, MemoryBudget, MemoryHit, MemoryQuery, RetrieveResponse,
    TraversalDirection,
};

pub use record::{MemoryProvenance, MemoryRecord};

pub use storage::{
    JsonlProjectionStore, ProjectionStore, RelationQuery, RelationQueryResponse,
    StorageCapabilities, StorageContext, StorageError, StorageFuture, StorageManifest,
    StoreGetRequest, StoreOrder, StoreQuery, StoreQueryResponse, StoreReceipt, StoreWriteRequest,
};

pub use types::{
    ContentRef, MemoryItem, MemoryKind, MemoryRelation, MemoryScope, Modality, RelationType,
    SubjectRef, TimeRange,
};
