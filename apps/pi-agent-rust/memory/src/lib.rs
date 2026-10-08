//! Public entry point for the pluggable agent-memory crate.
//!
//! The crate exposes domain types first. Storage backends, retrieval methods,
//! and benchmark adapters will be added as separate modules so callers do not
//! depend on one concrete memory implementation.

pub mod query;
pub mod types;

pub use query::{
    EvidenceSpan, GraphTraversal, MemoryBudget, MemoryHit, MemoryQuery, RetrieveResponse,
    TraversalDirection,
};

pub use types::{
    ContentRef, MemoryItem, MemoryKind, MemoryRelation, MemoryScope, Modality, RelationType,
    SubjectRef, TimeRange,
};
