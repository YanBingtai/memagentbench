//! Shared domain types for pluggable agent memories.
//!
//! This file deliberately contains data definitions only. It does not know
//! which database, embedding model, or memory algorithm stores a value.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// The role a memory plays in an agent system.
///
/// This axis is independent from [`Modality`]. For example, a video of a
/// successful robot episode is episodic memory with video modality, while a
/// trajectory that describes a reusable skill is procedural memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Working,
    Episodic,
    Semantic,
    Procedural,
}

/// The ownership or visibility scope of a memory.
///
/// Scope is independent from [`MemoryKind`]. A personalized memory can be
/// episodic, semantic, or procedural depending on what it represents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    Personalized,
    Shared,
    Task,
    Environment,
    Agent,
    Organization,
    Custom(String),
}

impl Default for MemoryScope {
    fn default() -> Self {
        Self::Shared
    }
}

/// Identifies the person, agent, or other subject a memory belongs to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubjectRef {
    pub subject_type: String,
    pub subject_id: String,
}

/// The shape of the information carried by a memory item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    State,
    Action,
    Trajectory,
    Mixed,
}

/// A reference to memory content.
///
/// Small text and structured values can be kept inline. Large or binary
/// content must live in an artifact store and be referenced by URI, media
/// type, hash, and size. This keeps videos and images out of transcripts and
/// model request JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentRef {
    Text {
        text: String,
    },
    Json {
        value: Value,
    },
    Blob {
        uri: String,
        media_type: String,
        sha256: String,
        size_bytes: u64,
    },
}

/// A time interval in the source world or sensor clock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRange {
    /// Identifies the clock domain, such as `unix_ms` or a robot episode clock.
    pub clock_domain: String,
    pub start_ms: i64,
    pub end_ms: Option<i64>,
}

/// A typed, directed edge between two memory nodes.
///
/// A hierarchy is represented by relations such as `ParentOf`, while a more
/// general knowledge graph can use `Supports`, `Contradicts`, or `RelatedTo`.
/// Keeping edges separate from nodes allows one node to have multiple parents
/// and prevents duplicated child lists from becoming inconsistent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRelation {
    pub id: Uuid,
    pub from: Uuid,
    pub to: Uuid,
    pub relation_type: RelationType,
    pub weight: Option<f32>,
    pub confidence: Option<f32>,
    pub valid_time: Option<TimeRange>,
    pub source: String,
    pub metadata: BTreeMap<String, Value>,
    pub schema_version: u32,
}

/// Semantic meaning of an edge in the memory graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    ParentOf,
    PartOf,
    Summarizes,
    DerivedFrom,
    Refines,
    RelatedTo,
    Contradicts,
    Supports,
    Follows,
    Custom(String),
}

/// One versioned memory item shared by all memory methods.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: Uuid,
    pub version: u64,
    pub kind: MemoryKind,
    #[serde(default)]
    pub scope: MemoryScope,
    pub modality: Modality,
    pub content: ContentRef,

    /// Tenant, project, or benchmark namespace used for isolation.
    pub namespace: String,
    /// A conversation, video, or robot episode that produced this item.
    pub episode_id: Option<Uuid>,
    pub subject: Option<SubjectRef>,
    /// The source component that produced the item.
    pub source: String,
    pub metadata: BTreeMap<String, Value>,

    /// Optional abstraction level, where larger values can represent more
    /// consolidated memories. The exact meaning is controlled by the method.
    #[serde(default)]
    pub abstraction_level: u16,

    /// When the event happened in the source world.
    pub event_time: Option<TimeRange>,
    /// When the memory system accepted the item. This is used for benchmark
    /// visibility cutoffs and prevents future-data leakage.
    pub ingest_time_ms: u64,
    pub schema_version: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_kind_and_modality_are_independent() {
        let item = MemoryItem {
            id: Uuid::new_v4(),
            version: 1,
            kind: MemoryKind::Episodic,
            scope: MemoryScope::Agent,
            modality: Modality::Video,
            content: ContentRef::Blob {
                uri: "cas://episode-1.mp4".to_string(),
                media_type: "video/mp4".to_string(),
                sha256: "abc123".to_string(),
                size_bytes: 42,
            },
            namespace: "benchmark/demo".to_string(),
            episode_id: Some(Uuid::new_v4()),
            subject: Some(SubjectRef {
                subject_type: "robot".to_string(),
                subject_id: "robot-1".to_string(),
            }),
            source: "camera.front".to_string(),
            metadata: BTreeMap::new(),
            abstraction_level: 0,
            event_time: Some(TimeRange {
                clock_domain: "episode_ms".to_string(),
                start_ms: 1_000,
                end_ms: Some(2_000),
            }),
            ingest_time_ms: 10_000,
            schema_version: 1,
        };

        assert_eq!(item.kind, MemoryKind::Episodic);
        assert_eq!(item.scope, MemoryScope::Agent);
        assert_eq!(item.modality, Modality::Video);
        assert_ne!(
            item.event_time.as_ref().unwrap().start_ms,
            item.ingest_time_ms as i64
        );
    }

    #[test]
    fn blob_content_serializes_as_a_reference() {
        let content = ContentRef::Blob {
            uri: "cas://frame.png".to_string(),
            media_type: "image/png".to_string(),
            sha256: "digest".to_string(),
            size_bytes: 128,
        };

        let json = serde_json::to_value(content).expect("content should serialize");

        assert_eq!(json["type"], "blob");
        assert_eq!(json["media_type"], "image/png");
        assert_eq!(json["size_bytes"], 128);
    }

    #[test]
    fn graph_relations_serialize_their_direction_and_type() {
        let relation = MemoryRelation {
            id: Uuid::new_v4(),
            from: Uuid::new_v4(),
            to: Uuid::new_v4(),
            relation_type: RelationType::Summarizes,
            weight: Some(0.9),
            confidence: Some(0.8),
            valid_time: None,
            source: "memory-consolidator".to_string(),
            metadata: BTreeMap::new(),
            schema_version: 1,
        };

        let json = serde_json::to_value(relation).expect("relation should serialize");

        assert_eq!(json["relation_type"], "summarizes");
        let weight = json["weight"].as_f64().expect("weight should be numeric");
        assert!((weight - 0.9).abs() < 1e-6);
        assert!(json["from"].is_string());
        assert!(json["to"].is_string());
    }
}
