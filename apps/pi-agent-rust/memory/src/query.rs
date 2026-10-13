//! Query and evidence types shared by memory methods.
//!
//! A query describes the information boundary and resource budget. A hit
//! describes where the supporting evidence came from, so benchmark runners can
//! evaluate retrieval independently from the model's final answer.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{
    ContentRef, MemoryKind, MemoryScope, Modality, RelationType, SubjectRef, TimeRange,
};

/// Limits applied to one memory retrieval operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryBudget {
    pub max_hits: usize,
    pub max_evidence_bytes: usize,
    pub max_context_tokens: usize,
    pub max_latency_ms: Option<u64>,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            max_hits: 8,
            max_evidence_bytes: 64 * 1024,
            max_context_tokens: 4_096,
            max_latency_ms: None,
        }
    }
}

/// Direction used when walking typed memory relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraversalDirection {
    Outgoing,
    Incoming,
    Both,
}

/// Bounded graph traversal attached to a memory query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphTraversal {
    pub start_ids: Vec<Uuid>,
    pub relation_types: Vec<RelationType>,
    pub direction: TraversalDirection,
    pub max_depth: u16,
    pub max_nodes: usize,
}

/// A query over text, binary references, metadata, time, or graph structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryQuery {
    /// Text, image, video, audio, or structured input for the method encoder.
    pub input: Option<ContentRef>,
    pub namespace: String,
    pub scope: Option<MemoryScope>,
    pub subject: Option<SubjectRef>,
    pub kinds: Vec<MemoryKind>,
    pub modalities: Vec<Modality>,
    pub time_range: Option<TimeRange>,
    /// Only records accepted no later than this timestamp are visible.
    pub visibility_cutoff_ms: Option<u64>,
    pub graph: Option<GraphTraversal>,
    pub budget: MemoryBudget,
}

/// The portion of a memory item that supports a result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvidenceSpan {
    Text {
        start_char: usize,
        end_char: usize,
    },
    Image {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    Audio {
        start_ms: u64,
        end_ms: Option<u64>,
    },
    Video {
        start_ms: u64,
        end_ms: Option<u64>,
        frame_start: Option<u64>,
        frame_end: Option<u64>,
    },
    Trajectory {
        start_step: u64,
        end_step: Option<u64>,
    },
}

/// One ranked result returned by a memory method.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryHit {
    pub item_id: Uuid,
    pub version: u64,
    pub method_id: String,
    pub rank: usize,
    pub score: Option<f32>,
    pub evidence: Vec<EvidenceSpan>,
    /// A small projection suitable for context assembly, when available.
    pub projection: Option<ContentRef>,
    pub provenance: String,
    pub snapshot_id: Option<String>,
    pub latency_ms: u64,
}

/// The complete response from one retrieval operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrieveResponse {
    pub query_id: Uuid,
    pub hits: Vec<MemoryHit>,
    pub snapshot_id: Option<String>,
    pub latency_ms: u64,
    pub truncated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_budget_has_bounded_context_and_hit_count() {
        let budget = MemoryBudget::default();

        assert_eq!(budget.max_hits, 8);
        assert_eq!(budget.max_context_tokens, 4_096);
        assert!(budget.max_evidence_bytes > 0);
    }

    #[test]
    fn video_evidence_preserves_temporal_and_frame_locations() {
        let evidence = EvidenceSpan::Video {
            start_ms: 12_000,
            end_ms: Some(15_000),
            frame_start: Some(360),
            frame_end: Some(450),
        };

        let json = serde_json::to_value(evidence).expect("evidence should serialize");

        assert_eq!(json["type"], "video");
        assert_eq!(json["start_ms"], 12_000);
        assert_eq!(json["frame_start"], 360);
    }

    #[test]
    fn graph_query_keeps_namespace_and_traversal_budget() {
        let query = MemoryQuery {
            input: Some(ContentRef::Text {
                text: "successful grasp".to_string(),
            }),
            namespace: "benchmark/robotics".to_string(),
            scope: Some(MemoryScope::Agent),
            subject: Some(SubjectRef {
                subject_type: "robot".to_string(),
                subject_id: "robot-1".to_string(),
            }),
            kinds: vec![MemoryKind::Episodic],
            modalities: vec![Modality::Trajectory, Modality::Video],
            time_range: None,
            visibility_cutoff_ms: Some(100_000),
            graph: Some(GraphTraversal {
                start_ids: vec![Uuid::new_v4()],
                relation_types: vec![RelationType::DerivedFrom],
                direction: TraversalDirection::Incoming,
                max_depth: 2,
                max_nodes: 32,
            }),
            budget: MemoryBudget {
                max_hits: 5,
                ..MemoryBudget::default()
            },
        };

        let json = serde_json::to_value(query).expect("query should serialize");

        assert_eq!(json["namespace"], "benchmark/robotics");
        assert_eq!(json["graph"]["direction"], "incoming");
        assert_eq!(json["budget"]["max_hits"], 5);
    }
}
