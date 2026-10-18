//! Provenance-bearing records exchanged by memory methods.
//!
//! [`MemoryItem`] describes what a memory contains. [`MemoryRecord`] adds the
//! information needed to reproduce where that memory came from: the episode,
//! the precise media or trajectory evidence, and the component that produced
//! it. The record contains references only; large payloads stay in an artifact
//! store.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{ArtifactRef, EpisodeRef, EvidenceLocator, MemoryItem};

/// The lineage of a memory item across ingestion, consolidation, and derived
/// memory creation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryProvenance {
    /// Component or algorithm that emitted this record.
    pub producer: String,
    /// Stable identifier of the original source, such as a sensor stream,
    /// dataset, or external service.
    pub source_id: String,
    /// Identifier of the canonical event that caused this record, when one is
    /// available. It remains an opaque string so this crate does not depend on
    /// a particular runtime implementation.
    pub source_event_id: Option<String>,
    /// Records from which this record was derived, such as a summary built
    /// from several episodic memories.
    pub parent_record_ids: Vec<Uuid>,
    /// Version of the producing method used for reproducible benchmarks.
    pub method_version: Option<String>,
}

/// A memory item together with its episode, evidence, and lineage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub item: MemoryItem,
    pub episode: Option<EpisodeRef>,
    pub evidence: Vec<EvidenceLocator>,
    pub provenance: MemoryProvenance,
}

impl MemoryRecord {
    /// Return the artifact references supporting this record without exposing
    /// any artifact payload.
    pub fn artifacts(&self) -> impl Iterator<Item = &ArtifactRef> {
        self.evidence.iter().map(|locator| &locator.artifact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentRef, MemoryKind, MemoryScope, Modality};
    use std::collections::BTreeMap;

    fn record() -> MemoryRecord {
        MemoryRecord {
            item: MemoryItem {
                id: Uuid::new_v4(),
                version: 1,
                kind: MemoryKind::Episodic,
                scope: MemoryScope::Personalized,
                modality: Modality::Video,
                content: ContentRef::Text {
                    text: "抓取红色方块".to_string(),
                },
                namespace: "benchmark/robotics".to_string(),
                episode_id: None,
                subject: None,
                source: "episode-summarizer".to_string(),
                metadata: BTreeMap::new(),
                abstraction_level: 1,
                event_time: None,
                ingest_time_ms: 20_000,
                schema_version: 1,
            },
            episode: Some(EpisodeRef {
                dataset_or_run_id: "run-42".to_string(),
                episode_id: "episode-7".to_string(),
                task: "抓取红色方块".to_string(),
                time_range: None,
                dataset_revision: "v2".to_string(),
                source: "simulator".to_string(),
            }),
            evidence: Vec::new(),
            provenance: MemoryProvenance {
                producer: "episode-summarizer".to_string(),
                source_id: "episode-7".to_string(),
                source_event_id: Some("event-101".to_string()),
                parent_record_ids: vec![Uuid::new_v4(), Uuid::new_v4()],
                method_version: Some("summarizer-0.1".to_string()),
            },
        }
    }

    #[test]
    fn provenance_keeps_derivation_identity() {
        let record = record();

        assert_eq!(
            record.provenance.source_event_id.as_deref(),
            Some("event-101")
        );
        assert_eq!(record.provenance.parent_record_ids.len(), 2);
        assert_eq!(record.episode.as_ref().unwrap().episode_id, "episode-7");
    }

    #[test]
    fn serialization_contains_references_without_payloads() {
        let record = record();
        let json = serde_json::to_value(record).expect("record should serialize");

        assert_eq!(json["provenance"]["method_version"], "summarizer-0.1");
        assert!(json.get("data").is_none());
    }
}
