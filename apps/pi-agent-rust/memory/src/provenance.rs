//! Provenance and evidence references shared by memory methods and benchmarks.
//!
//! These types describe where a memory came from. They deliberately reference
//! large media instead of embedding bytes in a memory record or transcript.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::TimeRange;

/// A content-addressed media or structured-data artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub artifact_id: Uuid,
    pub uri: String,
    pub media_type: String,
    pub sha256: String,
    pub bytes: u64,
    pub codec: Option<String>,
    pub schema: Option<String>,
    pub stream_key: Option<String>,
    pub storage_revision: Option<String>,
}

/// A precise location inside an artifact that supports a memory hit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLocator {
    pub artifact: ArtifactRef,
    pub entity_path: Option<String>,
    pub time_range: Option<TimeRange>,
    pub frame_start: Option<u64>,
    pub frame_end: Option<u64>,
    pub step_start: Option<u64>,
    pub step_end: Option<u64>,
    pub byte_start: Option<u64>,
    pub byte_end: Option<u64>,
}

/// Identifies the dataset or run that produced one independently evaluable episode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeRef {
    pub dataset_or_run_id: String,
    pub episode_id: String,
    pub task: String,
    pub time_range: Option<TimeRange>,
    pub dataset_revision: String,
    pub source: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video_artifact() -> ArtifactRef {
        ArtifactRef {
            artifact_id: Uuid::new_v4(),
            uri: "cas://recordings/episode-7.mp4".to_string(),
            media_type: "video/mp4".to_string(),
            sha256: "sha256:episode-7".to_string(),
            bytes: 4_096,
            codec: Some("h264".to_string()),
            schema: Some("camera-recording-v1".to_string()),
            stream_key: Some("camera/front".to_string()),
            storage_revision: Some("store-r17".to_string()),
        }
    }

    #[test]
    fn artifact_reference_serializes_identity_without_payload() {
        let artifact = video_artifact();
        let json = serde_json::to_value(artifact).expect("artifact should serialize");

        assert_eq!(json["media_type"], "video/mp4");
        assert_eq!(json["stream_key"], "camera/front");
        assert_eq!(json["bytes"], 4_096);
        assert!(json.get("data").is_none());
    }

    #[test]
    fn evidence_locator_preserves_video_time_and_trajectory_steps() {
        let locator = EvidenceLocator {
            artifact: video_artifact(),
            entity_path: Some("robot/gripper/object-3".to_string()),
            time_range: Some(TimeRange {
                clock_domain: "episode_ms".to_string(),
                start_ms: 12_000,
                end_ms: Some(14_500),
            }),
            frame_start: Some(360),
            frame_end: Some(435),
            step_start: Some(30),
            step_end: Some(45),
            byte_start: None,
            byte_end: None,
        };

        let json = serde_json::to_value(locator).expect("locator should serialize");

        assert_eq!(json["time_range"]["start_ms"], 12_000);
        assert_eq!(json["frame_end"], 435);
        assert_eq!(json["step_start"], 30);
    }

    #[test]
    fn episode_reference_keeps_dataset_revision_separate_from_event_time() {
        let episode = EpisodeRef {
            dataset_or_run_id: "robotics-v2".to_string(),
            episode_id: "episode-7".to_string(),
            task: "place the red block".to_string(),
            time_range: Some(TimeRange {
                clock_domain: "robot_clock".to_string(),
                start_ms: 100,
                end_ms: Some(2_400),
            }),
            dataset_revision: "2026-10-10".to_string(),
            source: "simulator-a".to_string(),
        };

        assert_eq!(episode.dataset_revision, "2026-10-10");
        assert_eq!(
            episode.time_range.as_ref().unwrap().clock_domain,
            "robot_clock"
        );
    }
}
