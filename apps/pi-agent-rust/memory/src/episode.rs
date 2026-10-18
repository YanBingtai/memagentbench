//! Episode and step descriptions for embodied and multimodal memory.
//!
//! An episode manifest describes the data contract of one experience. A step
//! then references observations and actions without embedding sensor payloads.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{ArtifactRef, EpisodeRef, EvidenceLocator, Modality, TimeRange};

/// Metadata for one sensor or data stream in an episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorStream {
    pub key: String,
    pub modality: Modality,
    pub media_type: Option<String>,
    pub dtype: Option<String>,
    pub shape: Vec<u64>,
    pub rate_hz: Option<f32>,
    pub units: Option<String>,
    pub coordinate_frame: Option<String>,
    pub normalization: Option<String>,
}

/// Semantics of the action values recorded by an episode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionSpace {
    pub semantics: String,
    pub coordinate_frame: Option<String>,
    pub units: Option<String>,
}

/// Dataset and sensor contract needed to reproduce an episode evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeManifest {
    pub episode: EpisodeRef,
    pub split: Option<String>,
    pub embodiment: Option<String>,
    pub streams: Vec<SensorStream>,
    pub action_space: Option<ActionSpace>,
    pub calibration_artifacts: Vec<ArtifactRef>,
}

/// A reference to one observation at one step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationRef {
    pub stream_key: String,
    pub modality: Modality,
    pub evidence: EvidenceLocator,
}

/// A reference to an executed or recorded action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionRef {
    pub action_id: String,
    pub action_type: String,
    pub command_digest: Option<String>,
    pub evidence: Option<EvidenceLocator>,
}

/// Reward received for one step, with optional named components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reward {
    pub value: f32,
    pub components: BTreeMap<String, f32>,
}

/// Why an episode or step reached a terminal state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalReason {
    Success,
    Failure,
    Timeout,
    Cancelled,
    SafetyStop,
    EnvironmentReset,
    Unknown,
    Custom(String),
}

/// One ordered interaction in an episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub step_index: u64,
    pub event_time: Option<TimeRange>,
    /// Time when the producer published the observation or action result.
    pub publish_time_ms: u64,
    /// Time when this platform accepted the record.
    pub ingest_time_ms: u64,
    pub observations: Vec<ObservationRef>,
    pub action: Option<ActionRef>,
    pub reward: Option<Reward>,
    pub discount: Option<f32>,
    pub done: bool,
    pub terminal_reason: Option<TerminalReason>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn artifact() -> ArtifactRef {
        ArtifactRef {
            artifact_id: Uuid::new_v4(),
            uri: "cas://episode-7/frame-30.png".to_string(),
            media_type: "image/png".to_string(),
            sha256: "sha256:frame-30".to_string(),
            bytes: 512,
            codec: None,
            schema: Some("rgb-v1".to_string()),
            stream_key: Some("camera/front".to_string()),
            storage_revision: Some("store-r1".to_string()),
        }
    }

    fn evidence() -> EvidenceLocator {
        EvidenceLocator {
            artifact: artifact(),
            entity_path: None,
            time_range: None,
            frame_start: Some(30),
            frame_end: Some(30),
            step_start: Some(4),
            step_end: Some(4),
            byte_start: None,
            byte_end: None,
        }
    }

    #[test]
    fn manifest_keeps_stream_shape_and_action_semantics() {
        let manifest = EpisodeManifest {
            episode: EpisodeRef {
                dataset_or_run_id: "robotics-v2".to_string(),
                episode_id: "episode-7".to_string(),
                task: "place the red block".to_string(),
                time_range: None,
                dataset_revision: "2026-10-10".to_string(),
                source: "simulator-a".to_string(),
            },
            split: Some("test".to_string()),
            embodiment: Some("franka-panda".to_string()),
            streams: vec![SensorStream {
                key: "camera/front".to_string(),
                modality: Modality::Image,
                media_type: Some("image/png".to_string()),
                dtype: Some("u8".to_string()),
                shape: vec![480, 640, 3],
                rate_hz: Some(30.0),
                units: None,
                coordinate_frame: Some("camera_front".to_string()),
                normalization: Some("[0,255]".to_string()),
            }],
            action_space: Some(ActionSpace {
                semantics: "joint_position_delta".to_string(),
                coordinate_frame: Some("robot_base".to_string()),
                units: Some("radian".to_string()),
            }),
            calibration_artifacts: vec![artifact()],
        };

        let json = serde_json::to_value(manifest).expect("manifest should serialize");

        assert_eq!(json["streams"][0]["shape"][2], 3);
        assert_eq!(json["action_space"]["semantics"], "joint_position_delta");
        assert_eq!(json["episode"]["dataset_revision"], "2026-10-10");
    }

    #[test]
    fn step_separates_event_publish_and_ingest_times() {
        let step = Step {
            step_index: 4,
            event_time: Some(TimeRange {
                clock_domain: "robot_clock".to_string(),
                start_ms: 1_000,
                end_ms: Some(1_010),
            }),
            publish_time_ms: 1_020,
            ingest_time_ms: 1_050,
            observations: vec![ObservationRef {
                stream_key: "camera/front".to_string(),
                modality: Modality::Image,
                evidence: evidence(),
            }],
            action: Some(ActionRef {
                action_id: "action-4".to_string(),
                action_type: "joint_position_delta".to_string(),
                command_digest: Some("sha256:command-4".to_string()),
                evidence: None,
            }),
            reward: Some(Reward {
                value: 1.0,
                components: BTreeMap::from([(String::from("task_success"), 1.0)]),
            }),
            discount: Some(0.99),
            done: true,
            terminal_reason: Some(TerminalReason::Success),
        };

        assert_eq!(step.event_time.as_ref().unwrap().start_ms, 1_000);
        assert_eq!(step.publish_time_ms, 1_020);
        assert_eq!(step.ingest_time_ms, 1_050);
        assert_eq!(step.terminal_reason, Some(TerminalReason::Success));
    }
}
