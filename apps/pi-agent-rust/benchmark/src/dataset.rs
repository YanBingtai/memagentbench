//! Immutable identity metadata for benchmark datasets.
//!
//! A dataset manifest describes the input revision used by an experiment. It
//! deliberately stores identity and shape metadata only; records and large
//! artifacts remain in their own dataset and artifact stores.

use pi_agent_memory::Modality;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const DATASET_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Stable identity and shape metadata for one benchmark dataset revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub dataset_id: String,
    pub revision: String,
    pub schema_version: u32,
    pub content_digest: String,
    #[serde(default)]
    pub modalities: Vec<Modality>,
    #[serde(default)]
    pub sample_count: Option<u64>,
    #[serde(default)]
    pub episode_count: Option<u64>,
}

impl DatasetManifest {
    /// Creates a manifest using the current dataset metadata schema.
    pub fn new(
        dataset_id: impl Into<String>,
        revision: impl Into<String>,
        content_digest: impl Into<String>,
        modalities: Vec<Modality>,
    ) -> Self {
        Self {
            dataset_id: dataset_id.into(),
            revision: revision.into(),
            schema_version: DATASET_MANIFEST_SCHEMA_VERSION,
            content_digest: content_digest.into(),
            modalities,
            sample_count: None,
            episode_count: None,
        }
    }

    /// Validates the identity fields before a dataset is bound to a run.
    pub fn validate(&self) -> Result<(), DatasetManifestError> {
        if self.dataset_id.trim().is_empty() {
            return Err(DatasetManifestError::Invalid(
                "dataset_id must be non-empty".to_owned(),
            ));
        }
        if self.revision.trim().is_empty() {
            return Err(DatasetManifestError::Invalid(
                "dataset revision must be non-empty".to_owned(),
            ));
        }
        if self.schema_version == 0 {
            return Err(DatasetManifestError::Invalid(
                "dataset schema_version must be greater than zero".to_owned(),
            ));
        }
        if !is_sha256_digest(&self.content_digest) {
            return Err(DatasetManifestError::Invalid(
                "dataset content_digest must use sha256:<64 lowercase hex>".to_owned(),
            ));
        }
        for (index, modality) in self.modalities.iter().enumerate() {
            if self.modalities[..index].contains(modality) {
                return Err(DatasetManifestError::Invalid(format!(
                    "dataset modalities contain a duplicate at index {index}"
                )));
            }
        }
        Ok(())
    }

    /// Returns whether this dataset declares content that normally lives in
    /// an artifact store rather than inline JSONL fields.
    ///
    /// State, action, and trajectory are intentionally not included: a
    /// dataset can encode those modalities as small structured values. A
    /// `Mixed` declaration is treated as binary-capable because it may carry
    /// images, audio, or video alongside inline content.
    pub fn contains_binary_modality(&self) -> bool {
        self.modalities.iter().any(|modality| {
            matches!(
                modality,
                Modality::Image | Modality::Audio | Modality::Video | Modality::Mixed
            )
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DatasetManifestError {
    #[error("invalid dataset manifest: {0}")]
    Invalid(String),
}

fn is_sha256_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArtifactValidationPolicy, BenchmarkManifest, MethodManifest};
    use std::collections::BTreeMap;

    const DIGEST: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn manifest() -> DatasetManifest {
        DatasetManifest::new(
            "robot-episodes",
            "2026-10-14",
            DIGEST,
            vec![Modality::Text, Modality::Video, Modality::Trajectory],
        )
    }

    #[test]
    fn multimodal_manifest_round_trips() {
        let mut value = manifest();
        value.sample_count = Some(12);
        value.episode_count = Some(3);
        value.validate().unwrap();
        let encoded = serde_json::to_value(&value).unwrap();
        let decoded: DatasetManifest = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn validation_rejects_bad_digest_and_duplicate_modalities() {
        let mut bad_digest = manifest();
        bad_digest.content_digest = "sha256:BAD".to_owned();
        assert!(bad_digest.validate().is_err());

        let mut duplicate = manifest();
        duplicate.modalities.push(Modality::Video);
        assert!(duplicate.validate().is_err());
    }

    #[test]
    fn binary_modality_detection_is_explicit() {
        assert!(manifest().contains_binary_modality());
        let inline_only = DatasetManifest::new(
            "state-dataset",
            "2026-10-14",
            DIGEST,
            vec![
                Modality::Text,
                Modality::State,
                Modality::Action,
                Modality::Trajectory,
            ],
        );
        assert!(!inline_only.contains_binary_modality());
        let mixed = DatasetManifest::new("mixed", "2026-10-14", DIGEST, vec![Modality::Mixed]);
        assert!(mixed.contains_binary_modality());
    }

    #[test]
    fn binding_persists_dataset_identity_and_rejects_conflicts() {
        let mut benchmark = BenchmarkManifest::new(
            "retrieval",
            "bench",
            MethodManifest {
                method_id: "memory".to_owned(),
                version: "1".to_owned(),
                config: BTreeMap::new(),
            },
            "2026-10-14",
            Some(100),
        );
        benchmark.bind_dataset(manifest()).unwrap();
        assert_eq!(
            benchmark.dataset.as_ref().unwrap().dataset_id,
            "robot-episodes"
        );
        assert_eq!(benchmark.dataset_digest.as_deref(), Some(DIGEST));

        let mut conflict = manifest();
        conflict.revision = "other".to_owned();
        assert!(benchmark.bind_dataset(conflict).is_err());
    }

    #[test]
    fn old_benchmark_manifest_without_dataset_field_remains_readable() {
        let benchmark = BenchmarkManifest::new(
            "retrieval",
            "bench",
            MethodManifest {
                method_id: "memory".to_owned(),
                version: "1".to_owned(),
                config: BTreeMap::new(),
            },
            "legacy",
            None,
        );
        let mut encoded = serde_json::to_value(&benchmark).unwrap();
        encoded.as_object_mut().unwrap().remove("dataset");
        let decoded: BenchmarkManifest = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, benchmark);
        assert!(decoded.dataset.is_none());
    }

    #[test]
    fn artifact_validation_policy_round_trips_and_defaults_for_old_manifests() {
        let mut benchmark = BenchmarkManifest::new(
            "retrieval",
            "bench",
            MethodManifest {
                method_id: "memory".to_owned(),
                version: "1".to_owned(),
                config: BTreeMap::new(),
            },
            "dataset-1",
            None,
        );
        benchmark.artifact_validation_policy = ArtifactValidationPolicy::LocalStore;
        let encoded = serde_json::to_value(&benchmark).unwrap();
        let decoded: BenchmarkManifest = serde_json::from_value(encoded).unwrap();
        assert_eq!(
            decoded.artifact_validation_policy,
            ArtifactValidationPolicy::LocalStore
        );

        let mut legacy = serde_json::to_value(&benchmark).unwrap();
        legacy
            .as_object_mut()
            .unwrap()
            .remove("artifact_validation_policy");
        let decoded: BenchmarkManifest = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            decoded.artifact_validation_policy,
            ArtifactValidationPolicy::Unspecified
        );
    }
}
