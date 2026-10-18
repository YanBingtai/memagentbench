//! Stable tool input for outputs produced by a consumed preparation plan.
//!
//! A preparation worker may create a video, an embedding shard, a trajectory,
//! or another large object. The tool boundary must receive references to that
//! object and its evidence, rather than loading the object into the agent
//! transcript again. This module defines that boundary contract. It does not
//! open URIs or infer metadata; those checks belong to the artifact resolver
//! used while the runtime consumes the preparation. A handoff also does not
//! grant permission to execute a tool; admission and the tool's own policy
//! remain separate checks.

use pi_agent_memory::{ArtifactRef, EvidenceLocator};
use pi_agent_runtime::{
    ActionId, Budget, PreparationEffectClass, PreparationPlan, PreparationStatus, ResourceScope,
    Revision, RiskClass, RunId,
};
use pi_agent_rust::{
    Tool, ToolConcurrency, ToolDefinition, ToolError, ToolFuture, ToolResult, ToolResultFuture,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use thiserror::Error;
use uuid::Uuid;

/// Version of the JSON object passed to a tool.
pub const PREPARED_ARTIFACT_TOOL_SCHEMA_VERSION: u32 = 1;
/// Name of the metadata-only tool that exposes one consumed preparation.
pub const PREPARED_ARTIFACT_TOOL_NAME: &str = "prepared_artifact";

/// The source of a tool input assembled from a runtime preparation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedArtifactSource {
    pub kind: PreparedArtifactSourceKind,
    pub plan_id: Uuid,
    pub run_id: RunId,
    pub action_id: ActionId,
    pub base_revision: Revision,
    pub predicted_action: String,
    /// Digest of the action arguments the preparation was bound to.
    pub argument_digest: String,
    pub effect_class: PreparationEffectClass,
    pub resource_scope: ResourceScope,
    pub budget: Budget,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
}

/// Kinds of source currently supported by the tool handoff contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparedArtifactSourceKind {
    Preparation,
}

/// References and evidence handed to a tool after preparation is consumed.
///
/// The value is intentionally metadata-only. `artifacts` carries content
/// addresses and `evidence` carries locations such as video frames, time
/// ranges, trajectory steps, or byte ranges. No media bytes are copied here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedArtifactToolInput {
    pub schema_version: u32,
    pub source: PreparedArtifactSource,
    pub artifacts: Vec<ArtifactRef>,
    pub evidence: Vec<EvidenceLocator>,
}

impl PreparedArtifactToolInput {
    /// Builds a handoff using whole-artifact evidence for each output.
    ///
    /// This is useful when the preparation worker produced an artifact but did
    /// not produce a more precise frame, time, or trajectory locator.
    pub fn from_consumed(plan: &PreparationPlan) -> Result<Self, PreparedArtifactHandoffError> {
        let mut evidence = plan.evidence.clone();
        for artifact in &plan.artifacts {
            if !evidence
                .iter()
                .any(|locator| locator.artifact.artifact_id == artifact.artifact_id)
            {
                evidence.push(whole_artifact_evidence(artifact.clone()));
            }
        }
        Self::from_consumed_with_evidence(plan, evidence)
    }

    /// Builds a handoff and rejects a plan that is already past its expiry.
    pub fn from_consumed_at(
        plan: &PreparationPlan,
        now_ms: u64,
    ) -> Result<Self, PreparedArtifactHandoffError> {
        let input = Self::from_consumed(plan)?;
        input.validate_at(now_ms)?;
        Ok(input)
    }

    /// Builds a handoff while preserving precise evidence supplied by a
    /// preparation worker. Every locator must refer to an exact artifact
    /// returned by the consumed plan, so a caller cannot smuggle an
    /// unverified URI into the tool input.
    pub fn from_consumed_with_evidence(
        plan: &PreparationPlan,
        evidence: Vec<EvidenceLocator>,
    ) -> Result<Self, PreparedArtifactHandoffError> {
        if plan.status != PreparationStatus::Consumed {
            return Err(PreparedArtifactHandoffError::Status {
                plan_id: plan.plan_id,
                actual: plan.status,
            });
        }
        if plan.artifacts.is_empty() {
            return Err(PreparedArtifactHandoffError::NoArtifacts(plan.plan_id));
        }

        let artifacts = index_artifacts(&plan.artifacts)?;
        if evidence.is_empty() {
            return Err(PreparedArtifactHandoffError::NoEvidence(plan.plan_id));
        }

        let mut covered = BTreeMap::new();
        for locator in &evidence {
            let expected = artifacts.get(&locator.artifact.artifact_id).ok_or(
                PreparedArtifactHandoffError::UnknownEvidenceArtifact {
                    artifact_id: locator.artifact.artifact_id,
                },
            )?;
            if *expected != &locator.artifact {
                return Err(PreparedArtifactHandoffError::EvidenceMetadataMismatch {
                    artifact_id: locator.artifact.artifact_id,
                });
            }
            covered.insert(locator.artifact.artifact_id, ());
        }
        if let Some((artifact_id, _)) = artifacts
            .iter()
            .find(|(artifact_id, _)| !covered.contains_key(*artifact_id))
        {
            return Err(PreparedArtifactHandoffError::MissingEvidence {
                artifact_id: *artifact_id,
            });
        }

        Ok(Self {
            schema_version: PREPARED_ARTIFACT_TOOL_SCHEMA_VERSION,
            source: PreparedArtifactSource {
                kind: PreparedArtifactSourceKind::Preparation,
                plan_id: plan.plan_id,
                run_id: plan.run_id,
                action_id: plan.action_id,
                base_revision: plan.base_revision,
                predicted_action: plan.predicted_action.clone(),
                argument_digest: plan.argument_digest.clone(),
                effect_class: plan.effect_class,
                resource_scope: plan.resource_scope.clone(),
                budget: plan.budget,
                created_at_ms: plan.created_at_ms,
                expires_at_ms: plan.expires_at_ms,
            },
            artifacts: plan.artifacts.clone(),
            evidence,
        })
    }

    /// Serializes the handoff as tool arguments without touching any artifact.
    pub fn to_arguments(&self) -> Value {
        serde_json::to_value(self).expect("prepared tool input is serializable")
    }

    /// Returns the digest a runtime proposal can record for these arguments.
    pub fn arguments_digest(&self) -> String {
        pi_agent_runtime::canonical_json_digest(&self.to_arguments())
    }

    /// Checks whether this handoff may still be used at the supplied runtime
    /// time. Memory archival can use [`Self::from_consumed`] without this
    /// check; side-effecting tools should use this guard immediately before
    /// execution.
    pub fn validate_at(&self, now_ms: u64) -> Result<(), PreparedArtifactHandoffError> {
        if now_ms >= self.source.expires_at_ms {
            return Err(PreparedArtifactHandoffError::Expired {
                plan_id: self.source.plan_id,
                expires_at_ms: self.source.expires_at_ms,
                now_ms,
            });
        }
        Ok(())
    }
}

/// A read-only tool that returns one consumed preparation to the model loop.
///
/// The tool is intentionally metadata-only: it exposes content-addressed
/// artifact references and evidence locators, but never opens a URI or copies
/// media bytes into the prompt. A downstream modality-specific tool can use
/// the returned references under its own runtime admission policy.
#[derive(Debug, Clone)]
pub struct PreparedArtifactTool {
    input: PreparedArtifactToolInput,
}

impl PreparedArtifactTool {
    /// Creates a tool from a plan that has already passed runtime consumption.
    pub fn from_consumed(plan: &PreparationPlan) -> Result<Self, PreparedArtifactHandoffError> {
        Ok(Self {
            input: PreparedArtifactToolInput::from_consumed(plan)?,
        })
    }

    /// Creates a tool only while the consumed preparation is still valid.
    pub fn from_consumed_at(
        plan: &PreparationPlan,
        now_ms: u64,
    ) -> Result<Self, PreparedArtifactHandoffError> {
        Ok(Self {
            input: PreparedArtifactToolInput::from_consumed_at(plan, now_ms)?,
        })
    }

    pub fn input(&self) -> &PreparedArtifactToolInput {
        &self.input
    }

    fn result(&self) -> ToolResult {
        let value = self.input.to_arguments();
        let content = serde_json::to_string(&value)
            .expect("prepared artifact handoff should always serialize");
        ToolResult::text(content).with_details(value)
    }
}

impl Tool for PreparedArtifactTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            PREPARED_ARTIFACT_TOOL_NAME,
            "Return metadata for a verified prepared artifact without opening its content.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        )
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        if arguments
            .as_object()
            .is_some_and(|object| object.is_empty())
        {
            Ok(())
        } else {
            Err(ToolError::InvalidArgumentValue {
                name: PREPARED_ARTIFACT_TOOL_NAME.to_owned(),
                reason: "prepared_artifact accepts an empty JSON object".to_owned(),
            })
        }
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    fn replay_policy(&self) -> pi_agent_rust::tools::ToolReplayPolicy {
        pi_agent_rust::tools::ToolReplayPolicy::Safe
    }

    fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move { Ok(self.result().content) })
    }

    fn execute_result<'a>(&'a self, _arguments: Value) -> ToolResultFuture<'a> {
        Box::pin(async move { Ok(self.result()) })
    }
}

fn index_artifacts(
    artifacts: &[ArtifactRef],
) -> Result<BTreeMap<Uuid, &ArtifactRef>, PreparedArtifactHandoffError> {
    let mut indexed = BTreeMap::new();
    for artifact in artifacts {
        if artifact.uri.trim().is_empty()
            || artifact.media_type.trim().is_empty()
            || artifact.sha256.trim().is_empty()
        {
            return Err(PreparedArtifactHandoffError::InvalidArtifact {
                artifact_id: artifact.artifact_id,
            });
        }
        if indexed.insert(artifact.artifact_id, artifact).is_some() {
            return Err(PreparedArtifactHandoffError::DuplicateArtifact {
                artifact_id: artifact.artifact_id,
            });
        }
    }
    Ok(indexed)
}

fn whole_artifact_evidence(artifact: ArtifactRef) -> EvidenceLocator {
    EvidenceLocator {
        artifact,
        entity_path: None,
        time_range: None,
        frame_start: None,
        frame_end: None,
        step_start: None,
        step_end: None,
        byte_start: None,
        byte_end: None,
    }
}

/// Errors raised before a prepared value crosses the tool boundary.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PreparedArtifactHandoffError {
    #[error("preparation {plan_id} must be consumed before tool handoff; status is {actual:?}")]
    Status {
        plan_id: Uuid,
        actual: PreparationStatus,
    },
    #[error("consumed preparation {0} has no artifacts")]
    NoArtifacts(Uuid),
    #[error("consumed preparation {0} has no evidence")]
    NoEvidence(Uuid),
    #[error("preparation artifact {artifact_id} has incomplete metadata")]
    InvalidArtifact { artifact_id: Uuid },
    #[error("preparation contains duplicate artifact {artifact_id}")]
    DuplicateArtifact { artifact_id: Uuid },
    #[error("evidence refers to unknown preparation artifact {artifact_id}")]
    UnknownEvidenceArtifact { artifact_id: Uuid },
    #[error("evidence metadata does not match preparation artifact {artifact_id}")]
    EvidenceMetadataMismatch { artifact_id: Uuid },
    #[error("preparation artifact {artifact_id} has no evidence")]
    MissingEvidence { artifact_id: Uuid },
    #[error(
        "prepared artifact handoff {plan_id} expired at {expires_at_ms}; current time is {now_ms}"
    )]
    Expired {
        plan_id: Uuid,
        expires_at_ms: u64,
        now_ms: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_runtime::{ActionId, Budget, PreparationEffectClass, ResourceScope};

    fn consumed_plan() -> PreparationPlan {
        let mut plan = PreparationPlan::new(
            RunId::from_uuid(Uuid::from_u128(1)),
            ActionId::from_uuid(Uuid::from_u128(2)),
            Revision::new(7),
            "memory.retrieve_video",
            "sha256:original-action-arguments",
            PreparationEffectClass::ReadOnly,
            ResourceScope::new(["memory:video"]),
            Budget::unbounded(),
            0.9,
            100,
            200,
        )
        .expect("test plan should be valid");
        plan.status = PreparationStatus::Consumed;
        plan.artifacts = vec![ArtifactRef {
            artifact_id: Uuid::from_u128(3),
            uri: "cas://video/3".to_owned(),
            media_type: "video/mp4".to_owned(),
            sha256: "sha256:video-3".to_owned(),
            bytes: 10,
            codec: Some("h264".to_owned()),
            schema: Some("video-v1".to_owned()),
            stream_key: Some("front".to_owned()),
            storage_revision: Some("store-7".to_owned()),
        }];
        plan
    }

    #[test]
    fn consumed_plan_becomes_metadata_only_tool_arguments() {
        let input = PreparedArtifactToolInput::from_consumed(&consumed_plan())
            .expect("consumed plan should cross the handoff");
        let value = input.to_arguments();

        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["source"]["kind"], "preparation");
        assert_eq!(value["artifacts"][0]["uri"], "cas://video/3");
        assert_eq!(value["evidence"][0]["artifact"]["sha256"], "sha256:video-3");
        assert!(value.get("data").is_none());
        assert!(value["artifacts"][0].get("data").is_none());
    }

    #[test]
    fn ready_plan_cannot_cross_the_tool_boundary() {
        let plan = {
            let mut plan = consumed_plan();
            plan.status = PreparationStatus::Ready;
            plan
        };

        assert!(matches!(
            PreparedArtifactToolInput::from_consumed(&plan),
            Err(PreparedArtifactHandoffError::Status { .. })
        ));
    }

    #[test]
    fn evidence_must_match_the_verified_artifact_metadata() {
        let plan = consumed_plan();
        let evidence = vec![EvidenceLocator {
            artifact: ArtifactRef {
                uri: "cas://attacker/3".to_owned(),
                ..plan.artifacts[0].clone()
            },
            entity_path: Some("frame/3".to_owned()),
            time_range: None,
            frame_start: Some(3),
            frame_end: Some(4),
            step_start: None,
            step_end: None,
            byte_start: None,
            byte_end: None,
        }];

        assert!(matches!(
            PreparedArtifactToolInput::from_consumed_with_evidence(&plan, evidence),
            Err(PreparedArtifactHandoffError::EvidenceMetadataMismatch { .. })
        ));
    }

    #[test]
    fn precise_evidence_is_preserved_and_digest_is_stable() {
        let plan = consumed_plan();
        let evidence = vec![EvidenceLocator {
            artifact: plan.artifacts[0].clone(),
            entity_path: Some("robot/gripper".to_owned()),
            time_range: None,
            frame_start: Some(30),
            frame_end: Some(45),
            step_start: Some(7),
            step_end: Some(9),
            byte_start: None,
            byte_end: None,
        }];
        let input = PreparedArtifactToolInput::from_consumed_with_evidence(&plan, evidence.clone())
            .expect("precise evidence should be accepted");

        assert_eq!(input.evidence, evidence);
        assert_eq!(
            input.arguments_digest(),
            pi_agent_runtime::canonical_json_digest(&input.to_arguments())
        );
        let mut changed = input.clone();
        changed.evidence[0].frame_start = Some(31);
        assert_ne!(input.arguments_digest(), changed.arguments_digest());
    }

    #[test]
    fn consumed_plan_evidence_is_forwarded_without_rebuilding_locations() {
        let mut plan = consumed_plan();
        let locator = EvidenceLocator {
            artifact: plan.artifacts[0].clone(),
            entity_path: Some("robot/gripper/object-3".to_owned()),
            time_range: None,
            frame_start: Some(36),
            frame_end: Some(42),
            step_start: Some(12),
            step_end: Some(14),
            byte_start: None,
            byte_end: None,
        };
        plan.evidence = vec![locator.clone()];

        let input = PreparedArtifactToolInput::from_consumed(&plan)
            .expect("plan evidence should be forwarded");
        assert_eq!(input.evidence, vec![locator]);
    }

    #[test]
    fn tool_handoff_rejects_expired_preparation() {
        let plan = consumed_plan();
        assert!(PreparedArtifactToolInput::from_consumed_at(&plan, 199).is_ok());
        assert!(matches!(
            PreparedArtifactToolInput::from_consumed_at(&plan, 200),
            Err(PreparedArtifactHandoffError::Expired { .. })
        ));
    }

    #[tokio::test]
    async fn prepared_artifact_tool_returns_consumed_metadata() {
        let tool = PreparedArtifactTool::from_consumed(&consumed_plan())
            .expect("consumed plan should create a metadata tool");
        assert_eq!(tool.definition().function.name, PREPARED_ARTIFACT_TOOL_NAME);

        let result = tool
            .execute_result(serde_json::json!({}))
            .await
            .expect("metadata tool should accept an empty object");
        assert_eq!(result.details["schema_version"], 1);
        assert_eq!(result.details["artifacts"][0]["uri"], "cas://video/3");
        assert_eq!(result.details["evidence"][0]["frame_start"], Value::Null);
        assert!(result.content.contains("cas://video/3"));

        let error = tool
            .validate_arguments(&serde_json::json!({"path": "video"}))
            .expect_err("metadata tool must reject arbitrary arguments");
        assert!(matches!(
            error,
            pi_agent_rust::ToolError::InvalidArgumentValue { .. }
        ));
    }

    #[test]
    fn prepared_artifact_tool_rejects_expired_consumed_plan() {
        let plan = consumed_plan();
        assert!(PreparedArtifactTool::from_consumed_at(&plan, 199).is_ok());
        assert!(matches!(
            PreparedArtifactTool::from_consumed_at(&plan, 200),
            Err(PreparedArtifactHandoffError::Expired { .. })
        ));
    }
}
