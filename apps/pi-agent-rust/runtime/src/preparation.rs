//! Version-aware state for speculative resource preparation.
//!
//! A preparation plan describes work that may be started before the main
//! agent commits an action. This module deliberately does not execute a
//! sandbox, child agent, or external tool. It only owns the lifecycle and the
//! final identity check that makes a prepared result safe to reuse.

use crate::{ActionId, ActionProposal, Budget, ResourceScope, Revision, RunId};
use pi_agent_memory::{ArtifactRef, EvidenceLocator};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};
use uuid::Uuid;

/// Side-effect class permitted while preparing a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparationEffectClass {
    /// Read data or metadata; external reads may still consume a budget.
    ReadOnly,
    /// Work confined to an isolated sandbox or temporary resource.
    Isolated,
    /// A draft written only to a disposable workspace.
    Draft,
}

/// Lifecycle of a speculative preparation plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparationStatus {
    Planned,
    Starting,
    Ready,
    Consumed,
    Invalidated,
    Released,
    Failed,
    Unknown,
}

/// Durable terminal information produced by a preparation worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum PreparationOutcome {
    Failed { error: String },
    Unknown { reason: String },
}

/// Result returned by a preparation worker before the runtime records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum PreparationWorkResult {
    Ready {
        artifacts: Vec<ArtifactRef>,
        /// Optional precise locations inside the output artifacts.
        #[serde(default)]
        evidence: Vec<EvidenceLocator>,
    },
    Failed {
        error: String,
    },
    Unknown {
        reason: String,
    },
}

impl PreparationStatus {
    const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Consumed | Self::Invalidated | Self::Released | Self::Failed | Self::Unknown
        )
    }
}

/// A version-bound plan for work that may be prepared ahead of an action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparationPlan {
    pub plan_id: Uuid,
    pub run_id: RunId,
    pub action_id: ActionId,
    pub base_revision: Revision,
    /// Human-readable predicted action or tool identity.
    pub predicted_action: String,
    /// Digest of the arguments the preparation was based on.
    pub argument_digest: String,
    pub effect_class: PreparationEffectClass,
    pub resource_scope: ResourceScope,
    pub budget: Budget,
    pub confidence: f32,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
    pub status: PreparationStatus,
    /// References to outputs written by the preparation worker.
    #[serde(default)]
    pub artifacts: Vec<ArtifactRef>,
    /// Precise locations supporting the prepared artifacts. Older snapshots
    /// omit this field and are interpreted as whole-artifact references.
    #[serde(default)]
    pub evidence: Vec<EvidenceLocator>,
    /// Failure or uncertainty details for a terminal worker result.
    #[serde(default)]
    pub outcome: Option<PreparationOutcome>,
}

impl PreparationPlan {
    /// Creates a planned preparation after checking fields that are invariant
    /// for the whole lifecycle.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: RunId,
        action_id: ActionId,
        base_revision: Revision,
        predicted_action: impl Into<String>,
        argument_digest: impl Into<String>,
        effect_class: PreparationEffectClass,
        resource_scope: ResourceScope,
        budget: Budget,
        confidence: f32,
        created_at_ms: u64,
        expires_at_ms: u64,
    ) -> Result<Self, PreparationError> {
        let plan = Self {
            plan_id: Uuid::new_v4(),
            run_id,
            action_id,
            base_revision,
            predicted_action: predicted_action.into(),
            argument_digest: argument_digest.into(),
            effect_class,
            resource_scope,
            budget,
            confidence,
            created_at_ms,
            expires_at_ms,
            status: PreparationStatus::Planned,
            artifacts: Vec::new(),
            evidence: Vec::new(),
            outcome: None,
        };
        plan.validate()?;
        Ok(plan)
    }

    fn validate(&self) -> Result<(), PreparationError> {
        if self.predicted_action.trim().is_empty() {
            return Err(PreparationError::InvalidField("predicted_action"));
        }
        if self.argument_digest.trim().is_empty() {
            return Err(PreparationError::InvalidField("argument_digest"));
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(PreparationError::InvalidField("confidence"));
        }
        if self.expires_at_ms <= self.created_at_ms {
            return Err(PreparationError::InvalidExpiry {
                created_at_ms: self.created_at_ms,
                expires_at_ms: self.expires_at_ms,
            });
        }
        validate_artifacts(&self.artifacts)?;
        validate_evidence(&self.artifacts, &self.evidence)?;
        match (&self.status, &self.outcome) {
            (PreparationStatus::Failed, Some(PreparationOutcome::Failed { error }))
                if !error.trim().is_empty() => {}
            (PreparationStatus::Unknown, Some(PreparationOutcome::Unknown { reason }))
                if !reason.trim().is_empty() => {}
            (PreparationStatus::Failed | PreparationStatus::Unknown, _) => {
                return Err(PreparationError::InvalidOutcome)
            }
            (_, Some(_)) => return Err(PreparationError::InvalidOutcome),
            _ => {}
        }
        Ok(())
    }

    fn is_expired(&self, now_ms: u64) -> bool {
        now_ms >= self.expires_at_ms
    }
}

/// Errors raised by the preparation state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparationError {
    InvalidField(&'static str),
    InvalidExpiry {
        created_at_ms: u64,
        expires_at_ms: u64,
    },
    InvalidCapacity,
    CapacityExceeded {
        maximum: usize,
    },
    DuplicatePlan(Uuid),
    UnknownPlan(Uuid),
    RunMismatch {
        expected: RunId,
        actual: RunId,
    },
    RevisionMismatch {
        expected: Revision,
        actual: Revision,
    },
    ArgumentDigestMismatch,
    Expired(Uuid),
    InvalidOutcome,
    InvalidArtifact,
    InvalidEvidence,
    InvalidTransition {
        plan_id: Uuid,
        from: PreparationStatus,
        to: PreparationStatus,
    },
}

impl fmt::Display for PreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidField(field) => write!(formatter, "invalid preparation field: {field}"),
            Self::InvalidExpiry {
                created_at_ms,
                expires_at_ms,
            } => write!(
                formatter,
                "preparation expiry {expires_at_ms} must be after creation {created_at_ms}"
            ),
            Self::InvalidCapacity => formatter.write_str("preparation capacity must be positive"),
            Self::CapacityExceeded { maximum } => {
                write!(formatter, "preparation capacity reached: maximum {maximum}")
            }
            Self::DuplicatePlan(id) => write!(formatter, "preparation plan already exists: {id}"),
            Self::UnknownPlan(id) => write!(formatter, "preparation plan not found: {id}"),
            Self::RunMismatch { expected, actual } => {
                write!(
                    formatter,
                    "preparation run mismatch: expected {:?}, got {:?}",
                    expected.as_uuid(),
                    actual.as_uuid()
                )
            }
            Self::RevisionMismatch { expected, actual } => write!(
                formatter,
                "preparation revision mismatch: expected {}, got {}",
                expected.value(),
                actual.value()
            ),
            Self::ArgumentDigestMismatch => {
                formatter.write_str("prepared arguments do not match the final action")
            }
            Self::Expired(id) => write!(formatter, "preparation plan has expired: {id}"),
            Self::InvalidOutcome => {
                formatter.write_str("preparation outcome does not match its terminal status")
            }
            Self::InvalidArtifact => {
                formatter.write_str("preparation artifact reference is incomplete")
            }
            Self::InvalidEvidence => {
                formatter.write_str("preparation evidence does not match its artifact outputs")
            }
            Self::InvalidTransition { plan_id, from, to } => write!(
                formatter,
                "invalid preparation transition for {plan_id}: {from:?} -> {to:?}"
            ),
        }
    }
}

impl Error for PreparationError {}

/// In-memory lifecycle manager for bounded speculative preparations.
///
/// The manager is intentionally executor-agnostic. An async worker may call
/// `start`, perform isolated work, and then call `mark_ready`; the runtime can
/// call `consume_ready` only after it has a matching final action proposal.
#[derive(Debug, Clone)]
pub struct PreparationManager {
    max_plans: usize,
    plans: BTreeMap<Uuid, PreparationPlan>,
}

impl PreparationManager {
    pub fn new(max_plans: usize) -> Result<Self, PreparationError> {
        if max_plans == 0 {
            return Err(PreparationError::InvalidCapacity);
        }
        Ok(Self {
            max_plans,
            plans: BTreeMap::new(),
        })
    }

    pub const fn max_plans(&self) -> usize {
        self.max_plans
    }

    pub fn len(&self) -> usize {
        self.plans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    pub fn get(&self, plan_id: Uuid) -> Option<&PreparationPlan> {
        self.plans.get(&plan_id)
    }

    pub fn plans(&self) -> impl Iterator<Item = &PreparationPlan> {
        self.plans.values()
    }

    /// Restores a plan from a durable runtime event.
    pub(crate) fn restore(&mut self, plan: PreparationPlan) -> Result<(), PreparationError> {
        plan.validate()?;
        if let Some(existing) = self.plans.get(&plan.plan_id) {
            if existing == &plan {
                return Ok(());
            }
        } else {
            self.reclaim_terminal_slots();
            if self.plans.len() >= self.max_plans {
                return Err(PreparationError::CapacityExceeded {
                    maximum: self.max_plans,
                });
            }
        }
        self.plans.insert(plan.plan_id, plan);
        Ok(())
    }

    /// Adds a plan without starting any external work.
    pub fn submit(&mut self, plan: PreparationPlan, now_ms: u64) -> Result<Uuid, PreparationError> {
        plan.validate()?;
        if plan.is_expired(now_ms) {
            return Err(PreparationError::Expired(plan.plan_id));
        }
        if plan.status != PreparationStatus::Planned {
            return Err(PreparationError::InvalidTransition {
                plan_id: plan.plan_id,
                from: plan.status,
                to: PreparationStatus::Planned,
            });
        }
        if self.plans.contains_key(&plan.plan_id) {
            return Err(PreparationError::DuplicatePlan(plan.plan_id));
        }
        self.reclaim_terminal_slots();
        if self.plans.len() >= self.max_plans {
            return Err(PreparationError::CapacityExceeded {
                maximum: self.max_plans,
            });
        }
        let plan_id = plan.plan_id;
        self.plans.insert(plan_id, plan);
        Ok(plan_id)
    }

    pub fn start(
        &mut self,
        plan_id: Uuid,
        run_id: RunId,
        current_revision: Revision,
        now_ms: u64,
    ) -> Result<(), PreparationError> {
        self.transition(
            plan_id,
            run_id,
            current_revision,
            now_ms,
            PreparationStatus::Starting,
        )
    }

    pub fn mark_ready(
        &mut self,
        plan_id: Uuid,
        run_id: RunId,
        current_revision: Revision,
        now_ms: u64,
    ) -> Result<(), PreparationError> {
        self.mark_ready_with_artifacts(plan_id, run_id, current_revision, now_ms, Vec::new())
    }

    pub fn mark_ready_with_artifacts(
        &mut self,
        plan_id: Uuid,
        run_id: RunId,
        current_revision: Revision,
        now_ms: u64,
        artifacts: Vec<ArtifactRef>,
    ) -> Result<(), PreparationError> {
        self.mark_ready_with_artifacts_and_evidence(
            plan_id,
            run_id,
            current_revision,
            now_ms,
            artifacts,
            Vec::new(),
        )
    }

    /// Marks a worker result as ready and preserves precise output evidence.
    pub fn mark_ready_with_artifacts_and_evidence(
        &mut self,
        plan_id: Uuid,
        run_id: RunId,
        current_revision: Revision,
        now_ms: u64,
        artifacts: Vec<ArtifactRef>,
        evidence: Vec<EvidenceLocator>,
    ) -> Result<(), PreparationError> {
        validate_artifacts(&artifacts)?;
        validate_evidence(&artifacts, &evidence)?;
        self.transition(
            plan_id,
            run_id,
            current_revision,
            now_ms,
            PreparationStatus::Ready,
        )?;
        let plan = self
            .plans
            .get_mut(&plan_id)
            .expect("transitioned preparation must be present");
        plan.artifacts = artifacts;
        plan.evidence = evidence;
        Ok(())
    }

    /// Invalidates plans that can no longer be reused after a state change.
    pub fn invalidate_stale(
        &mut self,
        run_id: RunId,
        current_revision: Revision,
        now_ms: u64,
    ) -> usize {
        let mut invalidated = 0;
        for plan in self.plans.values_mut() {
            if plan.run_id == run_id
                && !plan.status.is_terminal()
                && (plan.base_revision != current_revision || plan.is_expired(now_ms))
            {
                plan.status = PreparationStatus::Invalidated;
                invalidated += 1;
            }
        }
        invalidated
    }

    /// Consumes a ready preparation only when the final action is identical
    /// to the action that the plan predicted.
    pub fn consume_ready(
        &mut self,
        plan_id: Uuid,
        proposal: &ActionProposal,
        now_ms: u64,
    ) -> Result<PreparationPlan, PreparationError> {
        let plan = self
            .plans
            .get_mut(&plan_id)
            .ok_or(PreparationError::UnknownPlan(plan_id))?;
        validate_action_binding(plan, proposal, now_ms)?;
        if plan.status != PreparationStatus::Ready {
            return Err(PreparationError::InvalidTransition {
                plan_id,
                from: plan.status,
                to: PreparationStatus::Consumed,
            });
        }
        plan.status = PreparationStatus::Consumed;
        Ok(plan.clone())
    }

    pub fn release(&mut self, plan_id: Uuid) -> Result<(), PreparationError> {
        let plan = self
            .plans
            .get_mut(&plan_id)
            .ok_or(PreparationError::UnknownPlan(plan_id))?;
        if plan.status.is_terminal() {
            return Err(PreparationError::InvalidTransition {
                plan_id,
                from: plan.status,
                to: PreparationStatus::Released,
            });
        }
        plan.status = PreparationStatus::Released;
        Ok(())
    }

    pub fn mark_failed(
        &mut self,
        plan_id: Uuid,
        error: impl Into<String>,
    ) -> Result<(), PreparationError> {
        let error = error.into();
        if error.trim().is_empty() {
            return Err(PreparationError::InvalidOutcome);
        }
        self.mark_terminal(
            plan_id,
            PreparationStatus::Failed,
            PreparationOutcome::Failed { error },
        )
    }

    pub fn mark_unknown(
        &mut self,
        plan_id: Uuid,
        reason: impl Into<String>,
    ) -> Result<(), PreparationError> {
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err(PreparationError::InvalidOutcome);
        }
        self.mark_terminal(
            plan_id,
            PreparationStatus::Unknown,
            PreparationOutcome::Unknown { reason },
        )
    }

    fn mark_terminal(
        &mut self,
        plan_id: Uuid,
        status: PreparationStatus,
        outcome: PreparationOutcome,
    ) -> Result<(), PreparationError> {
        let plan = self
            .plans
            .get_mut(&plan_id)
            .ok_or(PreparationError::UnknownPlan(plan_id))?;
        if plan.status.is_terminal() {
            return Err(PreparationError::InvalidTransition {
                plan_id,
                from: plan.status,
                to: status,
            });
        }
        plan.status = status;
        plan.outcome = Some(outcome);
        Ok(())
    }

    fn reclaim_terminal_slots(&mut self) {
        while self.plans.len() >= self.max_plans {
            let Some(plan_id) = self
                .plans
                .iter()
                .filter(|(_, plan)| plan.status.is_terminal())
                .min_by_key(|(plan_id, plan)| (plan.created_at_ms, **plan_id))
                .map(|(plan_id, _)| *plan_id)
            else {
                break;
            };
            self.plans.remove(&plan_id);
        }
    }

    fn transition(
        &mut self,
        plan_id: Uuid,
        run_id: RunId,
        current_revision: Revision,
        now_ms: u64,
        to: PreparationStatus,
    ) -> Result<(), PreparationError> {
        let plan = self
            .plans
            .get_mut(&plan_id)
            .ok_or(PreparationError::UnknownPlan(plan_id))?;
        if plan.run_id != run_id {
            return Err(PreparationError::RunMismatch {
                expected: plan.run_id,
                actual: run_id,
            });
        }
        if plan.base_revision != current_revision {
            return Err(PreparationError::RevisionMismatch {
                expected: plan.base_revision,
                actual: current_revision,
            });
        }
        if plan.is_expired(now_ms) {
            plan.status = PreparationStatus::Invalidated;
            return Err(PreparationError::Expired(plan_id));
        }
        let allowed = matches!(
            (plan.status, to),
            (PreparationStatus::Planned, PreparationStatus::Starting)
                | (PreparationStatus::Starting, PreparationStatus::Ready)
        );
        if !allowed {
            return Err(PreparationError::InvalidTransition {
                plan_id,
                from: plan.status,
                to,
            });
        }
        plan.status = to;
        Ok(())
    }
}

fn validate_action_binding(
    plan: &mut PreparationPlan,
    proposal: &ActionProposal,
    now_ms: u64,
) -> Result<(), PreparationError> {
    if plan.is_expired(now_ms) || proposal.is_expired(now_ms) {
        plan.status = PreparationStatus::Invalidated;
        return Err(PreparationError::Expired(plan.plan_id));
    }
    if plan.run_id != proposal.run_id {
        return Err(PreparationError::RunMismatch {
            expected: plan.run_id,
            actual: proposal.run_id,
        });
    }
    if plan.base_revision != proposal.base_revision {
        return Err(PreparationError::RevisionMismatch {
            expected: plan.base_revision,
            actual: proposal.base_revision,
        });
    }
    if plan.action_id != proposal.action_id || plan.argument_digest != proposal.arguments_digest {
        return Err(PreparationError::ArgumentDigestMismatch);
    }
    Ok(())
}

fn validate_artifacts(artifacts: &[ArtifactRef]) -> Result<(), PreparationError> {
    let mut artifact_ids = BTreeSet::new();
    for artifact in artifacts {
        if artifact.uri.trim().is_empty()
            || artifact.media_type.trim().is_empty()
            || artifact.sha256.trim().is_empty()
            || !artifact_ids.insert(artifact.artifact_id)
        {
            return Err(PreparationError::InvalidArtifact);
        }
    }
    Ok(())
}

fn validate_evidence(
    artifacts: &[ArtifactRef],
    evidence: &[EvidenceLocator],
) -> Result<(), PreparationError> {
    for locator in evidence {
        let Some(artifact) = artifacts
            .iter()
            .find(|artifact| artifact.artifact_id == locator.artifact.artifact_id)
        else {
            return Err(PreparationError::InvalidEvidence);
        };
        if artifact != &locator.artifact {
            return Err(PreparationError::InvalidEvidence);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_json_digest;
    use serde_json::json;

    fn proposal(run_id: RunId, action_id: ActionId, revision: Revision) -> ActionProposal {
        let arguments = json!({"format": "pdf"});
        ActionProposal {
            run_id,
            action_id,
            base_revision: revision,
            tool_name: "render_pdf".to_owned(),
            arguments_digest: canonical_json_digest(&arguments),
            arguments,
            risk: crate::RiskClass::ReadOnly,
            idempotency_key: "run-1/action-1".to_owned(),
            created_at_ms: 100,
            expires_at_ms: Some(2_000),
        }
    }

    fn plan(run_id: RunId, action_id: ActionId, revision: Revision) -> PreparationPlan {
        let digest = canonical_json_digest(&json!({"format": "pdf"}));
        PreparationPlan::new(
            run_id,
            action_id,
            revision,
            "render_pdf",
            digest,
            PreparationEffectClass::Draft,
            ResourceScope::new(["workspace:draft"]),
            Budget {
                wall_time_ms: Some(500),
                tokens: Some(128),
                tool_calls: Some(1),
            },
            0.8,
            100,
            1_000,
        )
        .expect("test plan should be valid")
    }

    #[test]
    fn preparation_is_consumed_only_after_matching_action_binding() {
        let run_id = RunId::new();
        let action_id = ActionId::new();
        let revision = Revision::new(4);
        let mut manager = PreparationManager::new(4).unwrap();
        let plan = plan(run_id, action_id, revision);
        let plan_id = plan.plan_id;

        manager.submit(plan, 100).unwrap();
        manager.start(plan_id, run_id, revision, 200).unwrap();
        manager.mark_ready(plan_id, run_id, revision, 300).unwrap();
        let consumed = manager
            .consume_ready(plan_id, &proposal(run_id, action_id, revision), 400)
            .unwrap();

        assert_eq!(consumed.status, PreparationStatus::Consumed);
        assert_eq!(
            manager.get(plan_id).unwrap().status,
            PreparationStatus::Consumed
        );
    }

    #[test]
    fn mismatched_arguments_do_not_consume_a_ready_plan() {
        let run_id = RunId::new();
        let action_id = ActionId::new();
        let revision = Revision::new(4);
        let mut manager = PreparationManager::new(4).unwrap();
        let prepared = plan(run_id, action_id, revision);
        let plan_id = prepared.plan_id;
        manager.submit(prepared, 100).unwrap();
        manager.start(plan_id, run_id, revision, 200).unwrap();
        manager.mark_ready(plan_id, run_id, revision, 300).unwrap();

        let mut changed = proposal(run_id, action_id, revision);
        changed.arguments = json!({"format": "html"});
        changed.arguments_digest = canonical_json_digest(&changed.arguments);
        assert_eq!(
            manager.consume_ready(plan_id, &changed, 400),
            Err(PreparationError::ArgumentDigestMismatch)
        );
        assert_eq!(
            manager.get(plan_id).unwrap().status,
            PreparationStatus::Ready
        );
    }

    #[test]
    fn revision_changes_invalidate_nonterminal_plans() {
        let run_id = RunId::new();
        let action_id = ActionId::new();
        let revision = Revision::new(4);
        let mut manager = PreparationManager::new(4).unwrap();
        let prepared = plan(run_id, action_id, revision);
        let plan_id = prepared.plan_id;
        manager.submit(prepared, 100).unwrap();
        assert_eq!(manager.invalidate_stale(run_id, Revision::new(5), 300), 1);
        assert_eq!(
            manager.get(plan_id).unwrap().status,
            PreparationStatus::Invalidated
        );
    }

    #[test]
    fn expiry_invalidates_a_plan_before_work_starts() {
        let run_id = RunId::new();
        let action_id = ActionId::new();
        let revision = Revision::new(4);
        let mut manager = PreparationManager::new(1).unwrap();
        let plan = plan(run_id, action_id, revision);
        let plan_id = plan.plan_id;
        manager.submit(plan, 100).unwrap();

        assert_eq!(
            manager.start(plan_id, run_id, revision, 1_000),
            Err(PreparationError::Expired(plan_id))
        );
        assert_eq!(
            manager.get(plan_id).unwrap().status,
            PreparationStatus::Invalidated
        );
    }

    #[test]
    fn incomplete_artifact_reference_is_rejected_before_ready_transition() {
        let run_id = RunId::new();
        let action_id = ActionId::new();
        let revision = Revision::new(4);
        let mut manager = PreparationManager::new(1).unwrap();
        let prepared = plan(run_id, action_id, revision);
        let plan_id = prepared.plan_id;
        manager.submit(prepared, 100).unwrap();
        manager.start(plan_id, run_id, revision, 200).unwrap();

        let error = manager
            .mark_ready_with_artifacts(
                plan_id,
                run_id,
                revision,
                300,
                vec![ArtifactRef {
                    artifact_id: Uuid::new_v4(),
                    uri: String::new(),
                    media_type: "video/mp4".to_owned(),
                    sha256: "sha256:video".to_owned(),
                    bytes: 1,
                    codec: None,
                    schema: None,
                    stream_key: None,
                    storage_revision: None,
                }],
            )
            .expect_err("incomplete artifact metadata must be rejected");
        assert_eq!(error, PreparationError::InvalidArtifact);
        assert_eq!(
            manager.get(plan_id).unwrap().status,
            PreparationStatus::Starting
        );
    }

    #[test]
    fn evidence_must_reference_the_exact_ready_artifact() {
        let run_id = RunId::new();
        let action_id = ActionId::new();
        let revision = Revision::new(4);
        let mut manager = PreparationManager::new(1).unwrap();
        let prepared = plan(run_id, action_id, revision);
        let plan_id = prepared.plan_id;
        manager.submit(prepared, 100).unwrap();
        manager.start(plan_id, run_id, revision, 200).unwrap();

        let artifact = ArtifactRef {
            artifact_id: Uuid::from_u128(71),
            uri: "cas://video/71".to_owned(),
            media_type: "video/mp4".to_owned(),
            sha256: "sha256:video-71".to_owned(),
            bytes: 10,
            codec: None,
            schema: None,
            stream_key: None,
            storage_revision: None,
        };
        let mut mismatched = artifact.clone();
        mismatched.uri = "cas://other/71".to_owned();
        let error = manager
            .mark_ready_with_artifacts_and_evidence(
                plan_id,
                run_id,
                revision,
                300,
                vec![artifact.clone()],
                vec![EvidenceLocator {
                    artifact: mismatched,
                    entity_path: None,
                    time_range: None,
                    frame_start: Some(1),
                    frame_end: Some(2),
                    step_start: None,
                    step_end: None,
                    byte_start: None,
                    byte_end: None,
                }],
            )
            .expect_err("evidence from another URI must be rejected");
        assert_eq!(error, PreparationError::InvalidEvidence);
        assert_eq!(
            manager.get(plan_id).unwrap().status,
            PreparationStatus::Starting
        );
    }

    #[test]
    fn duplicate_artifact_ids_are_rejected_before_ready_transition() {
        let run_id = RunId::new();
        let action_id = ActionId::new();
        let revision = Revision::new(4);
        let mut manager = PreparationManager::new(1).unwrap();
        let prepared = plan(run_id, action_id, revision);
        let plan_id = prepared.plan_id;
        manager.submit(prepared, 100).unwrap();
        manager.start(plan_id, run_id, revision, 200).unwrap();

        let artifact = ArtifactRef {
            artifact_id: Uuid::from_u128(72),
            uri: "cas://video/72".to_owned(),
            media_type: "video/mp4".to_owned(),
            sha256: "sha256:video-72".to_owned(),
            bytes: 10,
            codec: None,
            schema: None,
            stream_key: None,
            storage_revision: None,
        };
        let error = manager
            .mark_ready_with_artifacts(
                plan_id,
                run_id,
                revision,
                300,
                vec![artifact.clone(), artifact],
            )
            .expect_err("duplicate artifact IDs must be rejected");
        assert_eq!(error, PreparationError::InvalidArtifact);
        assert_eq!(
            manager.get(plan_id).unwrap().status,
            PreparationStatus::Starting
        );
    }

    #[test]
    fn old_ready_worker_results_default_to_empty_evidence() {
        let result: PreparationWorkResult = serde_json::from_value(json!({
            "kind": "ready",
            "artifacts": []
        }))
        .expect("old ready result should remain readable");
        assert_eq!(
            result,
            PreparationWorkResult::Ready {
                artifacts: Vec::new(),
                evidence: Vec::new(),
            }
        );
    }

    #[test]
    fn terminal_plans_release_capacity_for_later_predictions() {
        let run_id = RunId::new();
        let action_id = ActionId::new();
        let revision = Revision::new(4);
        let mut manager = PreparationManager::new(1).unwrap();
        let first = plan(run_id, action_id, revision);
        let first_id = first.plan_id;
        manager.submit(first, 100).unwrap();
        manager.mark_failed(first_id, "worker failed").unwrap();

        let second = plan(run_id, ActionId::new(), revision);
        let second_id = second.plan_id;
        manager.submit(second, 100).unwrap();

        assert_eq!(manager.len(), 1);
        assert!(manager.get(first_id).is_none());
        assert_eq!(
            manager.get(second_id).unwrap().status,
            PreparationStatus::Planned
        );
    }
}
