//! Runtime-facing contract for prepared memory injection.
//!
//! The memory agent may retrieve, rank, and rewrite memories asynchronously,
//! but it must return a versioned proposal instead of mutating a model prompt
//! directly. The runtime validates that proposal at the last possible moment
//! and selects only content that still fits the current request budget.
//!
//! This module deliberately uses opaque UUIDs and a small rendered projection.
//! It therefore does not depend on a concrete memory provider or on the main
//! agent's message representation. A provider can keep video, audio, graph,
//! and robot-state artifacts behind the projection's provenance and render a
//! compact text view for the current model call.

use crate::{Revision, RunId};
use serde::{Deserialize, Serialize};
use std::{error::Error, fmt};
use uuid::Uuid;

/// Maximum number of candidates a single hook evaluation may inspect.
///
/// This is a runtime hard limit. A caller may configure a lower limit through
/// [`MemoryHookContext::max_candidates`], but cannot raise it by constructing a
/// larger context value.
pub const MAX_HOOK_CANDIDATES: u32 = 128;

/// Maximum rendered projection size accepted by the hook, in UTF-8 bytes.
///
/// The model context budget is usually lower, but this bound also protects the
/// runtime when a proposal is evaluated with a generous token budget.
pub const MAX_PROJECTION_BYTES: u32 = 256 * 1024;

const APPROX_CHARS_PER_TOKEN: usize = 4;

/// Version of the memory data snapshot used to build a proposal.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct MemoryRevision(u64);

impl MemoryRevision {
    /// The initial memory snapshot before any item has been accepted.
    pub const INITIAL: Self = Self(0);

    /// Creates a revision from a persisted value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric representation used at integration boundaries.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Version of the policy that decided which memory may be injected.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct PolicyRevision(u64);

impl PolicyRevision {
    /// The initial policy snapshot.
    pub const INITIAL: Self = Self(0);

    /// Creates a revision from a persisted value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric representation used at integration boundaries.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// The compact, task-bound view prepared for a model request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryProjection {
    /// Identity of this derived view, distinct from the source memory id.
    pub projection_id: Uuid,
    /// Opaque identity of the canonical memory item.
    pub source_memory_id: Uuid,
    /// Version of the canonical item used to create this view.
    pub source_version: u64,
    /// Rendered content suitable for the current model message format.
    pub content: String,
    /// Provider-neutral evidence references. The runtime keeps these as JSON
    /// values so memory providers can preserve modality-specific locations
    /// without making the runtime depend on a concrete memory crate.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_refs: Vec<serde_json::Value>,
    /// Provider and transformation provenance for audit and debugging.
    pub provenance: String,
    /// Estimated context-token cost of injecting this projection.
    pub token_cost: u32,
}

/// One ranked memory candidate returned by the memory agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryCandidate {
    pub candidate_id: Uuid,
    pub projection: MemoryProjection,
    /// Ranking score in the inclusive range [0, 1]. The runtime does not
    /// recompute provider-specific ranking features; it only validates them.
    pub score: f32,
    /// Calibrated confidence that this candidate is useful for the task.
    pub confidence: f32,
    /// Candidate expiry is usually earlier than the proposal expiry.
    pub expires_at_ms: Option<u64>,
}

/// Provider-neutral retrieval facts carried from the memory agent to the
/// runtime audit boundary.
///
/// The runtime records these facts without knowing how a provider searched,
/// ranked, or rendered a memory. A missing summary means the producer used an
/// older proposal contract and is therefore kept backward compatible.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRetrievalSummary {
    /// Digest of the fully bound query sent to the provider fan-out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_digest: Option<String>,
    /// Snapshot marker covering the retrieval response when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_id: Option<String>,
    /// Number of hits returned by the retrieval fan-out before projection.
    #[serde(default)]
    pub hit_count: u32,
    /// Whether the retrieval result was bounded by a provider or coordinator
    /// limit (including a retrieval deadline).
    #[serde(default)]
    pub truncated: bool,
    /// Stable identities for hits considered by the memory agent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hits: Vec<MemoryRetrievalHit>,
    /// Provider failures retained alongside successful retrievals.
    #[serde(default)]
    pub provider_failures: Vec<MemoryProviderFailure>,
}

/// Provider-neutral identity of one retrieval hit.
///
/// Evidence IDs remain opaque so text spans, image regions, video intervals,
/// and trajectory steps can share the runtime ledger contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRetrievalHit {
    pub item_id: Uuid,
    /// Canonical source records supporting this provider hit. The provider
    /// item id remains the durable runtime identity; these IDs preserve
    /// lineage for benchmark evaluation when a method returns a derived item.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_record_ids: Vec<Uuid>,
    pub version: u64,
    pub method_id: String,
    pub rank: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_ids: Vec<String>,
}

/// A provider failure suitable for durable, provider-neutral audit records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryProviderFailure {
    pub method_id: String,
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub retryable: bool,
}

impl MemoryCandidate {
    fn validate(&self) -> Result<(), MemoryProposalError> {
        if !self.score.is_finite() || !(0.0..=1.0).contains(&self.score) {
            return Err(MemoryProposalError::InvalidScore(self.candidate_id));
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(MemoryProposalError::InvalidConfidence(self.candidate_id));
        }
        if self.projection.content.trim().is_empty() {
            return Err(MemoryProposalError::EmptyProjection(self.candidate_id));
        }
        if self.projection.provenance.trim().is_empty() {
            return Err(MemoryProposalError::EmptyProvenance(self.candidate_id));
        }

        let content_bytes = self.projection.content.len();
        let evidence_bytes = serialized_evidence_bytes(&self.projection.evidence_refs);
        let projection_bytes = content_bytes.saturating_add(evidence_bytes);
        let minimum_tokens = approximate_projection_token_cost(
            &self.projection.content,
            &self.projection.evidence_refs,
        );
        if self.projection.token_cost < minimum_tokens
            || usize::try_from(self.projection.token_cost).is_ok_and(|cost| cost > projection_bytes)
        {
            return Err(MemoryProposalError::InvalidTokenCost {
                candidate_id: self.candidate_id,
                token_cost: self.projection.token_cost,
                minimum: minimum_tokens,
                maximum: projection_bytes.min(u32::MAX as usize) as u32,
            });
        }

        Ok(())
    }

    fn is_expired(&self, now_ms: u64) -> bool {
        self.expires_at_ms.is_some_and(|expiry| now_ms >= expiry)
    }
}

/// Snapshot of runtime state available to the pre-model-call hook.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryHookContext {
    pub run_id: RunId,
    /// Optional model/turn identity used to correlate memory decisions with
    /// one request in a multi-turn run. Older serialized contexts may omit it.
    #[serde(default)]
    pub turn_id: Option<Uuid>,
    /// Namespace/tenant that owns the current run and its memory.
    pub namespace: String,
    pub current_revision: Revision,
    pub memory_revision: MemoryRevision,
    pub policy_revision: PolicyRevision,
    pub now_ms: u64,
    /// Maximum estimated context tokens this hook may add to the request.
    pub max_tokens: u32,
    /// Configured upper bound for candidates inspected by this hook.
    pub max_candidates: u32,
    /// Configured upper bound for each rendered projection in UTF-8 bytes.
    pub max_projection_bytes: u32,
}

/// A versioned proposal prepared by the memory agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryProposal {
    pub proposal_id: Uuid,
    pub run_id: RunId,
    /// Namespace/tenant for which this proposal was produced.
    pub namespace: String,
    /// Runtime revision observed when the proposal was created.
    pub base_revision: Revision,
    pub memory_revision: MemoryRevision,
    pub policy_revision: PolicyRevision,
    /// Candidates are ordered from highest to lowest provider rank.
    pub candidates: Vec<MemoryCandidate>,
    pub created_at_ms: u64,
    pub expires_at_ms: Option<u64>,
    /// Optional retrieval facts preserved for runtime and benchmark audit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_summary: Option<MemoryRetrievalSummary>,
}

impl MemoryProposal {
    /// Checks the proposal and selects ranked candidates within the hook
    /// budget. A stale or empty proposal becomes an explicit skip decision;
    /// malformed provider output remains an error for observability.
    pub fn evaluate(
        &self,
        context: &MemoryHookContext,
    ) -> Result<InjectionDecision, MemoryProposalError> {
        let skip = |reason| InjectionDecision::Skip {
            proposal_id: self.proposal_id,
            reason,
        };

        if self.run_id != context.run_id {
            return Ok(skip(InjectionSkipReason::RunMismatch));
        }
        if self.namespace != context.namespace {
            return Ok(skip(InjectionSkipReason::NamespaceMismatch));
        }
        if self.base_revision != context.current_revision {
            return Ok(skip(InjectionSkipReason::RuntimeRevisionMismatch));
        }
        if self.memory_revision != context.memory_revision {
            return Ok(skip(InjectionSkipReason::MemoryRevisionMismatch));
        }
        if self.policy_revision != context.policy_revision {
            return Ok(skip(InjectionSkipReason::PolicyRevisionMismatch));
        }
        if self
            .expires_at_ms
            .is_some_and(|expiry| context.now_ms >= expiry)
        {
            return Ok(skip(InjectionSkipReason::ProposalExpired));
        }

        let candidate_limit = context.max_candidates.min(MAX_HOOK_CANDIDATES) as usize;
        if self.candidates.len() > candidate_limit {
            return Err(MemoryProposalError::TooManyCandidates {
                actual: self.candidates.len(),
                maximum: candidate_limit,
            });
        }
        let projection_limit = context.max_projection_bytes.min(MAX_PROJECTION_BYTES) as usize;

        let mut selected = Vec::new();
        let mut total_token_cost = 0_u32;

        for candidate in &self.candidates {
            candidate.validate()?;
            let projection_bytes =
                candidate
                    .projection
                    .content
                    .len()
                    .saturating_add(serialized_evidence_bytes(
                        &candidate.projection.evidence_refs,
                    ));
            if projection_bytes > projection_limit {
                return Err(MemoryProposalError::ProjectionTooLarge {
                    candidate_id: candidate.candidate_id,
                    actual: projection_bytes,
                    maximum: projection_limit,
                });
            }
            if candidate.is_expired(context.now_ms) {
                continue;
            }

            let Some(next_total) = total_token_cost.checked_add(candidate.projection.token_cost)
            else {
                continue;
            };
            if next_total > context.max_tokens {
                continue;
            }

            total_token_cost = next_total;
            selected.push(MemoryInjection {
                candidate_id: candidate.candidate_id,
                projection: candidate.projection.clone(),
            });
        }

        if selected.is_empty() {
            return Ok(skip(InjectionSkipReason::NoEligibleCandidate));
        }

        Ok(InjectionDecision::Inject {
            proposal_id: self.proposal_id,
            injections: selected,
            total_token_cost,
        })
    }
}

/// Content accepted by the runtime for one model request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryInjection {
    pub candidate_id: Uuid,
    pub projection: MemoryProjection,
}

/// The final decision returned by the synchronous memory hook.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InjectionDecision {
    Inject {
        proposal_id: Uuid,
        injections: Vec<MemoryInjection>,
        total_token_cost: u32,
    },
    Skip {
        proposal_id: Uuid,
        reason: InjectionSkipReason,
    },
}

/// A normal control-flow outcome that should be measured but does not fail the
/// main agent request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InjectionSkipReason {
    RunMismatch,
    NamespaceMismatch,
    RuntimeRevisionMismatch,
    MemoryRevisionMismatch,
    PolicyRevisionMismatch,
    ProposalExpired,
    NoEligibleCandidate,
}

/// Malformed output from a memory provider or memory-agent adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryProposalError {
    InvalidScore(Uuid),
    InvalidConfidence(Uuid),
    EmptyProjection(Uuid),
    EmptyProvenance(Uuid),
    InvalidTokenCost {
        candidate_id: Uuid,
        token_cost: u32,
        minimum: u32,
        maximum: u32,
    },
    TooManyCandidates {
        actual: usize,
        maximum: usize,
    },
    ProjectionTooLarge {
        candidate_id: Uuid,
        actual: usize,
        maximum: usize,
    },
}

impl fmt::Display for MemoryProposalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScore(id) => write!(formatter, "memory candidate {id} has invalid score"),
            Self::InvalidConfidence(id) => {
                write!(formatter, "memory candidate {id} has invalid confidence")
            }
            Self::EmptyProjection(id) => {
                write!(formatter, "memory candidate {id} has empty projection")
            }
            Self::EmptyProvenance(id) => {
                write!(formatter, "memory candidate {id} has empty provenance")
            }
            Self::InvalidTokenCost {
                candidate_id,
                token_cost,
                minimum,
                maximum,
            } => write!(
                formatter,
                "memory candidate {candidate_id} has token cost {token_cost}, expected {minimum}..={maximum}"
            ),
            Self::TooManyCandidates { actual, maximum } => write!(
                formatter,
                "memory proposal contains {actual} candidates, maximum is {maximum}"
            ),
            Self::ProjectionTooLarge {
                candidate_id,
                actual,
                maximum,
            } => write!(
                formatter,
                "memory candidate {candidate_id} projection is {actual} bytes, maximum is {maximum}"
            ),
        }
    }
}

impl Error for MemoryProposalError {}

fn approximate_projection_token_cost(content: &str, evidence_refs: &[serde_json::Value]) -> u32 {
    let evidence_bytes = serialized_evidence_bytes(evidence_refs);
    let chars = content.chars().count().saturating_add(evidence_bytes);
    chars
        .saturating_add(APPROX_CHARS_PER_TOKEN - 1)
        .checked_div(APPROX_CHARS_PER_TOKEN)
        .unwrap_or(usize::MAX)
        .max(1)
        .min(u32::MAX as usize) as u32
}

fn serialized_evidence_bytes(evidence_refs: &[serde_json::Value]) -> usize {
    if evidence_refs.is_empty() {
        0
    } else {
        serde_json::to_vec(evidence_refs).map_or(usize::MAX, |bytes| bytes.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(token_cost: u32, expires_at_ms: Option<u64>) -> MemoryCandidate {
        MemoryCandidate {
            candidate_id: Uuid::new_v4(),
            projection: MemoryProjection {
                projection_id: Uuid::new_v4(),
                source_memory_id: Uuid::new_v4(),
                source_version: 3,
                content: "检查上一次的生命周期修复".to_string(),
                evidence_refs: Vec::new(),
                provenance: "semantic-provider:task-42".to_string(),
                token_cost,
            },
            score: 0.9,
            confidence: 0.8,
            expires_at_ms,
        }
    }

    fn context() -> MemoryHookContext {
        MemoryHookContext {
            run_id: RunId::new(),
            turn_id: None,
            namespace: "project-a".to_owned(),
            current_revision: Revision::new(7),
            memory_revision: MemoryRevision::new(4),
            policy_revision: PolicyRevision::new(2),
            now_ms: 1_000,
            max_tokens: 20,
            max_candidates: 8,
            max_projection_bytes: 4_096,
        }
    }

    #[test]
    fn stale_proposal_is_skipped_before_injection() {
        let context = context();
        let proposal = MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: context.namespace.clone(),
            base_revision: Revision::new(6),
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: vec![candidate(2, None)],
            created_at_ms: 900,
            expires_at_ms: None,
            retrieval_summary: None,
        };

        assert_eq!(
            proposal.evaluate(&context).unwrap(),
            InjectionDecision::Skip {
                proposal_id: proposal.proposal_id,
                reason: InjectionSkipReason::RuntimeRevisionMismatch,
            }
        );
    }

    #[test]
    fn evaluation_skips_expired_candidates_and_respects_token_budget() {
        let mut context = context();
        context.max_tokens = 5;
        let proposal = MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: context.namespace.clone(),
            base_revision: context.current_revision,
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: vec![candidate(3, Some(1_000)), candidate(3, None)],
            created_at_ms: 900,
            expires_at_ms: Some(2_000),
            retrieval_summary: None,
        };

        let decision = proposal.evaluate(&context).unwrap();
        let InjectionDecision::Inject {
            injections,
            total_token_cost,
            ..
        } = decision
        else {
            panic!("expected one eligible candidate");
        };
        assert_eq!(injections.len(), 1);
        assert_eq!(total_token_cost, 3);
    }

    #[test]
    fn malformed_scores_are_rejected() {
        let context = context();
        let mut item = candidate(1, None);
        item.score = f32::NAN;
        let proposal = MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: context.namespace.clone(),
            base_revision: context.current_revision,
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: vec![item],
            created_at_ms: 900,
            expires_at_ms: None,
            retrieval_summary: None,
        };

        assert!(matches!(
            proposal.evaluate(&context),
            Err(MemoryProposalError::InvalidScore(_))
        ));
    }

    #[test]
    fn serialized_proposal_keeps_revisions_and_provenance() {
        let context = context();
        let proposal = MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: context.namespace.clone(),
            base_revision: context.current_revision,
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: vec![candidate(2, None)],
            created_at_ms: 900,
            expires_at_ms: Some(2_000),
            retrieval_summary: None,
        };

        let json = serde_json::to_value(proposal).expect("proposal should serialize");
        assert_eq!(json["base_revision"], 7);
        assert_eq!(json["memory_revision"], 4);
        assert_eq!(json["policy_revision"], 2);
        assert_eq!(
            json["candidates"][0]["projection"]["provenance"],
            "semantic-provider:task-42"
        );
    }

    #[test]
    fn older_proposal_without_retrieval_summary_deserializes() {
        let context = context();
        let proposal = MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: context.namespace,
            base_revision: context.current_revision,
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: Vec::new(),
            created_at_ms: 900,
            expires_at_ms: None,
            retrieval_summary: None,
        };
        let mut json = serde_json::to_value(proposal).expect("proposal should serialize");
        json.as_object_mut()
            .expect("proposal should be an object")
            .remove("retrieval_summary");

        let restored: MemoryProposal =
            serde_json::from_value(json).expect("legacy proposal should deserialize");
        assert_eq!(restored.retrieval_summary, None);
    }

    #[test]
    fn retrieval_summary_keeps_hit_evidence_and_query_identity() {
        let summary = MemoryRetrievalSummary {
            query_digest: Some("sha256:query".to_owned()),
            snapshot_id: Some("snapshot:7".to_owned()),
            hit_count: 1,
            truncated: false,
            hits: vec![MemoryRetrievalHit {
                item_id: Uuid::from_u128(7),
                source_record_ids: vec![Uuid::from_u128(9)],
                version: 2,
                method_id: "video.timeline".to_owned(),
                rank: 1,
                snapshot_id: Some("snapshot:7".to_owned()),
                evidence_ids: vec!["evidence:frame-12".to_owned()],
            }],
            provider_failures: Vec::new(),
        };

        let restored: MemoryRetrievalSummary = serde_json::from_value(
            serde_json::to_value(&summary).expect("retrieval summary should serialize"),
        )
        .expect("retrieval summary should deserialize");

        assert_eq!(restored, summary);
        assert_eq!(restored.hits[0].source_record_ids, vec![Uuid::from_u128(9)]);
    }

    #[test]
    fn legacy_retrieval_summary_defaults_new_trace_fields() {
        let legacy = serde_json::json!({
            "hit_count": 2,
            "truncated": true,
            "provider_failures": []
        });

        let restored: MemoryRetrievalSummary =
            serde_json::from_value(legacy).expect("legacy summary should deserialize");

        assert_eq!(restored.query_digest, None);
        assert_eq!(restored.snapshot_id, None);
        assert!(restored.hits.is_empty());
        assert_eq!(restored.hit_count, 2);
        assert!(restored.truncated);
    }

    #[test]
    fn older_context_without_turn_id_deserializes_as_unscoped() {
        let context = context();
        let mut json = serde_json::to_value(context).expect("context should serialize");
        json.as_object_mut()
            .expect("context should be an object")
            .remove("turn_id");

        let restored: MemoryHookContext =
            serde_json::from_value(json).expect("legacy context should deserialize");
        assert_eq!(restored.turn_id, None);
    }

    #[test]
    fn cross_namespace_proposal_is_skipped() {
        let context = context();
        let proposal = MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: "project-b".to_owned(),
            base_revision: context.current_revision,
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: vec![candidate(3, None)],
            created_at_ms: 900,
            expires_at_ms: None,
            retrieval_summary: None,
        };

        assert_eq!(
            proposal.evaluate(&context).unwrap(),
            InjectionDecision::Skip {
                proposal_id: proposal.proposal_id,
                reason: InjectionSkipReason::NamespaceMismatch,
            }
        );
    }

    #[test]
    fn candidate_count_is_bounded_before_selection() {
        let mut context = context();
        context.max_candidates = 1;
        let proposal = MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: context.namespace.clone(),
            base_revision: context.current_revision,
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: vec![candidate(3, None), candidate(3, None)],
            created_at_ms: 900,
            expires_at_ms: None,
            retrieval_summary: None,
        };

        assert!(matches!(
            proposal.evaluate(&context),
            Err(MemoryProposalError::TooManyCandidates {
                actual: 2,
                maximum: 1,
            })
        ));
    }

    #[test]
    fn configured_candidate_limit_cannot_raise_runtime_hard_cap() {
        let mut context = context();
        context.max_candidates = MAX_HOOK_CANDIDATES + 1;
        let candidates = (0..=MAX_HOOK_CANDIDATES)
            .map(|_| candidate(3, None))
            .collect();
        let proposal = MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: context.namespace.clone(),
            base_revision: context.current_revision,
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates,
            created_at_ms: 900,
            expires_at_ms: None,
            retrieval_summary: None,
        };

        assert!(matches!(
            proposal.evaluate(&context),
            Err(MemoryProposalError::TooManyCandidates {
                actual: 129,
                maximum: 128,
            })
        ));
    }

    #[test]
    fn projection_size_is_bounded_before_injection() {
        let mut context = context();
        context.max_projection_bytes = 8;
        let proposal = MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: context.namespace.clone(),
            base_revision: context.current_revision,
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: vec![candidate(3, None)],
            created_at_ms: 900,
            expires_at_ms: None,
            retrieval_summary: None,
        };

        assert!(matches!(
            proposal.evaluate(&context),
            Err(MemoryProposalError::ProjectionTooLarge {
                actual: 36,
                maximum: 8,
                ..
            })
        ));
    }

    #[test]
    fn forged_zero_token_cost_is_rejected() {
        let context = context();
        let proposal = MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: context.namespace.clone(),
            base_revision: context.current_revision,
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: vec![candidate(0, None)],
            created_at_ms: 900,
            expires_at_ms: None,
            retrieval_summary: None,
        };

        assert!(matches!(
            proposal.evaluate(&context),
            Err(MemoryProposalError::InvalidTokenCost { token_cost: 0, .. })
        ));
    }

    #[test]
    fn forged_token_cost_above_content_size_is_rejected() {
        let context = context();
        let proposal = MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: context.namespace.clone(),
            base_revision: context.current_revision,
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: vec![candidate(37, None)],
            created_at_ms: 900,
            expires_at_ms: None,
            retrieval_summary: None,
        };

        assert!(matches!(
            proposal.evaluate(&context),
            Err(MemoryProposalError::InvalidTokenCost { token_cost: 37, .. })
        ));
    }
}
