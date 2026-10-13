//! Bounded, task-local set of prepared memory candidates.

use std::{cmp::Ordering, error::Error, fmt};

use pi_agent_runtime::{MemoryCandidate, MemoryHookContext, MemoryProposal};
use uuid::Uuid;

/// A candidate's lifecycle inside the memory agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateStatus {
    Eligible,
    Expired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkingSetConfig {
    pub max_candidates: usize,
}

impl Default for WorkingSetConfig {
    fn default() -> Self {
        Self { max_candidates: 16 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkingSetEntry {
    pub candidate: MemoryCandidate,
    pub status: CandidateStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkingSetError {
    InvalidCapacity,
    InvalidCandidate(Uuid),
}

impl fmt::Display for WorkingSetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCapacity => write!(formatter, "working set capacity must be positive"),
            Self::InvalidCandidate(id) => write!(formatter, "invalid memory candidate: {id}"),
        }
    }
}

impl Error for WorkingSetError {}

/// Keeps only the highest utility candidate for each source memory version.
#[derive(Debug, Clone)]
pub struct WorkingSet {
    config: WorkingSetConfig,
    entries: Vec<WorkingSetEntry>,
}

impl WorkingSet {
    pub fn new(config: WorkingSetConfig) -> Result<Self, WorkingSetError> {
        if config.max_candidates == 0 {
            return Err(WorkingSetError::InvalidCapacity);
        }
        Ok(Self {
            config,
            entries: Vec::new(),
        })
    }

    pub fn config(&self) -> WorkingSetConfig {
        self.config
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[WorkingSetEntry] {
        &self.entries
    }

    /// Inserts or replaces a source memory version, then keeps the best N.
    pub fn upsert(
        &mut self,
        candidate: MemoryCandidate,
        now_ms: u64,
    ) -> Result<(), WorkingSetError> {
        validate_candidate(&candidate)?;
        let key = (
            candidate.projection.source_memory_id,
            candidate.projection.source_version,
        );
        if let Some(index) = self.entries.iter().position(|entry| {
            (
                entry.candidate.projection.source_memory_id,
                entry.candidate.projection.source_version,
            ) == key
        }) {
            if self.entries[index].status == CandidateStatus::Expired
                || compare_candidates(&candidate, &self.entries[index].candidate) == Ordering::Less
            {
                self.entries[index] = WorkingSetEntry {
                    status: status_at(&candidate, now_ms),
                    candidate,
                };
            }
        } else {
            self.entries.push(WorkingSetEntry {
                status: status_at(&candidate, now_ms),
                candidate,
            });
        }
        self.entries
            .sort_by(|left, right| compare_candidates(&left.candidate, &right.candidate));
        self.entries.truncate(self.config.max_candidates);
        Ok(())
    }

    pub fn remove_expired(&mut self, now_ms: u64) {
        self.entries.retain(|entry| {
            entry
                .candidate
                .expires_at_ms
                .is_none_or(|expiry| now_ms < expiry)
        });
        for entry in &mut self.entries {
            entry.status = status_at(&entry.candidate, now_ms);
        }
    }

    pub fn proposal(
        &mut self,
        context: &MemoryHookContext,
        expires_at_ms: Option<u64>,
    ) -> MemoryProposal {
        self.remove_expired(context.now_ms);
        MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: context.run_id,
            namespace: context.namespace.clone(),
            base_revision: context.current_revision,
            memory_revision: context.memory_revision,
            policy_revision: context.policy_revision,
            candidates: self
                .entries
                .iter()
                .filter(|entry| entry.status == CandidateStatus::Eligible)
                .map(|entry| entry.candidate.clone())
                .collect(),
            created_at_ms: context.now_ms,
            expires_at_ms,
        }
    }
}

fn validate_candidate(candidate: &MemoryCandidate) -> Result<(), WorkingSetError> {
    if !candidate.score.is_finite()
        || !(0.0..=1.0).contains(&candidate.score)
        || !candidate.confidence.is_finite()
        || !(0.0..=1.0).contains(&candidate.confidence)
        || candidate.projection.content.trim().is_empty()
        || candidate.projection.provenance.trim().is_empty()
    {
        return Err(WorkingSetError::InvalidCandidate(candidate.candidate_id));
    }
    Ok(())
}

fn status_at(candidate: &MemoryCandidate, now_ms: u64) -> CandidateStatus {
    if candidate
        .expires_at_ms
        .is_some_and(|expiry| now_ms >= expiry)
    {
        CandidateStatus::Expired
    } else {
        CandidateStatus::Eligible
    }
}

fn compare_candidates(left: &MemoryCandidate, right: &MemoryCandidate) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| right.confidence.total_cmp(&left.confidence))
        .then_with(|| {
            left.projection
                .source_memory_id
                .cmp(&right.projection.source_memory_id)
        })
        .then_with(|| {
            left.projection
                .source_version
                .cmp(&right.projection.source_version)
        })
        .then_with(|| left.candidate_id.cmp(&right.candidate_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_runtime::{MemoryProjection, MemoryRevision, PolicyRevision, Revision, RunId};

    fn candidate(id: u128, score: f32) -> MemoryCandidate {
        MemoryCandidate {
            candidate_id: Uuid::from_u128(id),
            projection: MemoryProjection {
                projection_id: Uuid::from_u128(id + 100),
                source_memory_id: Uuid::from_u128(id),
                source_version: 1,
                content: format!("memory-{id}"),
                provenance: "test-provider:record".to_string(),
                token_cost: 2,
            },
            score,
            confidence: 0.8,
            expires_at_ms: None,
        }
    }

    #[test]
    fn capacity_and_order_are_deterministic() {
        let mut set = WorkingSet::new(WorkingSetConfig { max_candidates: 2 }).unwrap();
        set.upsert(candidate(1, 0.2), 10).unwrap();
        set.upsert(candidate(2, 0.9), 10).unwrap();
        set.upsert(candidate(3, 0.5), 10).unwrap();
        assert_eq!(set.len(), 2);
        assert_eq!(
            set.entries()[0].candidate.projection.source_memory_id,
            Uuid::from_u128(2)
        );
        assert_eq!(
            set.entries()[1].candidate.projection.source_memory_id,
            Uuid::from_u128(3)
        );
    }

    #[test]
    fn proposal_drops_expired_candidates() {
        let mut set = WorkingSet::new(WorkingSetConfig::default()).unwrap();
        let mut item = candidate(1, 0.8);
        item.expires_at_ms = Some(10);
        set.upsert(item, 0).unwrap();
        let proposal = set.proposal(
            &MemoryHookContext {
                run_id: RunId::new(),
                namespace: "test".to_string(),
                current_revision: Revision::new(1),
                memory_revision: MemoryRevision::new(2),
                policy_revision: PolicyRevision::new(3),
                now_ms: 10,
                max_tokens: 100,
                max_candidates: 16,
                max_projection_bytes: 1024,
            },
            None,
        );
        assert!(proposal.candidates.is_empty());
    }
}
