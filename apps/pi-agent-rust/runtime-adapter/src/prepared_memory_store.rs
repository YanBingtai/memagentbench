//! Durable snapshots for speculative memory proposals.
//!
//! The preparation ledger records lifecycle state, while this small projection
//! stores the proposal payload needed to resume a read-only memory handoff.
//! They remain separate so the runtime core does not depend on a concrete
//! memory serialization format.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use pi_agent_runtime::{
    EventDraft, EventDraftError, EventKind, EventSequence, JsonlLedger, JsonlLedgerError,
    MemoryProposal, Revision, RunId,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Mutex;
use uuid::Uuid;

const STORED_EVENT: &str = "memory_prepared";
const REMOVED_EVENT: &str = "memory_consumed";

/// Errors raised while persisting or restoring a prepared memory proposal.
#[derive(Debug, Error)]
pub enum PreparedMemoryStoreError {
    #[error("prepared memory ledger failed: {0}")]
    Ledger(#[from] JsonlLedgerError),
    #[error("prepared memory event kind is invalid: {0}")]
    EventKind(#[from] EventDraftError),
    #[error("prepared memory payload is invalid: {0}")]
    Payload(#[from] serde_json::Error),
    #[error("prepared memory proposal conflicts with plan {plan_id}")]
    Conflict { plan_id: Uuid },
}

/// A process-safe handle over a durable prepared-proposal ledger.
#[derive(Clone)]
pub struct PreparedMemoryStore {
    ledger: Arc<Mutex<JsonlLedger>>,
}

impl PreparedMemoryStore {
    /// Opens or creates a JSONL proposal ledger.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, PreparedMemoryStoreError> {
        Ok(Self {
            ledger: Arc::new(Mutex::new(JsonlLedger::open(path)?)),
        })
    }

    /// Persists a proposal before the corresponding preparation becomes ready.
    pub async fn put(
        &self,
        plan_id: Uuid,
        proposal: &MemoryProposal,
        revision: Revision,
        now_ms: u64,
    ) -> Result<(), PreparedMemoryStoreError> {
        if let Some(existing) = self.get(proposal.run_id, plan_id).await? {
            if existing != *proposal {
                return Err(PreparedMemoryStoreError::Conflict { plan_id });
            }
            return Ok(());
        }

        let payload = serde_json::to_value(StoredPayload {
            plan_id,
            proposal: proposal.clone(),
        })?;
        let draft = EventDraft::new(
            proposal.run_id,
            revision,
            EventKind::new(STORED_EVENT)?,
            payload,
            now_ms,
        )
        .with_idempotency_key(format!(
            "prepared-memory:{}:{}:stored",
            proposal.run_id.as_uuid(),
            plan_id
        ))?;
        self.append(draft).await
    }

    /// Returns all proposals persisted for one run, replaying consume tombstones.
    pub async fn proposals_for_run(
        &self,
        run_id: RunId,
    ) -> Result<BTreeMap<Uuid, MemoryProposal>, PreparedMemoryStoreError> {
        let ledger = self.ledger.lock().await;
        let events = ledger.scan_after(EventSequence::INITIAL)?;
        let mut proposals = BTreeMap::new();
        for event in events {
            if event.draft.run_id != run_id {
                continue;
            }
            match event.draft.kind.as_str() {
                STORED_EVENT => {
                    let stored: StoredPayload = serde_json::from_value(event.draft.payload)?;
                    if stored.proposal.run_id != run_id {
                        return Err(PreparedMemoryStoreError::Conflict {
                            plan_id: stored.plan_id,
                        });
                    }
                    proposals.insert(stored.plan_id, stored.proposal);
                }
                REMOVED_EVENT => {
                    let removed: RemovedPayload = serde_json::from_value(event.draft.payload)?;
                    proposals.remove(&removed.plan_id);
                }
                _ => {}
            }
        }
        Ok(proposals)
    }

    /// Returns one proposal if it is still present in the durable projection.
    pub async fn get(
        &self,
        run_id: RunId,
        plan_id: Uuid,
    ) -> Result<Option<MemoryProposal>, PreparedMemoryStoreError> {
        Ok(self.proposals_for_run(run_id).await?.remove(&plan_id))
    }

    /// Appends an idempotent consume tombstone for one proposal.
    pub async fn remove(
        &self,
        run_id: RunId,
        plan_id: Uuid,
        revision: Revision,
        now_ms: u64,
    ) -> Result<(), PreparedMemoryStoreError> {
        if self.get(run_id, plan_id).await?.is_none() {
            return Ok(());
        }
        let payload = serde_json::to_value(RemovedPayload { plan_id })?;
        let draft = EventDraft::new(
            run_id,
            revision,
            EventKind::new(REMOVED_EVENT)?,
            payload,
            now_ms,
        )
        .with_idempotency_key(format!(
            "prepared-memory:{}:{}:consumed",
            run_id.as_uuid(),
            plan_id
        ))?;
        self.append(draft).await
    }

    /// Removes every proposal belonging to a completed or abandoned run.
    pub async fn clear_run(
        &self,
        run_id: RunId,
        revision: Revision,
        now_ms: u64,
    ) -> Result<usize, PreparedMemoryStoreError> {
        let plan_ids = self
            .proposals_for_run(run_id)
            .await?
            .into_keys()
            .collect::<Vec<_>>();
        for plan_id in &plan_ids {
            self.remove(run_id, *plan_id, revision, now_ms).await?;
        }
        Ok(plan_ids.len())
    }

    /// Closes the underlying ledger and releases its cooperative lock.
    pub async fn close(&self) -> Result<(), PreparedMemoryStoreError> {
        let mut ledger = self.ledger.lock().await;
        ledger.close()?;
        Ok(())
    }

    async fn append(&self, draft: EventDraft) -> Result<(), PreparedMemoryStoreError> {
        let mut ledger = self.ledger.lock().await;
        let receipt = ledger.append(vec![draft])?;
        ledger.flush(receipt.last_sequence)?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredPayload {
    plan_id: Uuid,
    proposal: MemoryProposal,
}

#[derive(Debug, Serialize, Deserialize)]
struct RemovedPayload {
    plan_id: Uuid,
}

#[cfg(test)]
mod tests {
    use pi_agent_runtime::{MemoryProposal, MemoryRevision, PolicyRevision, Revision};
    use tempfile::tempdir;
    use uuid::Uuid;

    use super::PreparedMemoryStore;

    fn empty_proposal(
        run_id: pi_agent_runtime::RunId,
        revision: Revision,
        now_ms: u64,
    ) -> MemoryProposal {
        MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id,
            namespace: "test".to_owned(),
            base_revision: revision,
            memory_revision: MemoryRevision::INITIAL,
            policy_revision: PolicyRevision::INITIAL,
            candidates: Vec::new(),
            created_at_ms: now_ms,
            expires_at_ms: Some(now_ms + 1_000),
            retrieval_summary: None,
        }
    }

    #[tokio::test]
    async fn proposal_survives_reopen_and_is_removed_by_tombstone() {
        let directory = tempdir().expect("temporary directory should be created");
        let path = directory.path().join("prepared.jsonl");
        let run_id = pi_agent_runtime::RunId::new();
        let plan_id = uuid::Uuid::new_v4();
        let proposal = empty_proposal(run_id, Revision::INITIAL, 1_000);
        let store = PreparedMemoryStore::open(&path).expect("store should open");
        store
            .put(plan_id, &proposal, Revision::INITIAL, 1_000)
            .await
            .expect("proposal should be persisted");
        assert_eq!(
            store
                .get(run_id, plan_id)
                .await
                .expect("proposal should be readable"),
            Some(proposal.clone())
        );
        store.close().await.expect("store should close");

        let reopened = PreparedMemoryStore::open(&path).expect("store should reopen");
        assert_eq!(
            reopened
                .get(run_id, plan_id)
                .await
                .expect("reopened proposal should be readable"),
            Some(proposal)
        );
        reopened
            .remove(run_id, plan_id, Revision::INITIAL, 1_001)
            .await
            .expect("consume tombstone should be durable");
        assert!(reopened
            .get(run_id, plan_id)
            .await
            .expect("removed proposal should be readable")
            .is_none());
        reopened.close().await.expect("reopened store should close");
    }
}
