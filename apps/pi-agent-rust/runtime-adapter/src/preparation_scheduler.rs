//! Concurrent execution of version-bound preparation plans.
//!
//! The core runtime owns preparation state transitions and durable events. This
//! adapter only supplies the async orchestration needed to run independent
//! workers in parallel while serializing those transitions through one
//! coordinator.

use std::{
    fmt,
    future::Future,
    sync::Arc,
    time::{SystemTime, SystemTimeError, UNIX_EPOCH},
};

use pi_agent_runtime::{
    CoordinatorError, EventLedger, PreparationPlan, PreparationWorkResult, RuntimeCoordinator,
};
use thiserror::Error;
use tokio::{
    sync::{Mutex, OwnedSemaphorePermit, Semaphore},
    task::JoinHandle,
};
use uuid::Uuid;

/// Errors raised while scheduling or committing an async preparation worker.
#[derive(Debug, Error)]
pub enum PreparationSchedulerError {
    #[error("preparation scheduler concurrency must be positive")]
    InvalidConcurrency,
    #[error("preparation coordinator failed: {0}")]
    Coordinator(#[from] CoordinatorError),
    #[error("preparation plan {0} disappeared before worker start")]
    MissingPlan(Uuid),
    #[error("preparation scheduler clock failed: {0}")]
    Clock(#[from] SystemTimeError),
    #[error("preparation scheduler semaphore was closed")]
    SemaphoreClosed,
    #[error("preparation worker task failed: {0}")]
    TaskJoin(String),
    #[error("preparation reconciler failed: {0}")]
    Reconciler(String),
}

/// A submitted preparation that can be awaited independently of other work.
pub struct PreparationTask {
    plan_id: Uuid,
    join: JoinHandle<Result<PreparationWorkResult, PreparationSchedulerError>>,
}

impl PreparationTask {
    /// Returns the plan handled by this task.
    pub const fn plan_id(&self) -> Uuid {
        self.plan_id
    }

    /// Waits for the worker and for its durable runtime state transition.
    pub async fn wait(self) -> Result<PreparationWorkResult, PreparationSchedulerError> {
        self.join
            .await
            .map_err(|error| PreparationSchedulerError::TaskJoin(error.to_string()))?
    }
}

/// Runs preparation workers with a bounded number of concurrent executions.
///
/// The coordinator lock is held only while recording lifecycle events. Worker
/// futures run outside that lock, so a slow memory or artifact provider cannot
/// serialize unrelated preparations.
pub struct PreparationScheduler<L> {
    coordinator: Arc<Mutex<RuntimeCoordinator<L>>>,
    permits: Arc<Semaphore>,
}

impl<L> Clone for PreparationScheduler<L> {
    fn clone(&self) -> Self {
        Self {
            coordinator: Arc::clone(&self.coordinator),
            permits: Arc::clone(&self.permits),
        }
    }
}

impl<L> PreparationScheduler<L> {
    /// Creates a scheduler around one runtime coordinator.
    pub fn new(
        coordinator: RuntimeCoordinator<L>,
        max_concurrency: usize,
    ) -> Result<Self, PreparationSchedulerError> {
        if max_concurrency == 0 {
            return Err(PreparationSchedulerError::InvalidConcurrency);
        }
        Ok(Self {
            coordinator: Arc::new(Mutex::new(coordinator)),
            permits: Arc::new(Semaphore::new(max_concurrency)),
        })
    }

    /// Returns a shared handle for consuming or inspecting prepared plans.
    pub fn coordinator(&self) -> Arc<Mutex<RuntimeCoordinator<L>>> {
        Arc::clone(&self.coordinator)
    }
}

impl<L> PreparationScheduler<L>
where
    L: EventLedger + Send + 'static,
{
    /// Reconciles plans restored from a ledger after a worker-process restart.
    ///
    /// The scheduler cannot infer whether a worker that was already starting
    /// completed outside the process, so the runtime records that state as
    /// `Unknown` and leaves any provider-specific reconciliation to the caller.
    pub async fn recover_interrupted(
        &self,
        now_ms: u64,
    ) -> Result<usize, PreparationSchedulerError> {
        let mut coordinator = self.coordinator.lock().await;
        coordinator
            .recover_interrupted_preparations(now_ms)
            .map_err(PreparationSchedulerError::Coordinator)
    }

    /// Releases ready results that were only held in the previous worker
    /// process and have no durable artifact to resolve after restart.
    pub async fn release_unmaterialized(
        &self,
        now_ms: u64,
    ) -> Result<usize, PreparationSchedulerError> {
        let mut coordinator = self.coordinator.lock().await;
        coordinator
            .release_unmaterialized_preparations(now_ms)
            .map_err(PreparationSchedulerError::Coordinator)
    }

    /// Marks a read-only preparation consumed after its result is used by an
    /// adapter without producing an external action proposal.
    pub async fn consume_read_only(
        &self,
        plan_id: Uuid,
        now_ms: u64,
    ) -> Result<(), PreparationSchedulerError> {
        let mut coordinator = self.coordinator.lock().await;
        coordinator
            .consume_read_only_preparation(plan_id, now_ms)
            .map(|_| ())
            .map_err(PreparationSchedulerError::Coordinator)
    }

    /// Lets a provider-specific reconciler close plans whose worker outcome is
    /// unknown after a restart. Provider work runs without the coordinator
    /// lock; only the durable state transition is serialized.
    pub async fn reconcile_unknown<F, Fut, E>(
        &self,
        now_ms: u64,
        reconciler: F,
    ) -> Result<usize, PreparationSchedulerError>
    where
        F: Fn(PreparationPlan) -> Fut + Send + Sync,
        Fut: Future<Output = Result<PreparationWorkResult, E>> + Send,
        E: fmt::Display,
    {
        let plans = {
            let coordinator = self.coordinator.lock().await;
            coordinator
                .preparations()
                .plans()
                .filter(|plan| plan.status == pi_agent_runtime::PreparationStatus::Unknown)
                .cloned()
                .collect::<Vec<_>>()
        };
        let mut reconciled = 0;
        for plan in plans {
            let result = reconciler(plan.clone())
                .await
                .map_err(|error| PreparationSchedulerError::Reconciler(error.to_string()))?;
            let mut coordinator = self.coordinator.lock().await;
            coordinator.reconcile_preparation(plan.plan_id, result, now_ms)?;
            reconciled += 1;
        }
        Ok(reconciled)
    }

    /// Persists a plan, then starts its worker as soon as a concurrency permit
    /// is available. Worker errors become durable `Failed` preparation
    /// outcomes; coordinator errors are returned from [`PreparationTask::wait`].
    pub async fn submit<F, Fut, E>(
        &self,
        plan: PreparationPlan,
        submitted_at_ms: u64,
        worker: F,
    ) -> Result<PreparationTask, PreparationSchedulerError>
    where
        F: FnOnce(PreparationPlan) -> Fut + Send + 'static,
        Fut: Future<Output = Result<PreparationWorkResult, E>> + Send + 'static,
        E: fmt::Display + Send + 'static,
    {
        let plan_id = {
            let mut coordinator = self.coordinator.lock().await;
            coordinator.submit_preparation(plan, submitted_at_ms)?
        };
        let coordinator = Arc::clone(&self.coordinator);
        let permits = Arc::clone(&self.permits);
        let join = tokio::spawn(async move {
            let _permit = acquire_permit(permits).await?;
            let started_at_ms = unix_time_ms()?;
            let plan = {
                let mut coordinator = coordinator.lock().await;
                coordinator.start_preparation(plan_id, started_at_ms)?;
                coordinator
                    .preparations()
                    .get(plan_id)
                    .cloned()
                    .ok_or(PreparationSchedulerError::MissingPlan(plan_id))?
            };

            let result = worker(plan)
                .await
                .unwrap_or_else(|error| PreparationWorkResult::Failed {
                    error: error.to_string(),
                });
            let completed_at_ms = unix_time_ms()?;
            let mut coordinator = coordinator.lock().await;
            match &result {
                PreparationWorkResult::Ready {
                    artifacts,
                    evidence,
                } => coordinator.mark_preparation_ready_with_artifacts_and_evidence(
                    plan_id,
                    artifacts.clone(),
                    evidence.clone(),
                    completed_at_ms,
                )?,
                PreparationWorkResult::Failed { error } => coordinator
                    .fail_preparation_with_reason(plan_id, error.clone(), completed_at_ms)?,
                PreparationWorkResult::Unknown { reason } => coordinator
                    .mark_preparation_unknown_with_reason(
                        plan_id,
                        reason.clone(),
                        completed_at_ms,
                    )?,
            }
            Ok(result)
        });
        Ok(PreparationTask { plan_id, join })
    }
}

async fn acquire_permit(
    permits: Arc<Semaphore>,
) -> Result<OwnedSemaphorePermit, PreparationSchedulerError> {
    permits
        .acquire_owned()
        .await
        .map_err(|_| PreparationSchedulerError::SemaphoreClosed)
}

fn unix_time_ms() -> Result<u64, SystemTimeError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .min(u64::MAX as u128) as u64)
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use pi_agent_runtime::{
        canonical_json_digest, ActionId, AdmissionPolicy, ArtifactRef, Budget, InMemoryLedger,
        JsonlLedger, PreparationEffectClass, PreparationStatus, PreparationWorkResult,
        ResourceScope, Revision, RunId, RuntimeCoordinator,
    };
    use serde_json::json;
    use tempfile::tempdir;
    use tokio::{
        sync::Barrier,
        time::{sleep, timeout, Duration},
    };
    use uuid::Uuid;

    use super::PreparationScheduler;

    fn plan(
        run_id: RunId,
        action_id: ActionId,
        now_ms: u64,
        name: &str,
    ) -> pi_agent_runtime::PreparationPlan {
        pi_agent_runtime::PreparationPlan::new(
            run_id,
            action_id,
            Revision::INITIAL,
            name,
            canonical_json_digest(&json!({"name": name})),
            PreparationEffectClass::ReadOnly,
            ResourceScope::new([format!("memory:{name}")]),
            Budget {
                wall_time_ms: Some(1_000),
                tokens: Some(128),
                tool_calls: Some(1),
            },
            0.9,
            now_ms,
            now_ms + 10_000,
        )
        .expect("test preparation plan should be valid")
    }

    #[tokio::test]
    async fn independent_workers_run_in_parallel_and_commit_ready_state() {
        let now_ms = super::unix_time_ms().expect("test clock should be valid");
        let run_id = RunId::new();
        let scheduler = PreparationScheduler::new(
            RuntimeCoordinator::new(
                run_id,
                AdmissionPolicy::deny_all(),
                InMemoryLedger::default(),
            ),
            2,
        )
        .expect("positive concurrency should be accepted");
        let barrier = Arc::new(Barrier::new(2));
        let active = Arc::new(AtomicUsize::new(0));
        let maximum_active = Arc::new(AtomicUsize::new(0));

        let worker = |barrier: Arc<Barrier>,
                      active: Arc<AtomicUsize>,
                      maximum_active: Arc<AtomicUsize>| async move {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            maximum_active.fetch_max(current, Ordering::SeqCst);
            barrier.wait().await;
            sleep(Duration::from_millis(10)).await;
            active.fetch_sub(1, Ordering::SeqCst);
            Ok::<_, &'static str>(PreparationWorkResult::Ready {
                artifacts: Vec::new(),
                evidence: Vec::new(),
            })
        };

        let first = scheduler
            .submit(
                plan(run_id, ActionId::new(), now_ms, "memory:first"),
                now_ms,
                {
                    let barrier = Arc::clone(&barrier);
                    let active = Arc::clone(&active);
                    let maximum_active = Arc::clone(&maximum_active);
                    move |_| worker(barrier, active, maximum_active)
                },
            )
            .await
            .expect("first preparation should be submitted");
        let second = scheduler
            .submit(
                plan(run_id, ActionId::new(), now_ms, "memory:second"),
                now_ms,
                {
                    let barrier = Arc::clone(&barrier);
                    let active = Arc::clone(&active);
                    let maximum_active = Arc::clone(&maximum_active);
                    move |_| worker(barrier, active, maximum_active)
                },
            )
            .await
            .expect("second preparation should be submitted");

        let results = timeout(Duration::from_secs(2), async {
            (first.wait().await, second.wait().await)
        })
        .await
        .expect("parallel workers should not deadlock");
        assert!(results.0.is_ok());
        assert!(results.1.is_ok());
        assert_eq!(maximum_active.load(Ordering::SeqCst), 2);

        let coordinator = scheduler.coordinator();
        let coordinator = coordinator.lock().await;
        assert!(coordinator
            .preparations()
            .plans()
            .all(|plan| plan.status == PreparationStatus::Ready));
    }

    #[tokio::test]
    async fn interrupted_plans_are_reconciled_after_ledger_recovery() {
        let run_id = RunId::new();
        let now_ms = 1_000;
        let directory = tempdir().expect("temporary ledger directory should be created");
        let ledger_path = directory.path().join("preparations.jsonl");
        let mut coordinator = RuntimeCoordinator::new(
            run_id,
            AdmissionPolicy::deny_all(),
            JsonlLedger::open(&ledger_path).expect("JSONL ledger should open"),
        );
        let planned_id = coordinator
            .submit_preparation(
                plan(run_id, ActionId::new(), now_ms, "memory:planned"),
                now_ms,
            )
            .expect("planned preparation should be durable");
        let starting_id = coordinator
            .submit_preparation(
                plan(run_id, ActionId::new(), now_ms, "memory:starting"),
                now_ms,
            )
            .expect("starting preparation should be durable");
        coordinator
            .start_preparation(starting_id, now_ms + 1)
            .expect("starting preparation should be durable");

        let (ledger, policy, _, _) = coordinator.into_parts();
        let recovered = RuntimeCoordinator::recover(run_id, policy, ledger)
            .expect("runtime coordinator should recover preparation snapshots");
        let scheduler = PreparationScheduler::new(recovered, 1)
            .expect("recovered coordinator should create a scheduler");
        assert_eq!(
            scheduler
                .recover_interrupted(now_ms + 2)
                .await
                .expect("interrupted preparations should be reconciled"),
            2
        );

        {
            let coordinator = scheduler.coordinator();
            let coordinator = coordinator.lock().await;
            assert_eq!(
                coordinator.preparations().get(planned_id).unwrap().status,
                PreparationStatus::Released
            );
            assert_eq!(
                coordinator.preparations().get(starting_id).unwrap().status,
                PreparationStatus::Unknown
            );
        }
        assert_eq!(
            scheduler
                .reconcile_unknown(now_ms + 3, |plan| async move {
                    assert_eq!(plan.predicted_action, "memory:starting");
                    Ok::<_, &'static str>(PreparationWorkResult::Ready {
                        artifacts: Vec::new(),
                        evidence: Vec::new(),
                    })
                })
                .await
                .expect("provider evidence should reconcile the unknown plan"),
            1
        );
        {
            let coordinator = scheduler.coordinator();
            let coordinator = coordinator.lock().await;
            assert_eq!(
                coordinator.preparations().get(starting_id).unwrap().status,
                PreparationStatus::Ready
            );
        }
        drop(scheduler);

        let ledger = JsonlLedger::open(&ledger_path).expect("JSONL ledger should reopen");
        let recovered = RuntimeCoordinator::recover(run_id, AdmissionPolicy::deny_all(), ledger)
            .expect("reconciled preparation states should survive restart");
        assert_eq!(
            recovered.preparations().get(planned_id).unwrap().status,
            PreparationStatus::Released
        );
        assert_eq!(
            recovered.preparations().get(starting_id).unwrap().status,
            PreparationStatus::Ready
        );
    }

    #[tokio::test]
    async fn read_only_preparation_can_be_consumed_without_action_proposal() {
        let run_id = RunId::new();
        let now_ms = super::unix_time_ms().expect("test clock should be valid");
        let scheduler = PreparationScheduler::new(
            RuntimeCoordinator::new(
                run_id,
                AdmissionPolicy::deny_all(),
                InMemoryLedger::default(),
            ),
            1,
        )
        .expect("positive concurrency should be accepted");
        let task = scheduler
            .submit(
                plan(run_id, ActionId::new(), now_ms, "memory:read-only"),
                now_ms,
                |_| async {
                    Ok::<_, &'static str>(PreparationWorkResult::Ready {
                        artifacts: Vec::new(),
                        evidence: Vec::new(),
                    })
                },
            )
            .await
            .expect("read-only preparation should be submitted");
        let plan_id = task.plan_id();
        task.wait().await.expect("worker should become ready");
        scheduler
            .consume_read_only(plan_id, now_ms + 1)
            .await
            .expect("read-only result should be consumable");

        let coordinator = scheduler.coordinator();
        let coordinator = coordinator.lock().await;
        assert_eq!(
            coordinator.preparations().get(plan_id).unwrap().status,
            PreparationStatus::Consumed
        );
    }

    #[tokio::test]
    async fn unmaterialized_ready_preparation_is_released_after_restart() {
        let run_id = RunId::new();
        let now_ms = super::unix_time_ms().expect("test clock should be valid");
        let scheduler = PreparationScheduler::new(
            RuntimeCoordinator::new(
                run_id,
                AdmissionPolicy::deny_all(),
                InMemoryLedger::default(),
            ),
            1,
        )
        .expect("positive concurrency should be accepted");
        let plan_id = {
            let coordinator = scheduler.coordinator();
            let mut coordinator = coordinator.lock().await;
            let plan_id = coordinator
                .submit_preparation(
                    plan(run_id, ActionId::new(), now_ms, "memory:restart"),
                    now_ms,
                )
                .expect("preparation should be submitted");
            coordinator
                .start_preparation(plan_id, now_ms + 1)
                .expect("preparation should start");
            coordinator
                .mark_preparation_ready_with_artifacts_and_evidence(
                    plan_id,
                    Vec::new(),
                    Vec::new(),
                    now_ms + 2,
                )
                .expect("preparation should become ready");
            plan_id
        };

        assert_eq!(
            scheduler
                .release_unmaterialized(now_ms + 3)
                .await
                .expect("unmaterialized result should be released"),
            1
        );
        let coordinator = scheduler.coordinator();
        let coordinator = coordinator.lock().await;
        assert_eq!(
            coordinator.preparations().get(plan_id).unwrap().status,
            PreparationStatus::Released
        );
    }

    #[tokio::test]
    async fn artifact_backed_ready_preparation_is_retained_after_restart() {
        let run_id = RunId::new();
        let now_ms = super::unix_time_ms().expect("test clock should be valid");
        let scheduler = PreparationScheduler::new(
            RuntimeCoordinator::new(
                run_id,
                AdmissionPolicy::deny_all(),
                InMemoryLedger::default(),
            ),
            1,
        )
        .expect("positive concurrency should be accepted");
        let plan_id = {
            let coordinator = scheduler.coordinator();
            let mut coordinator = coordinator.lock().await;
            let plan_id = coordinator
                .submit_preparation(
                    plan(run_id, ActionId::new(), now_ms, "memory:artifact"),
                    now_ms,
                )
                .expect("preparation should be submitted");
            coordinator
                .start_preparation(plan_id, now_ms + 1)
                .expect("preparation should start");
            coordinator
                .mark_preparation_ready_with_artifacts_and_evidence(
                    plan_id,
                    vec![ArtifactRef {
                        artifact_id: Uuid::new_v4(),
                        uri: "cas://memory/projection.json".to_owned(),
                        media_type: "application/json".to_owned(),
                        sha256: "sha256:projection".to_owned(),
                        bytes: 32,
                        codec: None,
                        schema: Some("memory-projection-v1".to_owned()),
                        stream_key: None,
                        storage_revision: Some("store-1".to_owned()),
                    }],
                    Vec::new(),
                    now_ms + 2,
                )
                .expect("artifact-backed preparation should become ready");
            plan_id
        };

        assert_eq!(
            scheduler
                .release_unmaterialized(now_ms + 3)
                .await
                .expect("restart cleanup should succeed"),
            0
        );
        let coordinator = scheduler.coordinator();
        let coordinator = coordinator.lock().await;
        assert_eq!(
            coordinator.preparations().get(plan_id).unwrap().status,
            PreparationStatus::Ready
        );
    }

    #[test]
    fn zero_concurrency_is_rejected() {
        let result = PreparationScheduler::new(
            RuntimeCoordinator::new(
                RunId::new(),
                AdmissionPolicy::deny_all(),
                InMemoryLedger::default(),
            ),
            0,
        );
        assert!(result.is_err());
    }
}
