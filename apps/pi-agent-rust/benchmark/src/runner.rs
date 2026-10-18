//! Run-scoped bindings shared by benchmark execution code.
//!
//! This module does not choose a retrieval strategy or call a model. It keeps
//! the experiment identity in one place so every memory and trace operation
//! uses the same namespace and visibility cutoff.

use std::path::Path;

use pi_agent_memory::{MemoryQuery, OperationContext, StorageContext};
use uuid::Uuid;

use crate::{
    read_all, read_builds, read_run, BenchmarkRun, BuildTrace, JsonlTraceWriter, MetricValue,
    StepTrace, TraceRecord, TraceStoreError,
};

/// The shared boundary for one concrete benchmark execution.
#[derive(Debug)]
pub struct BenchmarkRunContext {
    run: BenchmarkRun,
    isolated_namespace: String,
    operation_run_id: Uuid,
    trace: JsonlTraceWriter,
    completed_builds: Vec<BuildTrace>,
    completed_steps: Vec<StepTrace>,
    finished_at_ms: Option<i64>,
}

impl BenchmarkRunContext {
    /// Opens or creates the trace and binds it to one benchmark definition.
    ///
    /// `operation_run_id` is the UUID used by memory providers. It is passed
    /// in explicitly because the trace's external string `run_id` may come
    /// from a dataset or an orchestration system and is not required to be a
    /// UUID.
    pub fn open(
        path: impl AsRef<Path>,
        mut run: BenchmarkRun,
        operation_run_id: Uuid,
    ) -> Result<Self, TraceStoreError> {
        let path = path.as_ref();
        let existed = path.exists();
        if !existed {
            run.manifest.validate_new_artifact_policy()?;
        }
        let mut trace = JsonlTraceWriter::open(path, run.run_id.clone())?;
        let mut completed_builds = Vec::new();
        let mut completed_steps = Vec::new();
        let mut finished_at_ms = None;

        if existed {
            let records = read_all(path)?;
            if records.is_empty() {
                return Err(TraceStoreError::InvalidManifest(
                    "existing trace has no complete run record".to_string(),
                ));
            }
            let previous = read_run(path, &run.run_id)?;
            completed_builds = read_builds(path, &run.run_id)?;
            if previous.run.manifest != run.manifest
                || previous.run.started_at_ms != run.started_at_ms
            {
                return Err(TraceStoreError::InvalidManifest(
                    "existing trace does not match the benchmark run".to_string(),
                ));
            }
            if previous.run.operation_run_id != Some(operation_run_id) {
                return Err(TraceStoreError::InvalidManifest(
                    "existing trace operation_run_id does not match".to_string(),
                ));
            }
            run = previous.run;
            run.finished_at_ms = previous.finished_at_ms;
            completed_steps = previous.steps;
            finished_at_ms = previous.finished_at_ms;
        } else {
            if run
                .operation_run_id
                .is_some_and(|existing| existing != operation_run_id)
            {
                return Err(TraceStoreError::InvalidManifest(
                    "new trace operation_run_id does not match the requested id".to_string(),
                ));
            }
            run.operation_run_id = Some(operation_run_id);
            trace.append_run(&run)?;
        }

        let isolated_namespace = run.isolated_namespace();

        Ok(Self {
            run,
            isolated_namespace,
            operation_run_id,
            trace,
            completed_builds,
            completed_steps,
            finished_at_ms,
        })
    }

    /// Opens a run only when its model and prompt metadata satisfy the
    /// reproducibility contract.
    pub fn open_reproducible(
        path: impl AsRef<Path>,
        run: BenchmarkRun,
        operation_run_id: Uuid,
    ) -> Result<Self, TraceStoreError> {
        run.manifest.validate_reproducibility_config()?;
        Self::open(path, run, operation_run_id)
    }

    /// Reopens a persisted run and recovers its provider correlation UUID.
    pub fn resume(
        path: impl AsRef<Path>,
        run_id: impl AsRef<str>,
    ) -> Result<Self, TraceStoreError> {
        let run_id = run_id.as_ref();
        let run = read_all(path.as_ref())?
            .into_iter()
            .find_map(|record| match record {
                TraceRecord::Run(run) if run.run_id == run_id => Some(run),
                _ => None,
            })
            .ok_or_else(|| TraceStoreError::RunNotFound(run_id.to_string()))?;
        let operation_run_id = run.operation_run_id.ok_or_else(|| {
            TraceStoreError::InvalidManifest(
                "persisted run has no operation_run_id; it cannot be resumed safely".to_string(),
            )
        })?;
        Self::open(path, run, operation_run_id)
    }

    pub fn run(&self) -> &BenchmarkRun {
        &self.run
    }

    /// Namespace for this concrete run's trace and run-scoped artifacts.
    pub fn namespace(&self) -> &str {
        &self.isolated_namespace
    }

    /// Immutable namespace containing the benchmark dataset projection.
    pub fn dataset_namespace(&self) -> &str {
        &self.run.manifest.namespace
    }

    pub fn operation_run_id(&self) -> Uuid {
        self.operation_run_id
    }

    /// Steps already persisted in this trace, in dataset order.
    ///
    /// A resumed runner uses these records to consume the same prefix from
    /// its dataset source before it executes any new sample. Returning the
    /// slice keeps the resume contract explicit without exposing the writer's
    /// internal state.
    pub fn completed_steps(&self) -> &[StepTrace] {
        &self.completed_steps
    }

    /// Build receipts already persisted in an offline build-then-query trace.
    ///
    /// The slice is ordered by `BuildTrace::build_index`, matching the source
    /// order consumed by the offline runner.
    pub fn completed_builds(&self) -> &[BuildTrace] {
        &self.completed_builds
    }

    /// The next build index accepted by the append-only trace.
    pub fn next_build_index(&self) -> u64 {
        self.completed_builds.len() as u64
    }

    pub fn completed_sample_ids(&self) -> impl Iterator<Item = &str> {
        self.completed_steps
            .iter()
            .map(|step| step.sample_id.as_str())
    }

    /// The next step index accepted by the append-only trace.
    pub fn next_step_index(&self) -> u64 {
        self.completed_steps.len() as u64
    }

    pub fn terminal_seen(&self) -> bool {
        self.completed_steps
            .last()
            .is_some_and(|step| step.terminal)
    }

    pub fn is_complete(&self) -> bool {
        self.finished_at_ms.is_some()
    }

    pub fn storage_context(&self, timeout_ms: Option<u64>) -> StorageContext {
        StorageContext {
            namespace: self.isolated_namespace.clone(),
            run_id: self.operation_run_id,
            snapshot_id: None,
            timeout_ms,
        }
    }

    pub fn operation_context(
        &self,
        session_id: Option<Uuid>,
        episode_id: Option<Uuid>,
        timeout_ms: Option<u64>,
        idempotency_key: Option<&str>,
    ) -> OperationContext {
        OperationContext {
            namespace: self.isolated_namespace.clone(),
            run_id: self.operation_run_id,
            session_id,
            episode_id,
            timeout_ms,
            idempotency_key: idempotency_key.map(|key| self.scoped_idempotency_key(key)),
        }
    }

    /// Builds a read-only memory context for the immutable benchmark dataset.
    /// The operation UUID still identifies the concrete run, while the
    /// namespace points at the shared dataset projection.
    pub fn retrieval_context(
        &self,
        session_id: Option<Uuid>,
        episode_id: Option<Uuid>,
        timeout_ms: Option<u64>,
        idempotency_key: Option<&str>,
    ) -> OperationContext {
        OperationContext {
            namespace: self.dataset_namespace().to_string(),
            run_id: self.operation_run_id,
            session_id,
            episode_id,
            timeout_ms,
            idempotency_key: idempotency_key.map(|key| self.scoped_idempotency_key(key)),
        }
    }

    pub fn bind_query(&self, mut query: MemoryQuery) -> Result<MemoryQuery, TraceStoreError> {
        if !query.namespace.is_empty() && query.namespace != self.isolated_namespace {
            return Err(TraceStoreError::NamespaceMismatch {
                expected: self.isolated_namespace.clone(),
                actual: query.namespace,
            });
        }
        if query.visibility_cutoff_ms.is_some()
            && query.visibility_cutoff_ms != self.run.manifest.visibility_cutoff_ms
        {
            return Err(TraceStoreError::CutoffMismatch);
        }
        query.namespace = self.isolated_namespace.clone();
        query.visibility_cutoff_ms = self.run.manifest.visibility_cutoff_ms;
        Ok(query)
    }

    pub fn bind_dataset_query(
        &self,
        mut query: MemoryQuery,
    ) -> Result<MemoryQuery, TraceStoreError> {
        if !query.namespace.is_empty() && query.namespace != self.dataset_namespace() {
            return Err(TraceStoreError::NamespaceMismatch {
                expected: self.dataset_namespace().to_string(),
                actual: query.namespace,
            });
        }
        if query.visibility_cutoff_ms.is_some()
            && query.visibility_cutoff_ms != self.run.manifest.visibility_cutoff_ms
        {
            return Err(TraceStoreError::CutoffMismatch);
        }
        query.namespace = self.dataset_namespace().to_string();
        query.visibility_cutoff_ms = self.run.manifest.visibility_cutoff_ms;
        Ok(query)
    }

    pub fn scoped_idempotency_key(&self, key: &str) -> String {
        format!("{}/{}", self.isolated_namespace, key)
    }

    pub fn append_step(&mut self, step: &StepTrace) -> Result<(), TraceStoreError> {
        self.trace.append_step(step)?;
        self.completed_steps.push(step.clone());
        Ok(())
    }

    pub fn append_build(&mut self, build: &BuildTrace) -> Result<(), TraceStoreError> {
        self.trace.append_build(build)?;
        self.completed_builds.push(build.clone());
        Ok(())
    }

    pub fn append_metric(&mut self, metric: &MetricValue) -> Result<(), TraceStoreError> {
        self.trace.append_metric(metric)
    }

    pub fn finalize(&mut self, finished_at_ms: i64) -> Result<(), TraceStoreError> {
        self.trace.finalize(finished_at_ms)?;
        self.run.finished_at_ms = Some(finished_at_ms);
        self.finished_at_ms = Some(finished_at_ms);
        Ok(())
    }

    pub fn trace(&self) -> &JsonlTraceWriter {
        &self.trace
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use pi_agent_memory::{ContentRef, MemoryBudget, Modality};
    use tempfile::tempdir;

    use super::*;
    use crate::{
        ArtifactValidationPolicy, BenchmarkManifest, DatasetManifest, JsonlTraceWriter,
        MethodManifest,
    };

    const DATASET_DIGEST: &str =
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn run() -> BenchmarkRun {
        BenchmarkRun::new(
            "run/1",
            BenchmarkManifest::new(
                "retrieval",
                "bench/news",
                MethodManifest {
                    method_id: "projection".into(),
                    version: "1".into(),
                    config: BTreeMap::new(),
                },
                "dataset-1",
                Some(42),
            ),
            10,
        )
    }

    fn run_with_dataset(
        modalities: Vec<Modality>,
        policy: ArtifactValidationPolicy,
    ) -> BenchmarkRun {
        let mut run = run();
        run.manifest
            .bind_dataset(DatasetManifest::new(
                "dataset-1",
                "dataset-1",
                DATASET_DIGEST,
                modalities,
            ))
            .unwrap();
        run.manifest.artifact_validation_policy = policy;
        run
    }

    #[test]
    fn binds_memory_context_to_isolated_run_namespace() {
        let directory = tempdir().unwrap();
        let run = run();
        let operation_run_id = Uuid::new_v4();
        let context = BenchmarkRunContext::open(
            directory.path().join("trace.jsonl"),
            run.clone(),
            operation_run_id,
        )
        .unwrap();

        let operation = context.operation_context(None, None, Some(100), Some("write-1"));
        assert_eq!(operation.namespace, run.isolated_namespace());
        assert_eq!(operation.run_id, operation_run_id);
        assert_eq!(
            operation.idempotency_key.as_deref(),
            Some("bench/news/r72756e2f31/write-1")
        );
        let storage = context.storage_context(Some(100));
        assert_eq!(storage.namespace, run.isolated_namespace());
        assert_eq!(storage.run_id, operation_run_id);

        let query = MemoryQuery {
            input: Some(ContentRef::Text {
                text: "hello".into(),
            }),
            namespace: String::new(),
            scope: None,
            subject: None,
            episode_id: None,
            kinds: Vec::new(),
            modalities: Vec::new(),
            time_range: None,
            visibility_cutoff_ms: None,
            graph: None,
            budget: MemoryBudget::default(),
        };
        let bound = context.bind_query(query).unwrap();
        assert_eq!(bound.namespace, run.isolated_namespace());
        assert_eq!(bound.visibility_cutoff_ms, Some(42));

        let mut mismatched = bound.clone();
        mismatched.namespace = "other-run".into();
        assert!(matches!(
            context.bind_query(mismatched),
            Err(TraceStoreError::NamespaceMismatch { .. })
        ));

        let mut wrong_cutoff = bound;
        wrong_cutoff.visibility_cutoff_ms = Some(43);
        assert!(matches!(
            context.bind_query(wrong_cutoff),
            Err(TraceStoreError::CutoffMismatch)
        ));
    }

    #[test]
    fn reproducible_open_requires_model_seed_and_prompt_metadata() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let operation_run_id = Uuid::new_v4();
        let incomplete_run = run();
        assert!(matches!(
            BenchmarkRunContext::open_reproducible(&path, incomplete_run, operation_run_id),
            Err(TraceStoreError::InvalidManifest(message))
                if message.contains("reproducibility config")
        ));

        let mut run = run();
        run.manifest
            .config
            .insert("model".to_owned(), "test-model".to_owned());
        run.manifest
            .config
            .insert("seed".to_owned(), "0".to_owned());
        run.manifest
            .config
            .insert("prompt_digest".to_owned(), "sha256:test".to_owned());
        run.manifest.seal_config();
        BenchmarkRunContext::open_reproducible(&path, run, operation_run_id)
            .expect("complete reproducibility metadata should be accepted");
    }

    #[test]
    fn reopens_matching_trace_without_appending_a_second_run() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let run = run();
        let operation_run_id = Uuid::new_v4();
        let context = BenchmarkRunContext::open(&path, run.clone(), operation_run_id).unwrap();
        drop(context);

        let reopened = BenchmarkRunContext::resume(&path, "run/1").unwrap();
        assert_eq!(read_all(&path).unwrap().len(), 1);
        assert_eq!(reopened.run().manifest, run.manifest);
        assert_eq!(reopened.run().operation_run_id, Some(operation_run_id));
        drop(reopened);

        let wrong_id = Uuid::new_v4();
        assert!(matches!(
            BenchmarkRunContext::open(&path, run, wrong_id),
            Err(TraceStoreError::InvalidManifest(message))
                if message.contains("operation_run_id")
        ));
    }

    #[test]
    fn new_binary_run_requires_explicit_artifact_policy_before_creating_trace() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let run = run_with_dataset(vec![Modality::Video], ArtifactValidationPolicy::Unspecified);

        let error = BenchmarkRunContext::open(&path, run, Uuid::new_v4()).unwrap_err();
        assert!(matches!(
            error,
            TraceStoreError::InvalidManifest(message)
                if message.contains("artifact_validation_policy")
        ));
        assert!(!path.exists());
    }

    #[test]
    fn explicit_binary_policies_and_inline_only_datasets_are_accepted() {
        for policy in [
            ArtifactValidationPolicy::LocalStore,
            ArtifactValidationPolicy::RemoteService,
            ArtifactValidationPolicy::Unchecked,
        ] {
            let directory = tempdir().unwrap();
            let path = directory.path().join("trace.jsonl");
            BenchmarkRunContext::open(
                &path,
                run_with_dataset(vec![Modality::Video], policy),
                Uuid::new_v4(),
            )
            .unwrap();
        }

        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        BenchmarkRunContext::open(
            &path,
            run_with_dataset(
                vec![Modality::Text, Modality::State],
                ArtifactValidationPolicy::Unspecified,
            ),
            Uuid::new_v4(),
        )
        .unwrap();
    }

    #[test]
    fn existing_binary_trace_without_policy_remains_readable() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let operation_run_id = Uuid::new_v4();
        let mut old_run =
            run_with_dataset(vec![Modality::Video], ArtifactValidationPolicy::Unspecified);
        old_run.operation_run_id = Some(operation_run_id);
        let mut writer = JsonlTraceWriter::open(&path, old_run.run_id.clone()).unwrap();
        writer.append_run(&old_run).unwrap();
        drop(writer);

        let reopened = BenchmarkRunContext::open(&path, old_run, operation_run_id).unwrap();
        assert_eq!(reopened.operation_run_id(), operation_run_id);
        assert_eq!(read_all(&path).unwrap().len(), 1);
    }
}
