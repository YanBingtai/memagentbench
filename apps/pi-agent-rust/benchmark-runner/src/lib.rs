//! Execution of memory retrieval and agent steps for benchmark contracts.
//!
//! The runner owns orchestration, not retrieval policy. A memory method still
//! decides how to rank and represent evidence; this crate gives it the frozen
//! query boundary, delegates agent work, and records the result in the common
//! trace format.

mod jsonl_dataset;
mod longmemeval;

pub use jsonl_dataset::{
    JsonlContinualDataset, JsonlDataset, JsonlOfflineDataset, JsonlSource,
    DEFAULT_MAX_JSONL_LINE_BYTES,
};
pub use longmemeval::{
    LongMemEvalCase, LongMemEvalCorpus, LongMemEvalError, LongMemEvalMessage, LongMemEvalPair,
    LongMemEvalReference, LongMemEvalSession, LongMemEvalSource,
};

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt::{Display, Formatter},
    future::Future,
    pin::Pin,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use pi_agent_benchmark::{
    ActionTrace, AgentObservation, BenchmarkLifecycle, BenchmarkRunContext, BuildTrace,
    IngestTrace, ObservedHit, QueryTrace, RewardTrace, RuntimeStepLink, StepTelemetry, StepTrace,
    TerminalReason, TraceErrorEntry, TraceErrorStage, TraceStoreError,
};
use pi_agent_memory::{
    EvidenceLocator, IngestReceipt, IngestRequest, MemoryCapabilities, MemoryError, MemoryFuture,
    MemoryItem, MemoryKind, MemoryManifest, MemoryMethod, MemoryQuery, MemoryRelation, Modality,
    OperationContext, RecordIngestRequest, RetrieveResponse,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

/// A dataset sample supplied to one retrieval step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetSample {
    pub sample_id: String,
    pub observation: AgentObservation,
    pub query: MemoryQuery,
    pub session_id: Option<Uuid>,
    pub episode_id: Option<Uuid>,
    pub terminal: bool,
    pub terminal_reason: Option<TerminalReason>,
}

impl DatasetSample {
    fn validate(&self) -> Result<(), RunnerError> {
        if self.sample_id.trim().is_empty() {
            return Err(RunnerError::InvalidSample(
                "sample_id must be non-empty".to_string(),
            ));
        }
        if self.terminal != self.terminal_reason.is_some() {
            return Err(RunnerError::InvalidSample(
                "terminal samples must have exactly one terminal reason".to_string(),
            ));
        }
        u32::try_from(self.query.budget.max_hits).map_err(|_| {
            RunnerError::InvalidSample("query max_hits exceeds trace representation".to_string())
        })?;
        if self.query.budget.max_hits == 0 {
            return Err(RunnerError::InvalidSample(
                "query max_hits must be greater than zero".to_string(),
            ));
        }
        if self.query.budget.max_latency_ms == Some(0) {
            return Err(RunnerError::InvalidSample(
                "query max_latency_ms must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }

    /// Identifies the source sample independently of one concrete run.
    /// Runtime namespace binding is persisted in the trace and provider
    /// request, but must not change the identity used to compare methods.
    fn identity_digest(&self, _bound_query: &MemoryQuery) -> Result<String, RunnerError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| RunnerError::InvalidSample(format!("sample digest: {error}")))?;
        Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
    }
}

/// A synchronous source of reproducible dataset samples.
pub trait DatasetSource {
    fn next_sample(&mut self) -> Result<Option<DatasetSample>, DatasetError>;

    /// Content identity required in the run manifest, when the source can
    /// calculate it. JSONL sources use the SHA-256 of their original bytes.
    fn dataset_digest(&self) -> Option<&str> {
        None
    }
}

/// A small source useful for local datasets and conformance tests.
#[derive(Debug, Clone)]
pub struct VecDataset {
    samples: std::vec::IntoIter<DatasetSample>,
}

impl VecDataset {
    pub fn new(samples: Vec<DatasetSample>) -> Self {
        Self {
            samples: samples.into_iter(),
        }
    }
}

impl DatasetSource for VecDataset {
    fn next_sample(&mut self) -> Result<Option<DatasetSample>, DatasetError> {
        Ok(self.samples.next())
    }
}

/// A dataset sample for continual-memory evaluation.
///
/// The observation and query describe the agent turn.  The ingest request is
/// the data accepted by the memory method before that query runs, so a
/// benchmark can measure online memory without changing the retrieval-only
/// [`DatasetSample`] contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContinualDatasetSample {
    pub sample: DatasetSample,
    pub ingest: IngestRequest,
}

impl ContinualDatasetSample {
    fn validate(&self) -> Result<(), RunnerError> {
        self.sample.validate()?;
        if self.ingest.idempotency_key.trim().is_empty() {
            return Err(RunnerError::InvalidSample(
                "ingest idempotency_key must be non-empty".to_string(),
            ));
        }
        Ok(())
    }

    fn identity_digest(
        &self,
        _bound_query: &MemoryQuery,
        _bound_ingest: &IngestRequest,
    ) -> Result<String, RunnerError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| RunnerError::InvalidSample(format!("sample digest: {error}")))?;
        Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
    }
}

/// A synchronous source of continual-memory samples.
pub trait ContinualDatasetSource {
    fn next_sample(&mut self) -> Result<Option<ContinualDatasetSample>, DatasetError>;

    /// Content identity required in the run manifest, if available.
    fn dataset_digest(&self) -> Option<&str> {
        None
    }
}

/// One sample for a two-phase offline build-then-query benchmark.
///
/// Every complete record is admitted before the first query is executed. The
/// sample remains available after a source rewind so the second phase can
/// replay its observation and query without submitting the records again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OfflineDatasetSample {
    pub sample: DatasetSample,
    pub build: RecordIngestRequest,
}

impl OfflineDatasetSample {
    fn validate(&self) -> Result<(), RunnerError> {
        self.sample.validate()?;
        if self.build.idempotency_key.trim().is_empty() {
            return Err(RunnerError::InvalidSample(
                "build idempotency_key must be non-empty".to_string(),
            ));
        }
        if self.build.records.is_empty() && self.build.relations.is_empty() {
            return Err(RunnerError::InvalidSample(
                "build must contain records or relations".to_string(),
            ));
        }
        Ok(())
    }

    fn identity_digest(
        &self,
        _bound_query: &MemoryQuery,
        _bound_build: &RecordIngestRequest,
    ) -> Result<String, RunnerError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| RunnerError::InvalidSample(format!("sample digest: {error}")))?;
        Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
    }
}

/// A replayable source used by the offline build and query phases.
pub trait ReplayableOfflineDatasetSource {
    fn next_sample(&mut self) -> Result<Option<OfflineDatasetSample>, DatasetError>;

    fn rewind(&mut self) -> Result<(), DatasetError>;

    fn dataset_digest(&self) -> Option<&str> {
        None
    }
}

/// In-memory replayable source useful for tests and small fixtures.
#[derive(Debug, Clone)]
pub struct VecOfflineDataset {
    samples: Vec<OfflineDatasetSample>,
    index: usize,
}

impl VecOfflineDataset {
    pub fn new(samples: Vec<OfflineDatasetSample>) -> Self {
        Self { samples, index: 0 }
    }
}

impl ReplayableOfflineDatasetSource for VecOfflineDataset {
    fn next_sample(&mut self) -> Result<Option<OfflineDatasetSample>, DatasetError> {
        let sample = self.samples.get(self.index).cloned();
        if sample.is_some() {
            self.index += 1;
        }
        Ok(sample)
    }

    fn rewind(&mut self) -> Result<(), DatasetError> {
        self.index = 0;
        Ok(())
    }
}

/// Inputs presented to an artifact validator immediately before a memory
/// operation. The validator may inspect both the online ingest payload and
/// the retrieval query without owning either value.
#[derive(Debug, Clone, Copy)]
pub struct ArtifactValidationInput<'a> {
    pub ingest: Option<&'a IngestRequest>,
    pub record_ingest: Option<&'a RecordIngestRequest>,
    pub query: &'a MemoryQuery,
}

/// A provider-specific artifact admission check.
///
/// The runner owns when validation happens and how a failure is persisted;
/// adapters own how Blob references are resolved (local store, HTTP service,
/// or another backend). Keeping this callback in the runner avoids coupling
/// the benchmark lifecycle to one artifact implementation.
pub trait ArtifactValidator: Send {
    fn validate<'a>(
        &'a mut self,
        input: ArtifactValidationInput<'a>,
        context: OperationContext,
    ) -> ArtifactValidationFuture<'a>;
}

impl<T> ArtifactValidator for Box<T>
where
    T: ArtifactValidator + ?Sized,
{
    fn validate<'a>(
        &'a mut self,
        input: ArtifactValidationInput<'a>,
        context: OperationContext,
    ) -> ArtifactValidationFuture<'a> {
        (**self).validate(input, context)
    }
}

pub type ArtifactValidationFuture<'a> =
    Pin<Box<dyn Future<Output = Result<(), ArtifactValidationError>> + Send + 'a>>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("artifact validation failed: {message}")]
pub struct ArtifactValidationError {
    message: String,
}

impl ArtifactValidationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// A small continual source useful for local datasets and conformance tests.
#[derive(Debug, Clone)]
pub struct VecContinualDataset {
    samples: std::vec::IntoIter<ContinualDatasetSample>,
}

impl VecContinualDataset {
    pub fn new(samples: Vec<ContinualDatasetSample>) -> Self {
        Self {
            samples: samples.into_iter(),
        }
    }
}

impl ContinualDatasetSource for VecContinualDataset {
    fn next_sample(&mut self) -> Result<Option<ContinualDatasetSample>, DatasetError> {
        Ok(self.samples.next())
    }
}

/// The result of one bounded memory retrieval.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalOutcome {
    pub query: MemoryQuery,
    pub response: Option<RetrieveResponse>,
    pub hits: Vec<ObservedHit>,
    pub latency_ms: u64,
    pub errors: Vec<TraceErrorEntry>,
}

/// Input handed to the agent after a memory method returns.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentStepInput {
    pub sample_id: String,
    pub observation: AgentObservation,
    pub session_id: Option<Uuid>,
    pub episode_id: Option<Uuid>,
    pub terminal: bool,
    pub terminal_reason: Option<TerminalReason>,
    pub retrieval: RetrievalOutcome,
}

/// Output produced by one agent step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentStepOutput {
    pub answer: Option<String>,
    pub action: Option<ActionTrace>,
    pub reward: Option<RewardTrace>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub errors: Vec<TraceErrorEntry>,
    /// Optional correlation with the runtime ledger for integrated adapters.
    pub runtime_link: Option<RuntimeStepLink>,
}

/// A model, tool loop, or embodied controller can implement this boundary
/// without making the benchmark crate depend on the root agent binary.
pub type AgentFuture<'a> =
    Pin<Box<dyn Future<Output = Result<AgentStepOutput, AgentExecutionError>> + Send + 'a>>;

pub trait AgentStepExecutor: Send {
    fn execute<'a>(&'a mut self, input: AgentStepInput) -> AgentFuture<'a>;
}

/// Structured failure returned by an agent executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentExecutionError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
    /// A partially completed action captured before the terminal error.
    pub action: Option<Box<ActionTrace>>,
    /// Structured errors emitted before the executor reached its terminal
    /// failure. The runner keeps these alongside the final agent error so a
    /// failed step preserves every observed execution stage.
    pub trace_errors: Vec<TraceErrorEntry>,
    /// Correlation captured even when the agent fails after starting a run.
    pub runtime_link: Option<RuntimeStepLink>,
}

impl AgentExecutionError {
    pub fn new(code: impl Into<String>, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            retryable,
            action: None,
            trace_errors: Vec::new(),
            runtime_link: None,
        }
    }

    /// Attach structured stage errors observed before the terminal failure.
    pub fn with_trace_errors(mut self, trace_errors: Vec<TraceErrorEntry>) -> Self {
        self.trace_errors = trace_errors;
        self
    }

    /// Attach the action observed before the executor failed.
    pub fn with_action(mut self, action: Option<ActionTrace>) -> Self {
        self.action = action.map(Box::new);
        self
    }

    pub fn with_runtime_link(mut self, runtime_link: Option<RuntimeStepLink>) -> Self {
        self.runtime_link = runtime_link;
        self
    }
}

impl Display for AgentExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl Error for AgentExecutionError {}

/// Errors produced while reading a dataset source.
#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("dataset source failed: {0}")]
    Source(String),
}

/// Errors that prevent a trace from being a valid benchmark execution.
#[derive(Debug, Error)]
pub enum RunnerError {
    #[error(transparent)]
    Dataset(#[from] DatasetError),
    #[error(transparent)]
    Trace(#[from] TraceStoreError),
    #[error("invalid dataset sample: {0}")]
    InvalidSample(String),
    #[error("invalid memory response: {0}")]
    InvalidResponse(String),
    #[error("dataset source has a content digest but manifest.dataset_digest is missing")]
    DatasetDigestMissing,
    #[error(
        "manifest requires dataset digest {expected}, but the dataset source did not provide one"
    )]
    DatasetDigestUnavailable { expected: String },
    #[error("dataset digest does not match the manifest: expected {expected}, got {actual}")]
    DatasetDigestMismatch { expected: String, actual: String },
    #[error(
        "memory method manifest mismatch: expected {expected_id}@{expected_version}, got {actual_id}@{actual_version}"
    )]
    MethodManifestMismatch {
        expected_id: String,
        expected_version: String,
        actual_id: String,
        actual_version: String,
    },
    #[error(
        "benchmark manifest requires visibility cutoff {cutoff_ms}, but memory method {method_id}@{version} does not support visibility cutoffs"
    )]
    VisibilityCutoffUnsupported {
        method_id: String,
        version: String,
        cutoff_ms: u64,
    },
    #[error("memory method {method_id}@{version} does not support {capability}")]
    CapabilityUnsupported {
        method_id: String,
        version: String,
        capability: String,
    },
    #[error("dataset ended without a terminal sample")]
    MissingTerminalSample,
    #[error("benchmark trace is already complete")]
    TraceAlreadyComplete,
    #[error("benchmark trace already contains a terminal step")]
    TraceAlreadyTerminal,
    #[error("resumed dataset does not match persisted sample order: expected {expected:?}, got {actual:?}")]
    ResumeMismatch { expected: String, actual: String },
    #[error(
        "resumed sample {sample_id:?} changed: expected digest {expected_digest}, got {actual_digest}"
    )]
    ResumeSampleMismatch {
        sample_id: String,
        expected_digest: String,
        actual_digest: String,
    },
    #[error("persisted sample {sample_id:?} has no identity digest and cannot be resumed safely")]
    ResumeSampleDigestMissing { sample_id: String },
    #[error("clock returned a timestamp before unix epoch")]
    ClockBeforeEpoch,
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
}

/// Result of one benchmark run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSummary {
    pub run_id: String,
    pub step_count: u64,
    pub ingest_error_count: u64,
    pub retrieval_error_count: u64,
    pub agent_error_count: u64,
}

/// Runs one memory method against a dataset and appends a replayable trace.
pub struct RetrievalRunner<M, D> {
    method: M,
    dataset: D,
    context: BenchmarkRunContext,
    artifact_validator: Option<Box<dyn ArtifactValidator>>,
}

/// Explicit no-memory baseline for fair benchmark comparisons.
///
/// The baseline still participates in the same runner, query binding, timeout
/// handling, and trace format as a real method. It simply returns no hits and
/// treats online writes as accepted no-ops, so continual metrics measure the
/// value added by a memory implementation rather than a failed write path.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoMemoryMethod;

impl MemoryMethod for NoMemoryMethod {
    fn manifest(&self) -> MemoryManifest {
        MemoryManifest {
            method_id: "no-memory".to_string(),
            version: "1".to_string(),
            schema_version: 1,
            capabilities: MemoryCapabilities {
                modalities: vec![
                    Modality::Text,
                    Modality::Image,
                    Modality::Audio,
                    Modality::Video,
                    Modality::State,
                    Modality::Action,
                    Modality::Trajectory,
                    Modality::Mixed,
                ],
                memory_kinds: vec![
                    MemoryKind::Episodic,
                    MemoryKind::Semantic,
                    MemoryKind::Procedural,
                    MemoryKind::Working,
                ],
                supports_content_search: true,
                supports_time_filter: true,
                supports_visibility_cutoff: true,
                supports_graph: false,
                supports_cross_modal: false,
                supports_update: false,
                supports_delete: false,
                supports_batch: true,
                supports_record_ingest: true,
                supports_record_get: false,
                deterministic: true,
                persistent: false,
            },
        }
    }

    fn ingest<'a>(
        &'a self,
        _request: IngestRequest,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        Box::pin(async {
            Ok(IngestReceipt {
                item_ids: Vec::new(),
                relation_ids: Vec::new(),
                deduplicated: false,
                snapshot_id: None,
                telemetry: None,
            })
        })
    }

    fn ingest_records<'a>(
        &'a self,
        _request: RecordIngestRequest,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        Box::pin(async {
            Ok(IngestReceipt {
                item_ids: Vec::new(),
                relation_ids: Vec::new(),
                deduplicated: false,
                snapshot_id: None,
                telemetry: None,
            })
        })
    }

    fn retrieve<'a>(
        &'a self,
        _query: MemoryQuery,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
        Box::pin(async {
            Ok(RetrieveResponse {
                query_id: Uuid::nil(),
                hits: Vec::new(),
                snapshot_id: None,
                latency_ms: 0,
                truncated: false,
                telemetry: None,
            })
        })
    }

    fn get<'a>(
        &'a self,
        _item_id: Uuid,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
        Box::pin(async { Err(MemoryError::UnsupportedCapability("get".to_string())) })
    }
}

impl<M, D> RetrievalRunner<M, D>
where
    M: MemoryMethod,
    D: DatasetSource,
{
    pub fn new(method: M, dataset: D, context: BenchmarkRunContext) -> Self {
        Self {
            method,
            dataset,
            context,
            artifact_validator: None,
        }
    }

    /// Attach an adapter-owned artifact validator to each retrieval step.
    pub fn with_artifact_validator<V>(mut self, validator: V) -> Self
    where
        V: ArtifactValidator + 'static,
    {
        self.artifact_validator = Some(Box::new(validator));
        self
    }

    pub fn context(&self) -> &BenchmarkRunContext {
        &self.context
    }

    pub async fn run(self) -> Result<RunSummary, RunnerError> {
        self.run_with_agent(NoopAgent).await
    }

    /// Runs retrieval and delegates every step, including retrieval failures,
    /// to an agent executor. Agent output is recorded in the same step trace.
    pub async fn run_with_agent<A>(mut self, mut agent: A) -> Result<RunSummary, RunnerError>
    where
        A: AgentStepExecutor,
    {
        let expected_method = &self.context.run().manifest.method;
        let actual_method = self.method.manifest();
        if actual_method.method_id != expected_method.method_id
            || actual_method.version != expected_method.version
        {
            return Err(RunnerError::MethodManifestMismatch {
                expected_id: expected_method.method_id.clone(),
                expected_version: expected_method.version.clone(),
                actual_id: actual_method.method_id,
                actual_version: actual_method.version,
            });
        }
        if let Some(cutoff_ms) = self.context.run().manifest.visibility_cutoff_ms {
            if !actual_method.capabilities.supports_visibility_cutoff {
                return Err(RunnerError::VisibilityCutoffUnsupported {
                    method_id: actual_method.method_id,
                    version: actual_method.version,
                    cutoff_ms,
                });
            }
        }

        if self.context.is_complete() {
            return Err(RunnerError::TraceAlreadyComplete);
        }
        if self.context.terminal_seen() {
            return Err(RunnerError::TraceAlreadyTerminal);
        }

        validate_dataset_digest(
            self.context.run().manifest.dataset_digest.as_deref(),
            self.dataset.dataset_digest(),
        )?;
        let persisted_steps = self.context.completed_steps().to_vec();
        let mut persisted_index = 0_usize;
        let mut step_index = self.context.next_step_index();
        let mut retrieval_error_count = self
            .context
            .completed_steps()
            .iter()
            .flat_map(|step| step.errors.iter())
            .filter(|error| error.stage == TraceErrorStage::Retrieval)
            .count() as u64;
        let mut agent_error_count = self
            .context
            .completed_steps()
            .iter()
            .flat_map(|step| step.errors.iter())
            .filter(|error| matches!(error.stage, TraceErrorStage::Agent | TraceErrorStage::Tool))
            .count() as u64;
        let mut terminal_seen = false;
        let mut seen_sample_ids = BTreeSet::new();
        for step in &persisted_steps {
            if !seen_sample_ids.insert(step.sample_id.clone()) {
                return Err(RunnerError::InvalidSample(format!(
                    "persisted sample_id {:?} appears more than once",
                    step.sample_id
                )));
            }
        }

        while let Some(sample) = self.dataset.next_sample()? {
            sample.validate()?;
            let query = self.context.bind_dataset_query(sample.query.clone())?;
            validate_query_capabilities(
                &query,
                &actual_method.capabilities,
                &actual_method.method_id,
                &actual_method.version,
            )?;
            let sample_digest = sample.identity_digest(&query)?;
            if let Some(expected) = persisted_steps.get(persisted_index) {
                if sample.sample_id != expected.sample_id {
                    return Err(RunnerError::ResumeMismatch {
                        expected: expected.sample_id.clone(),
                        actual: sample.sample_id,
                    });
                }
                let Some(expected_digest) = expected.sample_digest.as_deref() else {
                    return Err(RunnerError::ResumeSampleDigestMissing {
                        sample_id: sample.sample_id,
                    });
                };
                if expected_digest != sample_digest {
                    return Err(RunnerError::ResumeSampleMismatch {
                        sample_id: sample.sample_id,
                        expected_digest: expected_digest.to_owned(),
                        actual_digest: sample_digest,
                    });
                }
                persisted_index += 1;
                continue;
            }
            if !seen_sample_ids.insert(sample.sample_id.clone()) {
                return Err(RunnerError::InvalidSample(format!(
                    "sample_id {:?} appears more than once",
                    sample.sample_id
                )));
            }
            let query_trace = QueryTrace {
                namespace: self.context.namespace().to_string(),
                dataset_namespace: Some(self.context.dataset_namespace().to_string()),
                query_digest: Some(query_digest(&query)?),
                snapshot_id: None,
                input_digest: query_input_digest(query.input.as_ref())?,
                visibility_cutoff_ms: query.visibility_cutoff_ms,
                requested_hits: query.budget.max_hits as u32,
                retrieval_truncated: false,
                telemetry: None,
            };
            let max_hits = query.budget.max_hits;
            let max_evidence_bytes = query.budget.max_evidence_bytes;
            let operation_key = format!("retrieve/{}", sample.sample_id);
            let operation = self.context.retrieval_context(
                sample.session_id,
                sample.episode_id,
                query.budget.max_latency_ms,
                Some(&operation_key),
            );
            let agent_query = query.clone();
            let agent_observation = sample.observation.clone();
            let started = std::time::Instant::now();
            let response = if let Some(validator) = self.artifact_validator.as_mut() {
                let validation_operation = self.context.retrieval_context(
                    sample.session_id,
                    sample.episode_id,
                    query.budget.max_latency_ms,
                    Some(&format!("artifact/{}", sample.sample_id)),
                );
                match validate_artifacts_with_timeout(
                    validator.as_mut(),
                    ArtifactValidationInput {
                        ingest: None,
                        record_ingest: None,
                        query: &query,
                    },
                    validation_operation,
                    query.budget.max_latency_ms,
                )
                .await
                {
                    Ok(()) => retrieve_with_timeout(&self.method, query, operation).await,
                    Err(error) => Err(error),
                }
            } else {
                retrieve_with_timeout(&self.method, query, operation).await
            };
            let elapsed_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;

            let (hits, mut errors, response, evidence_bytes) = match response {
                Ok(response) => {
                    let hits = observed_hits(&response, max_hits)?;
                    let evidence_bytes = validate_evidence_budget(&response, max_evidence_bytes)?;
                    (hits, Vec::new(), Some(response), evidence_bytes)
                }
                Err(error) => {
                    retrieval_error_count += 1;
                    (Vec::new(), vec![trace_error(&error)], None, 0)
                }
            };
            let retrieval_truncated = response.as_ref().is_some_and(|response| response.truncated);
            let query_trace = QueryTrace {
                snapshot_id: response
                    .as_ref()
                    .and_then(|response| response.snapshot_id.clone()),
                retrieval_truncated,
                telemetry: response
                    .as_ref()
                    .and_then(|response| response.telemetry.clone()),
                ..query_trace
            };

            let retrieval = RetrievalOutcome {
                query: agent_query,
                response,
                hits,
                latency_ms: elapsed_ms,
                errors: errors.clone(),
            };
            let observed_hits = retrieval.hits.clone();
            let agent_started = std::time::Instant::now();
            let (agent_output, input_tokens, output_tokens, total_tokens) = match agent
                .execute(AgentStepInput {
                    sample_id: sample.sample_id.clone(),
                    observation: agent_observation,
                    session_id: sample.session_id,
                    episode_id: sample.episode_id,
                    terminal: sample.terminal,
                    terminal_reason: sample.terminal_reason.clone(),
                    retrieval,
                })
                .await
            {
                Ok(output) => {
                    agent_error_count += output
                        .errors
                        .iter()
                        .filter(|error| {
                            matches!(error.stage, TraceErrorStage::Agent | TraceErrorStage::Tool)
                        })
                        .count() as u64;
                    errors.extend(output.errors.iter().cloned());
                    let input_tokens = output.input_tokens;
                    let output_tokens = output.output_tokens;
                    let total_tokens = output.total_tokens;
                    (Some(output), input_tokens, output_tokens, total_tokens)
                }
                Err(error) => {
                    agent_error_count += 1 + error
                        .trace_errors
                        .iter()
                        .filter(|trace_error| {
                            matches!(
                                trace_error.stage,
                                TraceErrorStage::Agent | TraceErrorStage::Tool
                            )
                        })
                        .count() as u64;
                    errors.extend(error.trace_errors.iter().cloned());
                    errors.push(agent_trace_error(&error));
                    (
                        Some(AgentStepOutput {
                            action: error.action.as_ref().map(|action| (**action).clone()),
                            runtime_link: error.runtime_link.clone(),
                            ..AgentStepOutput::default()
                        }),
                        None,
                        None,
                        None,
                    )
                }
            };
            let agent_latency_ms = agent_started.elapsed().as_millis().min(u64::MAX as u128) as u64;

            let terminal_reason = sample.terminal_reason.clone();
            let step = StepTrace {
                run_id: self.context.run().run_id.clone(),
                step_index,
                sample_id: sample.sample_id,
                sample_digest: Some(sample_digest),
                namespace: self.context.namespace().to_string(),
                visibility_cutoff_ms: self.context.run().manifest.visibility_cutoff_ms,
                observation: sample.observation,
                ingest: None,
                query: Some(query_trace),
                observed_hits,
                agent_answer: agent_output
                    .as_ref()
                    .and_then(|output| output.answer.clone()),
                action: agent_output
                    .as_ref()
                    .and_then(|output| output.action.clone()),
                reward: agent_output
                    .as_ref()
                    .and_then(|output| output.reward.clone()),
                terminal_reason,
                telemetry: StepTelemetry {
                    ingest_latency_ms: 0,
                    retrieval_latency_ms: elapsed_ms,
                    agent_latency_ms,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    evidence_bytes,
                },
                errors,
                runtime_link: agent_output
                    .as_ref()
                    .and_then(|output| output.runtime_link.clone()),
                terminal: sample.terminal,
            };
            if step.terminal && self.dataset.next_sample()?.is_some() {
                return Err(RunnerError::InvalidSample(
                    "dataset contains samples after the terminal sample".to_string(),
                ));
            }
            self.context.append_step(&step)?;
            step_index += 1;

            if step.terminal {
                terminal_seen = true;
                break;
            }
        }

        if let Some(expected) = persisted_steps.get(persisted_index) {
            return Err(RunnerError::ResumeMismatch {
                expected: expected.sample_id.clone(),
                actual: "<dataset ended>".to_string(),
            });
        }

        if !terminal_seen {
            return Err(RunnerError::MissingTerminalSample);
        }
        self.context.finalize(unix_time_ms()?)?;
        Ok(RunSummary {
            run_id: self.context.run().run_id.clone(),
            step_count: step_index,
            ingest_error_count: 0,
            retrieval_error_count,
            agent_error_count,
        })
    }
}

/// Runs an online continual-memory episode.
///
/// Each sample is admitted into the method before it is retrieved.  Both
/// operations use the run-isolated namespace and the same operation UUID,
/// while their idempotency keys remain distinct so a provider can retry them
/// independently.  Retrieval-only experiments continue to use
/// [`RetrievalRunner`] and are unaffected by this lifecycle.
pub struct ContinualEpisodeRunner<M, D> {
    method: M,
    dataset: D,
    context: BenchmarkRunContext,
    artifact_validator: Option<Box<dyn ArtifactValidator>>,
}

impl<M, D> ContinualEpisodeRunner<M, D>
where
    M: MemoryMethod,
    D: ContinualDatasetSource,
{
    pub fn new(method: M, dataset: D, context: BenchmarkRunContext) -> Self {
        Self {
            method,
            dataset,
            context,
            artifact_validator: None,
        }
    }

    /// Attach an adapter-owned artifact validator to each continual step.
    pub fn with_artifact_validator<V>(mut self, validator: V) -> Self
    where
        V: ArtifactValidator + 'static,
    {
        self.artifact_validator = Some(Box::new(validator));
        self
    }

    pub fn context(&self) -> &BenchmarkRunContext {
        &self.context
    }

    pub async fn run(self) -> Result<RunSummary, RunnerError> {
        self.run_with_agent(NoopAgent).await
    }

    /// Executes ingest, retrieval, and agent work for every sample and
    /// appends one auditable step only after all three stages have returned.
    pub async fn run_with_agent<A>(mut self, mut agent: A) -> Result<RunSummary, RunnerError>
    where
        A: AgentStepExecutor,
    {
        let expected_method = &self.context.run().manifest.method;
        let actual_method = self.method.manifest();
        if actual_method.method_id != expected_method.method_id
            || actual_method.version != expected_method.version
        {
            return Err(RunnerError::MethodManifestMismatch {
                expected_id: expected_method.method_id.clone(),
                expected_version: expected_method.version.clone(),
                actual_id: actual_method.method_id,
                actual_version: actual_method.version,
            });
        }
        if let Some(cutoff_ms) = self.context.run().manifest.visibility_cutoff_ms {
            if !actual_method.capabilities.supports_visibility_cutoff {
                return Err(RunnerError::VisibilityCutoffUnsupported {
                    method_id: actual_method.method_id,
                    version: actual_method.version,
                    cutoff_ms,
                });
            }
        }
        if self.context.is_complete() {
            return Err(RunnerError::TraceAlreadyComplete);
        }
        if self.context.terminal_seen() {
            return Err(RunnerError::TraceAlreadyTerminal);
        }

        validate_dataset_digest(
            self.context.run().manifest.dataset_digest.as_deref(),
            self.dataset.dataset_digest(),
        )?;
        let persisted_steps = self.context.completed_steps().to_vec();
        let mut persisted_index = 0_usize;
        let mut step_index = self.context.next_step_index();
        let mut ingest_error_count = self
            .context
            .completed_steps()
            .iter()
            .filter_map(|step| step.ingest.as_ref())
            .filter(|ingest| ingest.error.is_some())
            .count() as u64;
        let mut retrieval_error_count = self
            .context
            .completed_steps()
            .iter()
            .flat_map(|step| step.errors.iter())
            .filter(|error| error.stage == TraceErrorStage::Retrieval)
            .count() as u64;
        let mut agent_error_count = self
            .context
            .completed_steps()
            .iter()
            .flat_map(|step| step.errors.iter())
            .filter(|error| matches!(error.stage, TraceErrorStage::Agent | TraceErrorStage::Tool))
            .count() as u64;
        let mut terminal_seen = false;
        let mut seen_sample_ids = BTreeSet::new();
        for step in &persisted_steps {
            if !seen_sample_ids.insert(step.sample_id.clone()) {
                return Err(RunnerError::InvalidSample(format!(
                    "persisted sample_id {:?} appears more than once",
                    step.sample_id
                )));
            }
        }

        while let Some(continual_sample) = self.dataset.next_sample()? {
            continual_sample.validate()?;
            let sample = continual_sample.sample;
            let query = self.context.bind_query(sample.query.clone())?;
            let ingest = bind_ingest_request(&self.context, continual_sample.ingest)?;
            validate_query_capabilities(
                &query,
                &actual_method.capabilities,
                &actual_method.method_id,
                &actual_method.version,
            )?;
            validate_ingest_capabilities(
                &ingest.items,
                &ingest.relations,
                &actual_method.capabilities,
                &actual_method.method_id,
                &actual_method.version,
            )?;
            let sample_digest = ContinualDatasetSample {
                sample: sample.clone(),
                ingest: ingest.clone(),
            }
            .identity_digest(&query, &ingest)?;

            if let Some(expected) = persisted_steps.get(persisted_index) {
                if sample.sample_id != expected.sample_id {
                    return Err(RunnerError::ResumeMismatch {
                        expected: expected.sample_id.clone(),
                        actual: sample.sample_id,
                    });
                }
                let Some(expected_digest) = expected.sample_digest.as_deref() else {
                    return Err(RunnerError::ResumeSampleDigestMissing {
                        sample_id: sample.sample_id,
                    });
                };
                if expected_digest != sample_digest {
                    return Err(RunnerError::ResumeSampleMismatch {
                        sample_id: sample.sample_id,
                        expected_digest: expected_digest.to_owned(),
                        actual_digest: sample_digest,
                    });
                }
                persisted_index += 1;
                continue;
            }
            if !seen_sample_ids.insert(sample.sample_id.clone()) {
                return Err(RunnerError::InvalidSample(format!(
                    "sample_id {:?} appears more than once",
                    sample.sample_id
                )));
            }

            let request_digest = digest_serialized(&ingest)?;
            let mut ingest_operation = self.context.operation_context(
                sample.session_id,
                sample.episode_id,
                sample.query.budget.max_latency_ms,
                None,
            );
            // The provider contract requires the request and operation
            // idempotency keys to be identical.  The request was already
            // scoped to this run by `bind_ingest_request`.
            ingest_operation.idempotency_key = Some(ingest.idempotency_key.clone());
            let ingest_started = std::time::Instant::now();
            let ingest_result = if let Some(validator) = self.artifact_validator.as_mut() {
                let validation_operation = self.context.operation_context(
                    sample.session_id,
                    sample.episode_id,
                    sample.query.budget.max_latency_ms,
                    Some(&format!("artifact/{}", sample.sample_id)),
                );
                match validate_artifacts_with_timeout(
                    validator.as_mut(),
                    ArtifactValidationInput {
                        ingest: Some(&ingest),
                        record_ingest: None,
                        query: &query,
                    },
                    validation_operation,
                    sample.query.budget.max_latency_ms,
                )
                .await
                {
                    Ok(()) => {
                        ingest_with_timeout(
                            &self.method,
                            ingest,
                            ingest_operation,
                            sample.query.budget.max_latency_ms,
                        )
                        .await
                    }
                    Err(error) => Err(error),
                }
            } else {
                ingest_with_timeout(
                    &self.method,
                    ingest,
                    ingest_operation,
                    sample.query.budget.max_latency_ms,
                )
                .await
            };
            let ingest_latency_ms =
                ingest_started.elapsed().as_millis().min(u64::MAX as u128) as u64;
            let ingest_completed_at_ms = Some(unix_time_ms()?);
            let (ingest_trace, ingest_error) = match ingest_result {
                Ok(receipt) => (
                    IngestTrace {
                        request_digest,
                        item_ids: receipt.item_ids.iter().map(ToString::to_string).collect(),
                        relation_ids: receipt
                            .relation_ids
                            .iter()
                            .map(ToString::to_string)
                            .collect(),
                        deduplicated: receipt.deduplicated,
                        snapshot_id: receipt.snapshot_id,
                        telemetry: receipt.telemetry,
                        completed_at_ms: ingest_completed_at_ms,
                        error: None,
                    },
                    None,
                ),
                Err(error) => {
                    ingest_error_count += 1;
                    let trace_error = ingest_trace_error(&error);
                    (
                        IngestTrace {
                            request_digest,
                            item_ids: Vec::new(),
                            relation_ids: Vec::new(),
                            deduplicated: false,
                            snapshot_id: None,
                            telemetry: None,
                            completed_at_ms: ingest_completed_at_ms,
                            error: Some(trace_error.clone()),
                        },
                        Some(trace_error),
                    )
                }
            };

            let query_trace = QueryTrace {
                namespace: self.context.namespace().to_string(),
                dataset_namespace: Some(self.context.dataset_namespace().to_string()),
                query_digest: Some(query_digest(&query)?),
                snapshot_id: None,
                input_digest: query_input_digest(query.input.as_ref())?,
                visibility_cutoff_ms: query.visibility_cutoff_ms,
                requested_hits: query.budget.max_hits as u32,
                retrieval_truncated: false,
                telemetry: None,
            };
            let max_hits = query.budget.max_hits;
            let max_evidence_bytes = query.budget.max_evidence_bytes;
            let (response, hits, mut errors, retrieval_latency_ms, evidence_bytes) =
                if let Some(ingest_error) = ingest_error {
                    (None, Vec::new(), vec![ingest_error], 0, 0)
                } else {
                    let retrieve_started = std::time::Instant::now();
                    let retrieve_key = format!("retrieve/{}", sample.sample_id);
                    let retrieve_operation = self.context.operation_context(
                        sample.session_id,
                        sample.episode_id,
                        sample.query.budget.max_latency_ms,
                        Some(&retrieve_key),
                    );
                    let retrieve_result =
                        retrieve_with_timeout(&self.method, query.clone(), retrieve_operation)
                            .await;
                    let elapsed_ms =
                        retrieve_started.elapsed().as_millis().min(u64::MAX as u128) as u64;
                    match retrieve_result {
                        Ok(response) => {
                            let hits = observed_hits(&response, max_hits)?;
                            let evidence_bytes =
                                validate_evidence_budget(&response, max_evidence_bytes)?;
                            (Some(response), hits, Vec::new(), elapsed_ms, evidence_bytes)
                        }
                        Err(error) => {
                            retrieval_error_count += 1;
                            (None, Vec::new(), vec![trace_error(&error)], elapsed_ms, 0)
                        }
                    }
                };
            let query_trace = QueryTrace {
                snapshot_id: response
                    .as_ref()
                    .and_then(|response| response.snapshot_id.clone()),
                retrieval_truncated: response.as_ref().is_some_and(|response| response.truncated),
                telemetry: response
                    .as_ref()
                    .and_then(|response| response.telemetry.clone()),
                ..query_trace
            };
            let retrieval = RetrievalOutcome {
                query: query.clone(),
                response,
                hits,
                latency_ms: retrieval_latency_ms,
                errors: errors.clone(),
            };
            let observed_hits = retrieval.hits.clone();
            let agent_started = std::time::Instant::now();
            let (agent_output, input_tokens, output_tokens, total_tokens) = match agent
                .execute(AgentStepInput {
                    sample_id: sample.sample_id.clone(),
                    observation: sample.observation.clone(),
                    session_id: sample.session_id,
                    episode_id: sample.episode_id,
                    terminal: sample.terminal,
                    terminal_reason: sample.terminal_reason.clone(),
                    retrieval,
                })
                .await
            {
                Ok(output) => {
                    agent_error_count += output
                        .errors
                        .iter()
                        .filter(|error| {
                            matches!(error.stage, TraceErrorStage::Agent | TraceErrorStage::Tool)
                        })
                        .count() as u64;
                    errors.extend(output.errors.iter().cloned());
                    let input_tokens = output.input_tokens;
                    let output_tokens = output.output_tokens;
                    let total_tokens = output.total_tokens;
                    (Some(output), input_tokens, output_tokens, total_tokens)
                }
                Err(error) => {
                    agent_error_count += 1 + error
                        .trace_errors
                        .iter()
                        .filter(|trace_error| {
                            matches!(
                                trace_error.stage,
                                TraceErrorStage::Agent | TraceErrorStage::Tool
                            )
                        })
                        .count() as u64;
                    errors.extend(error.trace_errors.iter().cloned());
                    errors.push(agent_trace_error(&error));
                    (
                        Some(AgentStepOutput {
                            action: error.action.as_ref().map(|action| (**action).clone()),
                            runtime_link: error.runtime_link.clone(),
                            ..AgentStepOutput::default()
                        }),
                        None,
                        None,
                        None,
                    )
                }
            };
            let agent_latency_ms = agent_started.elapsed().as_millis().min(u64::MAX as u128) as u64;

            let step = StepTrace {
                run_id: self.context.run().run_id.clone(),
                step_index,
                sample_id: sample.sample_id,
                sample_digest: Some(sample_digest),
                namespace: self.context.namespace().to_string(),
                visibility_cutoff_ms: self.context.run().manifest.visibility_cutoff_ms,
                observation: sample.observation,
                ingest: Some(ingest_trace),
                query: Some(query_trace),
                observed_hits,
                agent_answer: agent_output
                    .as_ref()
                    .and_then(|output| output.answer.clone()),
                action: agent_output
                    .as_ref()
                    .and_then(|output| output.action.clone()),
                reward: agent_output
                    .as_ref()
                    .and_then(|output| output.reward.clone()),
                terminal_reason: sample.terminal_reason,
                telemetry: StepTelemetry {
                    ingest_latency_ms,
                    retrieval_latency_ms,
                    agent_latency_ms,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    evidence_bytes,
                },
                errors,
                runtime_link: agent_output
                    .as_ref()
                    .and_then(|output| output.runtime_link.clone()),
                terminal: sample.terminal,
            };
            if step.terminal && self.dataset.next_sample()?.is_some() {
                return Err(RunnerError::InvalidSample(
                    "dataset contains samples after the terminal sample".to_string(),
                ));
            }
            self.context.append_step(&step)?;
            step_index += 1;
            if step.terminal {
                terminal_seen = true;
                break;
            }
        }

        if let Some(expected) = persisted_steps.get(persisted_index) {
            return Err(RunnerError::ResumeMismatch {
                expected: expected.sample_id.clone(),
                actual: "<dataset ended>".to_string(),
            });
        }
        if !terminal_seen {
            return Err(RunnerError::MissingTerminalSample);
        }
        self.context.finalize(unix_time_ms()?)?;
        Ok(RunSummary {
            run_id: self.context.run().run_id.clone(),
            step_count: step_index,
            ingest_error_count,
            retrieval_error_count,
            agent_error_count,
        })
    }
}

/// Runs a complete offline memory projection before querying it.
///
/// The source is consumed once for record ingestion and rewound before the
/// query phase. This ordering matches benchmarks whose memory method builds an
/// index from a complete episode or corpus before any question is allowed to
/// observe it.
pub struct OfflineBuildThenQueryRunner<M, D> {
    method: M,
    dataset: D,
    context: BenchmarkRunContext,
    artifact_validator: Option<Box<dyn ArtifactValidator>>,
}

impl<M, D> OfflineBuildThenQueryRunner<M, D>
where
    M: MemoryMethod,
    D: ReplayableOfflineDatasetSource,
{
    pub fn new(method: M, dataset: D, context: BenchmarkRunContext) -> Self {
        Self {
            method,
            dataset,
            context,
            artifact_validator: None,
        }
    }

    /// Attach an adapter-owned artifact validator to build and query phases.
    pub fn with_artifact_validator<V>(mut self, validator: V) -> Self
    where
        V: ArtifactValidator + 'static,
    {
        self.artifact_validator = Some(Box::new(validator));
        self
    }

    pub fn context(&self) -> &BenchmarkRunContext {
        &self.context
    }

    pub async fn run(self) -> Result<RunSummary, RunnerError> {
        self.run_with_agent(NoopAgent).await
    }

    /// Executes all record builds before the first retrieval, then delegates
    /// every query to the agent executor and appends retrieval-only steps.
    pub async fn run_with_agent<A>(mut self, mut agent: A) -> Result<RunSummary, RunnerError>
    where
        A: AgentStepExecutor,
    {
        let manifest = &self.context.run().manifest;
        if manifest.lifecycle != BenchmarkLifecycle::OfflineBuildThenQuery {
            return Err(RunnerError::InvalidSample(
                "offline runner requires offline_build_then_query lifecycle".to_string(),
            ));
        }
        let expected_method = &manifest.method;
        let actual_method = self.method.manifest();
        if actual_method.method_id != expected_method.method_id
            || actual_method.version != expected_method.version
        {
            return Err(RunnerError::MethodManifestMismatch {
                expected_id: expected_method.method_id.clone(),
                expected_version: expected_method.version.clone(),
                actual_id: actual_method.method_id,
                actual_version: actual_method.version,
            });
        }
        if let Some(cutoff_ms) = manifest.visibility_cutoff_ms {
            if !actual_method.capabilities.supports_visibility_cutoff {
                return Err(RunnerError::VisibilityCutoffUnsupported {
                    method_id: actual_method.method_id,
                    version: actual_method.version,
                    cutoff_ms,
                });
            }
        }
        if !actual_method.capabilities.supports_batch {
            return Err(RunnerError::InvalidSample(
                "offline runner requires a memory method with batch support".to_string(),
            ));
        }
        if !actual_method.capabilities.supports_record_ingest {
            return Err(RunnerError::CapabilityUnsupported {
                method_id: actual_method.method_id,
                version: actual_method.version,
                capability: "record_ingest".to_string(),
            });
        }
        if self.context.is_complete() {
            return Err(RunnerError::TraceAlreadyComplete);
        }
        if self.context.terminal_seen() {
            return Err(RunnerError::TraceAlreadyTerminal);
        }
        validate_dataset_digest(
            self.context.run().manifest.dataset_digest.as_deref(),
            self.dataset.dataset_digest(),
        )?;

        // Validate the complete source shape before the first provider call.
        // This keeps malformed terminal order, duplicate identities, and
        // capability mismatches from leaving a partially built projection.
        self.dataset.rewind()?;
        let mut source_sample_ids = BTreeSet::new();
        let mut source_idempotency_keys = BTreeSet::new();
        let mut source_terminal_seen = false;
        while let Some(offline_sample) = self.dataset.next_sample()? {
            offline_sample.validate()?;
            let sample = &offline_sample.sample;
            if source_terminal_seen {
                return Err(RunnerError::InvalidSample(
                    "dataset contains samples after the terminal sample".to_string(),
                ));
            }
            if !source_sample_ids.insert(sample.sample_id.clone()) {
                return Err(RunnerError::InvalidSample(format!(
                    "dataset sample_id {:?} appears more than once",
                    sample.sample_id
                )));
            }
            if !source_idempotency_keys.insert(offline_sample.build.idempotency_key.clone()) {
                return Err(RunnerError::InvalidSample(format!(
                    "dataset build idempotency_key {:?} appears more than once",
                    offline_sample.build.idempotency_key
                )));
            }
            let query = self.context.bind_query(sample.query.clone())?;
            let build = bind_record_ingest_request(&self.context, offline_sample.build.clone())?;
            validate_query_capabilities(
                &query,
                &actual_method.capabilities,
                &actual_method.method_id,
                &actual_method.version,
            )?;
            validate_ingest_capabilities(
                build.records.iter().map(|record| &record.item),
                &build.relations,
                &actual_method.capabilities,
                &actual_method.method_id,
                &actual_method.version,
            )?;
            source_terminal_seen = sample.terminal;
        }
        if !source_terminal_seen {
            return Err(RunnerError::MissingTerminalSample);
        }
        self.dataset.rewind()?;

        let persisted_builds = self.context.completed_builds().to_vec();
        let persisted_steps = self.context.completed_steps().to_vec();
        if persisted_steps.len() > persisted_builds.len() {
            return Err(RunnerError::InvalidSample(
                "offline trace contains query steps without matching build records".to_string(),
            ));
        }
        let mut seen_build_ids = BTreeSet::new();
        for build in &persisted_builds {
            if !seen_build_ids.insert(build.sample_id.clone()) {
                return Err(RunnerError::InvalidSample(format!(
                    "persisted sample_id {:?} appears more than once in builds",
                    build.sample_id
                )));
            }
        }

        // Phase 1: build the complete projection. Failed writes are durable
        // observations, but do not prevent the remaining samples from being
        // evaluated. The evaluator reports their rate separately.
        let mut persisted_build_index = 0_usize;
        let mut build_index = self.context.next_build_index();
        let mut ingest_error_count = 0_u64;
        let mut build_terminal_seen = false;
        while let Some(offline_sample) = self.dataset.next_sample()? {
            offline_sample.validate()?;
            let sample = &offline_sample.sample;
            if build_terminal_seen {
                return Err(RunnerError::InvalidSample(
                    "dataset contains samples after the terminal sample".to_string(),
                ));
            }
            let query = self.context.bind_query(sample.query.clone())?;
            let build = bind_record_ingest_request(&self.context, offline_sample.build.clone())?;
            let sample_digest = offline_sample.identity_digest(&query, &build)?;

            if let Some(expected) = persisted_builds.get(persisted_build_index) {
                if sample.sample_id != expected.sample_id {
                    return Err(RunnerError::ResumeMismatch {
                        expected: expected.sample_id.clone(),
                        actual: sample.sample_id.clone(),
                    });
                }
                let Some(expected_digest) = expected.sample_digest.as_deref() else {
                    return Err(RunnerError::ResumeSampleDigestMissing {
                        sample_id: sample.sample_id.clone(),
                    });
                };
                if expected_digest != sample_digest {
                    return Err(RunnerError::ResumeSampleMismatch {
                        sample_id: sample.sample_id.clone(),
                        expected_digest: expected_digest.to_owned(),
                        actual_digest: sample_digest,
                    });
                }
                if expected.error.is_some() {
                    ingest_error_count += 1;
                }
                persisted_build_index += 1;
                build_terminal_seen = sample.terminal;
                continue;
            }
            if !seen_build_ids.insert(sample.sample_id.clone()) {
                return Err(RunnerError::InvalidSample(format!(
                    "sample_id {:?} appears more than once in builds",
                    sample.sample_id
                )));
            }

            let request_digest = digest_serialized(&build)?;
            let mut operation = self.context.operation_context(
                sample.session_id,
                sample.episode_id,
                sample.query.budget.max_latency_ms,
                None,
            );
            operation.idempotency_key = Some(build.idempotency_key.clone());
            let started = std::time::Instant::now();
            let result = if let Some(validator) = self.artifact_validator.as_mut() {
                let validation_operation = self.context.operation_context(
                    sample.session_id,
                    sample.episode_id,
                    sample.query.budget.max_latency_ms,
                    Some(&format!("artifact/build/{}", sample.sample_id)),
                );
                match validate_artifacts_with_timeout(
                    validator.as_mut(),
                    ArtifactValidationInput {
                        ingest: None,
                        record_ingest: Some(&build),
                        query: &query,
                    },
                    validation_operation,
                    sample.query.budget.max_latency_ms,
                )
                .await
                {
                    Ok(()) => {
                        ingest_records_with_timeout(
                            &self.method,
                            build.clone(),
                            operation,
                            sample.query.budget.max_latency_ms,
                        )
                        .await
                    }
                    Err(error) => Err(error),
                }
            } else {
                ingest_records_with_timeout(
                    &self.method,
                    build.clone(),
                    operation,
                    sample.query.budget.max_latency_ms,
                )
                .await
            };
            let latency_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
            let completed_at_ms = Some(unix_time_ms()?);
            let build_trace = match result {
                Ok(receipt) => BuildTrace {
                    run_id: self.context.run().run_id.clone(),
                    build_index,
                    sample_id: sample.sample_id.clone(),
                    sample_digest: Some(sample_digest),
                    namespace: self.context.namespace().to_string(),
                    request_digest,
                    item_ids: receipt.item_ids.iter().map(ToString::to_string).collect(),
                    relation_ids: receipt
                        .relation_ids
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                    deduplicated: receipt.deduplicated,
                    snapshot_id: receipt.snapshot_id,
                    telemetry: receipt.telemetry,
                    completed_at_ms,
                    latency_ms,
                    error: None,
                },
                Err(error) => {
                    let build_trace = BuildTrace {
                        run_id: self.context.run().run_id.clone(),
                        build_index,
                        sample_id: sample.sample_id.clone(),
                        sample_digest: Some(sample_digest),
                        namespace: self.context.namespace().to_string(),
                        request_digest,
                        item_ids: Vec::new(),
                        relation_ids: Vec::new(),
                        deduplicated: false,
                        snapshot_id: None,
                        telemetry: None,
                        completed_at_ms,
                        latency_ms,
                        error: Some(build_trace_error(&error)),
                    };
                    self.context.append_build(&build_trace)?;
                    ingest_error_count += 1;
                    build_index += 1;
                    build_terminal_seen = sample.terminal;
                    continue;
                }
            };
            self.context.append_build(&build_trace)?;
            build_index += 1;
            build_terminal_seen = sample.terminal;
        }
        if let Some(expected) = persisted_builds.get(persisted_build_index) {
            return Err(RunnerError::ResumeMismatch {
                expected: expected.sample_id.clone(),
                actual: "<dataset ended>".to_string(),
            });
        }

        let build_errors = self
            .context
            .completed_builds()
            .iter()
            .filter_map(|build| {
                build
                    .error
                    .clone()
                    .map(|error| (build.sample_id.clone(), error))
            })
            .collect::<BTreeMap<_, _>>();

        // Phase 2 starts from the exact same source snapshot. No build call
        // appears below this point; all writes happened before rewind.
        self.dataset.rewind()?;
        let mut persisted_index = 0_usize;
        let mut step_index = self.context.next_step_index();
        let mut retrieval_error_count = persisted_steps
            .iter()
            .flat_map(|step| step.errors.iter())
            .filter(|error| error.stage == TraceErrorStage::Retrieval)
            .count() as u64;
        let mut agent_error_count = persisted_steps
            .iter()
            .flat_map(|step| step.errors.iter())
            .filter(|error| matches!(error.stage, TraceErrorStage::Agent | TraceErrorStage::Tool))
            .count() as u64;
        let mut terminal_seen = false;
        let mut seen_step_ids = BTreeSet::new();
        for step in &persisted_steps {
            if !seen_step_ids.insert(step.sample_id.clone()) {
                return Err(RunnerError::InvalidSample(format!(
                    "persisted sample_id {:?} appears more than once in steps",
                    step.sample_id
                )));
            }
        }

        while let Some(offline_sample) = self.dataset.next_sample()? {
            offline_sample.validate()?;
            let sample = offline_sample.sample.clone();
            let query = self.context.bind_query(sample.query.clone())?;
            let build = bind_record_ingest_request(&self.context, offline_sample.build.clone())?;
            let sample_digest = offline_sample.identity_digest(&query, &build)?;
            if let Some(expected) = persisted_steps.get(persisted_index) {
                if sample.sample_id != expected.sample_id {
                    return Err(RunnerError::ResumeMismatch {
                        expected: expected.sample_id.clone(),
                        actual: sample.sample_id.clone(),
                    });
                }
                let Some(expected_digest) = expected.sample_digest.as_deref() else {
                    return Err(RunnerError::ResumeSampleDigestMissing {
                        sample_id: sample.sample_id.clone(),
                    });
                };
                if expected_digest != sample_digest {
                    return Err(RunnerError::ResumeSampleMismatch {
                        sample_id: sample.sample_id,
                        expected_digest: expected_digest.to_owned(),
                        actual_digest: sample_digest,
                    });
                }
                persisted_index += 1;
                continue;
            }
            if !seen_step_ids.insert(sample.sample_id.clone()) {
                return Err(RunnerError::InvalidSample(format!(
                    "sample_id {:?} appears more than once in steps",
                    sample.sample_id
                )));
            }

            let query_trace = QueryTrace {
                namespace: self.context.namespace().to_string(),
                dataset_namespace: Some(self.context.dataset_namespace().to_string()),
                query_digest: Some(query_digest(&query)?),
                snapshot_id: None,
                input_digest: query_input_digest(query.input.as_ref())?,
                visibility_cutoff_ms: query.visibility_cutoff_ms,
                requested_hits: query.budget.max_hits as u32,
                retrieval_truncated: false,
                telemetry: None,
            };
            let max_hits = query.budget.max_hits;
            let max_evidence_bytes = query.budget.max_evidence_bytes;
            let operation = self.context.operation_context(
                sample.session_id,
                sample.episode_id,
                query.budget.max_latency_ms,
                Some(&format!("retrieve/{}", sample.sample_id)),
            );
            let started = std::time::Instant::now();
            let response = if let Some(validator) = self.artifact_validator.as_mut() {
                let validation_operation = self.context.operation_context(
                    sample.session_id,
                    sample.episode_id,
                    query.budget.max_latency_ms,
                    Some(&format!("artifact/query/{}", sample.sample_id)),
                );
                match validate_artifacts_with_timeout(
                    validator.as_mut(),
                    ArtifactValidationInput {
                        ingest: None,
                        record_ingest: None,
                        query: &query,
                    },
                    validation_operation,
                    query.budget.max_latency_ms,
                )
                .await
                {
                    Ok(()) => retrieve_with_timeout(&self.method, query.clone(), operation).await,
                    Err(error) => Err(error),
                }
            } else {
                retrieve_with_timeout(&self.method, query.clone(), operation).await
            };
            let elapsed_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
            let (hits, mut errors, response, evidence_bytes) = match response {
                Ok(response) => {
                    let hits = observed_hits(&response, max_hits)?;
                    let evidence_bytes = validate_evidence_budget(&response, max_evidence_bytes)?;
                    let errors = build_errors
                        .get(&sample.sample_id)
                        .cloned()
                        .into_iter()
                        .collect();
                    (hits, errors, Some(response), evidence_bytes)
                }
                Err(error) => {
                    retrieval_error_count += 1;
                    let mut errors = build_errors
                        .get(&sample.sample_id)
                        .cloned()
                        .into_iter()
                        .collect::<Vec<_>>();
                    errors.push(trace_error(&error));
                    (Vec::new(), errors, None, 0)
                }
            };
            let query_trace = QueryTrace {
                snapshot_id: response
                    .as_ref()
                    .and_then(|response| response.snapshot_id.clone()),
                retrieval_truncated: response.as_ref().is_some_and(|response| response.truncated),
                telemetry: response
                    .as_ref()
                    .and_then(|response| response.telemetry.clone()),
                ..query_trace
            };
            let retrieval = RetrievalOutcome {
                query: query.clone(),
                response,
                hits,
                latency_ms: elapsed_ms,
                errors: errors.clone(),
            };
            let observed_hits = retrieval.hits.clone();
            let agent_started = std::time::Instant::now();
            let (agent_output, input_tokens, output_tokens, total_tokens) = match agent
                .execute(AgentStepInput {
                    sample_id: sample.sample_id.clone(),
                    observation: sample.observation.clone(),
                    session_id: sample.session_id,
                    episode_id: sample.episode_id,
                    terminal: sample.terminal,
                    terminal_reason: sample.terminal_reason.clone(),
                    retrieval,
                })
                .await
            {
                Ok(output) => {
                    agent_error_count += output
                        .errors
                        .iter()
                        .filter(|error| {
                            matches!(error.stage, TraceErrorStage::Agent | TraceErrorStage::Tool)
                        })
                        .count() as u64;
                    errors.extend(output.errors.iter().cloned());
                    let input_tokens = output.input_tokens;
                    let output_tokens = output.output_tokens;
                    let total_tokens = output.total_tokens;
                    (Some(output), input_tokens, output_tokens, total_tokens)
                }
                Err(error) => {
                    agent_error_count += 1 + error
                        .trace_errors
                        .iter()
                        .filter(|trace_error| {
                            matches!(
                                trace_error.stage,
                                TraceErrorStage::Agent | TraceErrorStage::Tool
                            )
                        })
                        .count() as u64;
                    errors.extend(error.trace_errors.iter().cloned());
                    errors.push(agent_trace_error(&error));
                    (
                        Some(AgentStepOutput {
                            action: error.action.as_ref().map(|action| (**action).clone()),
                            runtime_link: error.runtime_link.clone(),
                            ..AgentStepOutput::default()
                        }),
                        None,
                        None,
                        None,
                    )
                }
            };
            let agent_latency_ms = agent_started.elapsed().as_millis().min(u64::MAX as u128) as u64;
            let step = StepTrace {
                run_id: self.context.run().run_id.clone(),
                step_index,
                sample_id: sample.sample_id,
                sample_digest: Some(sample_digest),
                namespace: self.context.namespace().to_string(),
                visibility_cutoff_ms: self.context.run().manifest.visibility_cutoff_ms,
                observation: sample.observation,
                ingest: None,
                query: Some(query_trace),
                observed_hits,
                agent_answer: agent_output
                    .as_ref()
                    .and_then(|output| output.answer.clone()),
                action: agent_output
                    .as_ref()
                    .and_then(|output| output.action.clone()),
                reward: agent_output
                    .as_ref()
                    .and_then(|output| output.reward.clone()),
                terminal_reason: sample.terminal_reason,
                telemetry: StepTelemetry {
                    ingest_latency_ms: 0,
                    retrieval_latency_ms: elapsed_ms,
                    agent_latency_ms,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    evidence_bytes,
                },
                errors,
                runtime_link: agent_output
                    .as_ref()
                    .and_then(|output| output.runtime_link.clone()),
                terminal: sample.terminal,
            };
            if step.terminal && self.dataset.next_sample()?.is_some() {
                return Err(RunnerError::InvalidSample(
                    "dataset contains samples after the terminal sample".to_string(),
                ));
            }
            self.context.append_step(&step)?;
            step_index += 1;
            if step.terminal {
                terminal_seen = true;
                break;
            }
        }

        if let Some(expected) = persisted_steps.get(persisted_index) {
            return Err(RunnerError::ResumeMismatch {
                expected: expected.sample_id.clone(),
                actual: "<dataset ended>".to_string(),
            });
        }
        if !terminal_seen {
            return Err(RunnerError::MissingTerminalSample);
        }
        self.context.finalize(unix_time_ms()?)?;
        Ok(RunSummary {
            run_id: self.context.run().run_id.clone(),
            step_count: step_index,
            ingest_error_count,
            retrieval_error_count,
            agent_error_count,
        })
    }
}

fn validate_dataset_digest(
    expected: Option<&str>,
    actual: Option<&str>,
) -> Result<(), RunnerError> {
    match (expected, actual) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(RunnerError::DatasetDigestMissing),
        (Some(expected), None) => Err(RunnerError::DatasetDigestUnavailable {
            expected: expected.to_owned(),
        }),
        (Some(expected), Some(actual)) if expected == actual => Ok(()),
        (Some(expected), Some(actual)) => Err(RunnerError::DatasetDigestMismatch {
            expected: expected.to_owned(),
            actual: actual.to_owned(),
        }),
    }
}

fn validate_query_capabilities(
    query: &MemoryQuery,
    capabilities: &MemoryCapabilities,
    method_id: &str,
    version: &str,
) -> Result<(), RunnerError> {
    // `query.modalities` is a filter over returned items. Asking for more than
    // one modality therefore does not imply cross-modal fusion; that separate
    // capability is a provider-level behavior and is intentionally not gated
    // here.
    for modality in &query.modalities {
        if !capabilities.modalities.contains(modality) {
            return Err(RunnerError::CapabilityUnsupported {
                method_id: method_id.to_owned(),
                version: version.to_owned(),
                capability: format!("modality={modality:?}"),
            });
        }
    }
    for kind in &query.kinds {
        if !capabilities.memory_kinds.contains(kind) {
            return Err(RunnerError::CapabilityUnsupported {
                method_id: method_id.to_owned(),
                version: version.to_owned(),
                capability: format!("memory_kind={kind:?}"),
            });
        }
    }
    if query.input.is_some() && !capabilities.supports_content_search {
        return Err(RunnerError::CapabilityUnsupported {
            method_id: method_id.to_owned(),
            version: version.to_owned(),
            capability: "content_search".to_owned(),
        });
    }
    if query.time_range.is_some() && !capabilities.supports_time_filter {
        return Err(RunnerError::CapabilityUnsupported {
            method_id: method_id.to_owned(),
            version: version.to_owned(),
            capability: "time_filter".to_owned(),
        });
    }
    if query.graph.is_some() && !capabilities.supports_graph {
        return Err(RunnerError::CapabilityUnsupported {
            method_id: method_id.to_owned(),
            version: version.to_owned(),
            capability: "graph".to_owned(),
        });
    }
    Ok(())
}

fn validate_ingest_capabilities<'a, I>(
    items: I,
    relations: &[MemoryRelation],
    capabilities: &MemoryCapabilities,
    method_id: &str,
    version: &str,
) -> Result<(), RunnerError>
where
    I: IntoIterator<Item = &'a MemoryItem>,
{
    for item in items {
        if !capabilities.modalities.contains(&item.modality) {
            return Err(RunnerError::CapabilityUnsupported {
                method_id: method_id.to_owned(),
                version: version.to_owned(),
                capability: format!("modality={:?}", item.modality),
            });
        }
        if !capabilities.memory_kinds.contains(&item.kind) {
            return Err(RunnerError::CapabilityUnsupported {
                method_id: method_id.to_owned(),
                version: version.to_owned(),
                capability: format!("memory_kind={:?}", item.kind),
            });
        }
    }
    if !relations.is_empty() && !capabilities.supports_graph {
        return Err(RunnerError::CapabilityUnsupported {
            method_id: method_id.to_owned(),
            version: version.to_owned(),
            capability: "graph".to_owned(),
        });
    }
    Ok(())
}

struct NoopAgent;

impl AgentStepExecutor for NoopAgent {
    fn execute<'a>(&'a mut self, _input: AgentStepInput) -> AgentFuture<'a> {
        Box::pin(async { Ok(AgentStepOutput::default()) })
    }
}

fn observed_hits(
    response: &RetrieveResponse,
    max_hits: usize,
) -> Result<Vec<ObservedHit>, RunnerError> {
    if response.hits.len() > max_hits {
        return Err(RunnerError::InvalidResponse(format!(
            "memory response contains {} hits, exceeding query max_hits {}",
            response.hits.len(),
            max_hits
        )));
    }
    let mut seen_record_ids = BTreeSet::new();
    let mut seen_ranks = BTreeSet::new();
    response
        .hits
        .iter()
        .enumerate()
        .map(|(index, hit)| {
            let expected_rank = index + 1;
            if hit.rank != expected_rank || !seen_ranks.insert(hit.rank) {
                return Err(RunnerError::InvalidResponse(
                    "memory response ranks must be contiguous, start at 1, and be unique"
                        .to_string(),
                ));
            }
            if hit.version == 0 {
                return Err(RunnerError::InvalidResponse(
                    "memory response contains a zero record version".to_string(),
                ));
            }
            if !seen_record_ids.insert(hit.item_id) {
                return Err(RunnerError::InvalidResponse(
                    "memory response contains a duplicate record id".to_string(),
                ));
            }
            if hit.score.is_some_and(|score| !score.is_finite()) {
                return Err(RunnerError::InvalidResponse(
                    "memory response contains a non-finite score".to_string(),
                ));
            }
            let mut source_record_ids = BTreeSet::new();
            for source_record_id in &hit.source_record_ids {
                if source_record_id.is_nil() || !source_record_ids.insert(*source_record_id) {
                    return Err(RunnerError::InvalidResponse(
                        "memory response contains an empty or duplicate source record id"
                            .to_string(),
                    ));
                }
            }
            Ok(ObservedHit {
                record_id: hit.item_id.to_string(),
                source_record_ids: source_record_ids
                    .into_iter()
                    .map(|id| id.to_string())
                    .collect(),
                version: hit.version,
                rank: u32::try_from(hit.rank).map_err(|_| {
                    RunnerError::InvalidResponse(
                        "memory hit rank exceeds trace representation".to_string(),
                    )
                })?,
                score: hit.score.map(f64::from),
                evidence_ids: hit
                    .evidence_refs
                    .iter()
                    .map(evidence_locator_identity)
                    .collect::<Result<BTreeSet<_>, _>>()?
                    .into_iter()
                    .collect(),
            })
        })
        .collect()
}

/// Returns the stable trace identity for one evidence locator.
///
/// The artifact id is kept as a readable prefix. The digest is calculated
/// from the complete locator, including its artifact metadata and every span
/// coordinate, so two locations in one artifact remain distinct while the
/// same locator remains stable across runs.
fn evidence_locator_identity(locator: &EvidenceLocator) -> Result<String, RunnerError> {
    let canonical_locator = serde_json::to_vec(locator)?;
    Ok(format!(
        "artifact:{}#sha256:{:x}",
        locator.artifact.artifact_id,
        Sha256::digest(canonical_locator)
    ))
}

fn response_evidence_bytes(response: &RetrieveResponse) -> u64 {
    response
        .hits
        .iter()
        .map(|hit| {
            serde_json::to_vec(&(&hit.evidence, &hit.evidence_refs, &hit.projection))
                .map_or(usize::MAX, |bytes| bytes.len())
        })
        .fold(0_u64, |total, bytes| {
            total.saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX))
        })
}

/// Enforce the query's evidence budget at the benchmark boundary.
///
/// A memory provider may set `truncated` when it had to omit results, but the
/// flag does not grant permission to exceed the requested byte budget.  The
/// runner records the bytes that were actually returned and rejects an
/// over-budget response before it reaches the agent or the trace.
fn validate_evidence_budget(
    response: &RetrieveResponse,
    max_evidence_bytes: usize,
) -> Result<u64, RunnerError> {
    let evidence_bytes = response_evidence_bytes(response);
    let limit = u64::try_from(max_evidence_bytes).unwrap_or(u64::MAX);
    if evidence_bytes > limit {
        return Err(RunnerError::InvalidResponse(format!(
            "memory response evidence uses {evidence_bytes} bytes, exceeding query max_evidence_bytes {max_evidence_bytes}"
        )));
    }
    Ok(evidence_bytes)
}

fn trace_error(error: &MemoryError) -> TraceErrorEntry {
    let (code, retryable) = match error {
        MemoryError::InvalidRequest(_) => ("invalid_request", false),
        MemoryError::UnsupportedCapability(_) => ("unsupported_capability", false),
        MemoryError::Unauthorized => ("unauthorized", false),
        MemoryError::NotFound(_) => ("not_found", false),
        MemoryError::Timeout => ("timeout", true),
        MemoryError::ResourceExhausted(_) => ("resource_exhausted", true),
        MemoryError::Backend(_) => ("backend", true),
    };
    TraceErrorEntry {
        stage: TraceErrorStage::Retrieval,
        code: code.to_string(),
        message: error.to_string(),
        retryable,
    }
}

fn agent_trace_error(error: &AgentExecutionError) -> TraceErrorEntry {
    TraceErrorEntry {
        stage: TraceErrorStage::Agent,
        code: error.code.clone(),
        message: error.message.clone(),
        retryable: error.retryable,
    }
}

fn query_digest(query: &MemoryQuery) -> Result<String, serde_json::Error> {
    serde_json::to_vec(query).map(|bytes| digest_bytes(&bytes))
}

fn query_input_digest(
    input: Option<&pi_agent_memory::ContentRef>,
) -> Result<Option<String>, serde_json::Error> {
    input
        .map(|input| serde_json::to_vec(input).map(|bytes| digest_bytes(&bytes)))
        .transpose()
}

async fn retrieve_with_timeout<M: MemoryMethod>(
    method: &M,
    query: MemoryQuery,
    operation: OperationContext,
) -> Result<RetrieveResponse, MemoryError> {
    let timeout_ms = query.budget.max_latency_ms;
    let future = method.retrieve(query, operation);
    match timeout_ms {
        Some(timeout_ms) => tokio::time::timeout(Duration::from_millis(timeout_ms), future)
            .await
            .unwrap_or(Err(MemoryError::Timeout)),
        None => future.await,
    }
}

fn digest_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn digest_serialized<T: Serialize>(value: &T) -> Result<String, RunnerError> {
    Ok(digest_bytes(&serde_json::to_vec(value)?))
}

fn bind_ingest_request(
    context: &BenchmarkRunContext,
    mut request: IngestRequest,
) -> Result<IngestRequest, RunnerError> {
    for item in &mut request.items {
        if !item.namespace.is_empty() && item.namespace != context.namespace() {
            return Err(RunnerError::InvalidSample(format!(
                "ingest item namespace {:?} does not match run namespace {:?}",
                item.namespace,
                context.namespace()
            )));
        }
        item.namespace = context.namespace().to_string();
    }
    request.idempotency_key = context.scoped_idempotency_key(&request.idempotency_key);
    Ok(request)
}

fn bind_record_ingest_request(
    context: &BenchmarkRunContext,
    mut request: RecordIngestRequest,
) -> Result<RecordIngestRequest, RunnerError> {
    for record in &mut request.records {
        if !record.item.namespace.is_empty() && record.item.namespace != context.namespace() {
            return Err(RunnerError::InvalidSample(format!(
                "build record namespace {:?} does not match run namespace {:?}",
                record.item.namespace,
                context.namespace()
            )));
        }
        record.item.namespace = context.namespace().to_string();
    }
    request.idempotency_key = context.scoped_idempotency_key(&request.idempotency_key);
    Ok(request)
}

async fn ingest_with_timeout<M: MemoryMethod>(
    method: &M,
    request: IngestRequest,
    operation: OperationContext,
    timeout_ms: Option<u64>,
) -> Result<IngestReceipt, MemoryError> {
    let future = method.ingest(request, operation);
    match timeout_ms {
        Some(timeout_ms) => tokio::time::timeout(Duration::from_millis(timeout_ms), future)
            .await
            .unwrap_or(Err(MemoryError::Timeout)),
        None => future.await,
    }
}

async fn ingest_records_with_timeout<M: MemoryMethod>(
    method: &M,
    request: RecordIngestRequest,
    operation: OperationContext,
    timeout_ms: Option<u64>,
) -> Result<IngestReceipt, MemoryError> {
    let future = method.ingest_records(request, operation);
    match timeout_ms {
        Some(timeout_ms) => tokio::time::timeout(Duration::from_millis(timeout_ms), future)
            .await
            .unwrap_or(Err(MemoryError::Timeout)),
        None => future.await,
    }
}

async fn validate_artifacts_with_timeout(
    validator: &mut dyn ArtifactValidator,
    input: ArtifactValidationInput<'_>,
    context: OperationContext,
    timeout_ms: Option<u64>,
) -> Result<(), MemoryError> {
    let future = validator.validate(input, context);
    match timeout_ms {
        Some(timeout_ms) => match tokio::time::timeout(Duration::from_millis(timeout_ms), future)
            .await
        {
            Ok(result) => result.map_err(|error| MemoryError::InvalidRequest(error.to_string())),
            Err(_) => Err(MemoryError::Timeout),
        },
        None => future
            .await
            .map_err(|error| MemoryError::InvalidRequest(error.to_string())),
    }
}

fn ingest_trace_error(error: &MemoryError) -> TraceErrorEntry {
    let mut trace_error = trace_error(error);
    trace_error.stage = TraceErrorStage::Ingest;
    trace_error
}

fn build_trace_error(error: &MemoryError) -> TraceErrorEntry {
    let mut trace_error = trace_error(error);
    trace_error.stage = TraceErrorStage::Build;
    trace_error
}

fn unix_time_ms() -> Result<i64, RunnerError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| RunnerError::ClockBeforeEpoch)?;
    i64::try_from(duration.as_millis()).map_err(|_| RunnerError::ClockBeforeEpoch)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{Arc, Mutex},
    };

    use pi_agent_benchmark::{
        read_builds, read_run, ActionTrace, AgentObservation, BenchmarkLifecycle,
        BenchmarkManifest, BenchmarkRun, MethodManifest, ObservationPart, TraceErrorStage,
    };
    use pi_agent_memory::{
        ArtifactRef, ContentRef, EvidenceLocator, IngestReceipt, IngestRequest,
        JsonlProjectionStore, MemoryBudget, MemoryCapabilities, MemoryFuture, MemoryHit,
        MemoryItem, MemoryKind, MemoryManifest, MemoryMethod, MemoryProvenance, MemoryQuery,
        MemoryRecord, MemoryScope, Modality, OperationContext, ProjectionMemoryMethod,
        ProjectionStore, RecordIngestRequest, RetrieveResponse, StorageContext, StoreWriteRequest,
    };
    use tempfile::tempdir;

    use super::*;

    struct StubMethod {
        calls: Arc<Mutex<Vec<String>>>,
        delay_ms: u64,
        supports_visibility_cutoff: bool,
        truncated: bool,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ContinualCallKind {
        Ingest,
        Build,
        Retrieve,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ContinualCall {
        kind: ContinualCallKind,
        namespace: String,
        run_id: Uuid,
    }

    struct ContinualStubMethod {
        calls: Arc<Mutex<Vec<ContinualCall>>>,
        fail_ingest: bool,
    }

    struct RecordIngestCapabilityFalse {
        inner: ContinualStubMethod,
    }

    struct RejectingArtifactValidator;

    impl ArtifactValidator for RejectingArtifactValidator {
        fn validate<'a>(
            &'a mut self,
            _input: ArtifactValidationInput<'a>,
            _context: OperationContext,
        ) -> ArtifactValidationFuture<'a> {
            Box::pin(async {
                Err(ArtifactValidationError::new(
                    "fixture artifact is unavailable",
                ))
            })
        }
    }

    struct StubAgent {
        calls: Arc<Mutex<Vec<String>>>,
    }

    fn test_runtime_link() -> RuntimeStepLink {
        RuntimeStepLink {
            runtime_run_id: Uuid::from_u128(100),
            turn_id: Uuid::from_u128(101),
            memory_decision: Some(pi_agent_benchmark::RuntimeMemoryDecisionLink {
                proposal_id: Uuid::from_u128(102),
                event_sequence: 7,
                decision_kind: "injected".to_owned(),
            }),
            memory_decisions: Vec::new(),
        }
    }

    impl AgentStepExecutor for StubAgent {
        fn execute<'a>(&'a mut self, input: AgentStepInput) -> AgentFuture<'a> {
            self.calls.lock().unwrap().push(input.sample_id.clone());
            if input.sample_id == "sample-1" {
                return Box::pin(async {
                    Err(
                        AgentExecutionError::new("model_timeout", "model did not finish", true)
                            .with_action(Some(ActionTrace {
                                action_type: "tool_calls".into(),
                                action_digest: Some("sha256:partial".into()),
                                calls: Vec::new(),
                            }))
                            .with_trace_errors(vec![TraceErrorEntry {
                                stage: TraceErrorStage::Tool,
                                code: "tool_failed".into(),
                                message: "tool rejected the request".into(),
                                retryable: false,
                            }])
                            .with_runtime_link(Some(test_runtime_link())),
                    )
                });
            }
            Box::pin(async {
                Ok(AgentStepOutput {
                    answer: Some("agent answer".into()),
                    action: Some(ActionTrace {
                        action_type: "respond".into(),
                        action_digest: Some("sha256:action".into()),
                        calls: Vec::new(),
                    }),
                    reward: None,
                    input_tokens: Some(11),
                    output_tokens: Some(7),
                    total_tokens: Some(18),
                    errors: Vec::new(),
                    runtime_link: Some(test_runtime_link()),
                })
            })
        }
    }

    impl MemoryMethod for StubMethod {
        fn manifest(&self) -> MemoryManifest {
            MemoryManifest {
                method_id: "stub".into(),
                version: "1".into(),
                schema_version: 1,
                capabilities: MemoryCapabilities {
                    modalities: vec![Modality::Text],
                    memory_kinds: Vec::new(),
                    supports_content_search: true,
                    supports_time_filter: true,
                    supports_visibility_cutoff: self.supports_visibility_cutoff,
                    supports_graph: false,
                    supports_cross_modal: false,
                    supports_update: false,
                    supports_delete: false,
                    supports_batch: false,
                    supports_record_ingest: false,
                    supports_record_get: false,
                    deterministic: true,
                    persistent: false,
                },
            }
        }

        fn ingest<'a>(
            &'a self,
            _request: pi_agent_memory::IngestRequest,
            _context: OperationContext,
        ) -> MemoryFuture<'a, Result<pi_agent_memory::IngestReceipt, MemoryError>> {
            Box::pin(async { Err(MemoryError::UnsupportedCapability("ingest".into())) })
        }

        fn retrieve<'a>(
            &'a self,
            query: MemoryQuery,
            _context: OperationContext,
        ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
            let calls = Arc::clone(&self.calls);
            let delay_ms = self.delay_ms;
            let truncated = self.truncated;
            Box::pin(async move {
                if delay_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                calls.lock().unwrap().push(query.namespace);
                Ok(RetrieveResponse {
                    query_id: Uuid::new_v4(),
                    hits: vec![MemoryHit {
                        item_id: Uuid::from_u128(7),
                        source_record_ids: Vec::new(),
                        version: 1,
                        method_id: "stub".into(),
                        rank: 1,
                        score: Some(0.75),
                        evidence: Vec::new(),
                        evidence_refs: Vec::new(),
                        projection: None,
                        provenance: "stub".into(),
                        snapshot_id: None,
                        latency_ms: 0,
                    }],
                    snapshot_id: None,
                    latency_ms: 0,
                    truncated,
                    telemetry: None,
                })
            })
        }

        fn get<'a>(
            &'a self,
            _item_id: Uuid,
            _context: OperationContext,
        ) -> MemoryFuture<'a, Result<pi_agent_memory::MemoryItem, MemoryError>> {
            Box::pin(async { Err(MemoryError::UnsupportedCapability("get".into())) })
        }
    }

    impl MemoryMethod for ContinualStubMethod {
        fn manifest(&self) -> MemoryManifest {
            MemoryManifest {
                method_id: "continual".into(),
                version: "1".into(),
                schema_version: 1,
                capabilities: MemoryCapabilities {
                    modalities: vec![Modality::Text],
                    memory_kinds: vec![MemoryKind::Episodic],
                    supports_content_search: true,
                    supports_time_filter: true,
                    supports_visibility_cutoff: true,
                    supports_graph: false,
                    supports_cross_modal: false,
                    supports_update: false,
                    supports_delete: false,
                    supports_batch: true,
                    supports_record_ingest: true,
                    supports_record_get: false,
                    deterministic: true,
                    persistent: false,
                },
            }
        }

        fn ingest<'a>(
            &'a self,
            _request: IngestRequest,
            context: OperationContext,
        ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
            let calls = Arc::clone(&self.calls);
            let fail_ingest = self.fail_ingest;
            Box::pin(async move {
                calls.lock().unwrap().push(ContinualCall {
                    kind: ContinualCallKind::Ingest,
                    namespace: context.namespace,
                    run_id: context.run_id,
                });
                if fail_ingest {
                    return Err(MemoryError::Backend("ingest failed".into()));
                }
                Ok(IngestReceipt {
                    item_ids: vec![Uuid::from_u128(11)],
                    relation_ids: vec![Uuid::from_u128(12)],
                    deduplicated: false,
                    snapshot_id: Some("snapshot-1".into()),
                    telemetry: None,
                })
            })
        }

        fn retrieve<'a>(
            &'a self,
            _query: MemoryQuery,
            context: OperationContext,
        ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
            let calls = Arc::clone(&self.calls);
            Box::pin(async move {
                calls.lock().unwrap().push(ContinualCall {
                    kind: ContinualCallKind::Retrieve,
                    namespace: context.namespace,
                    run_id: context.run_id,
                });
                Ok(RetrieveResponse {
                    query_id: Uuid::from_u128(13),
                    hits: Vec::new(),
                    snapshot_id: Some("snapshot-1".into()),
                    latency_ms: 0,
                    truncated: false,
                    telemetry: None,
                })
            })
        }

        fn ingest_records<'a>(
            &'a self,
            _request: RecordIngestRequest,
            context: OperationContext,
        ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
            let calls = Arc::clone(&self.calls);
            let fail_ingest = self.fail_ingest;
            Box::pin(async move {
                calls.lock().unwrap().push(ContinualCall {
                    kind: ContinualCallKind::Build,
                    namespace: context.namespace,
                    run_id: context.run_id,
                });
                if fail_ingest {
                    return Err(MemoryError::Backend("build failed".into()));
                }
                Ok(IngestReceipt {
                    item_ids: vec![Uuid::from_u128(21)],
                    relation_ids: Vec::new(),
                    deduplicated: false,
                    snapshot_id: Some("build-snapshot".into()),
                    telemetry: None,
                })
            })
        }

        fn get<'a>(
            &'a self,
            _item_id: Uuid,
            _context: OperationContext,
        ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
            Box::pin(async { Err(MemoryError::UnsupportedCapability("get".into())) })
        }
    }

    impl MemoryMethod for RecordIngestCapabilityFalse {
        fn manifest(&self) -> MemoryManifest {
            let mut manifest = self.inner.manifest();
            manifest.capabilities.supports_record_ingest = false;
            manifest
        }

        fn ingest<'a>(
            &'a self,
            request: IngestRequest,
            context: OperationContext,
        ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
            self.inner.ingest(request, context)
        }

        fn ingest_records<'a>(
            &'a self,
            request: RecordIngestRequest,
            context: OperationContext,
        ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
            self.inner.ingest_records(request, context)
        }

        fn retrieve<'a>(
            &'a self,
            query: MemoryQuery,
            context: OperationContext,
        ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
            self.inner.retrieve(query, context)
        }

        fn get<'a>(
            &'a self,
            item_id: Uuid,
            context: OperationContext,
        ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
            self.inner.get(item_id, context)
        }
    }

    fn context_for_method(path: &std::path::Path, method_id: &str) -> BenchmarkRunContext {
        context_for_lifecycle(path, method_id, BenchmarkLifecycle::RetrievalOnly)
    }

    fn context_for_lifecycle(
        path: &std::path::Path,
        method_id: &str,
        lifecycle: BenchmarkLifecycle,
    ) -> BenchmarkRunContext {
        let run = BenchmarkRun::new(
            "run-1",
            BenchmarkManifest::new(
                "retrieval",
                "bench/test",
                MethodManifest {
                    method_id: method_id.into(),
                    version: "1".into(),
                    config: BTreeMap::new(),
                },
                "dataset-1",
                Some(100),
            ),
            1,
        );
        let mut run = run;
        run.manifest.lifecycle = lifecycle;
        BenchmarkRunContext::open(path, run, Uuid::from_u128(9)).unwrap()
    }

    fn context(path: &std::path::Path) -> BenchmarkRunContext {
        context_for_method(path, "stub")
    }

    fn context_with_dataset_digest(path: &std::path::Path, method_id: &str) -> BenchmarkRunContext {
        let mut run = BenchmarkRun::new(
            "run-1",
            BenchmarkManifest::new(
                "retrieval",
                "bench/test",
                MethodManifest {
                    method_id: method_id.into(),
                    version: "1".into(),
                    config: BTreeMap::new(),
                },
                "dataset-1",
                Some(100),
            ),
            1,
        );
        run.manifest.dataset_digest = Some("sha256:expected".into());
        BenchmarkRunContext::open(path, run, Uuid::from_u128(9)).unwrap()
    }

    fn sample(id: &str, terminal: bool) -> DatasetSample {
        DatasetSample {
            sample_id: id.into(),
            observation: AgentObservation {
                parts: vec![ObservationPart::Text("visible".into())],
            },
            query: MemoryQuery {
                input: Some(ContentRef::Text {
                    text: "query".into(),
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
            },
            session_id: None,
            episode_id: None,
            terminal,
            terminal_reason: terminal.then_some(TerminalReason::Success),
        }
    }

    fn continual_sample(id: &str, terminal: bool, content: &str) -> ContinualDatasetSample {
        ContinualDatasetSample {
            sample: sample(id, terminal),
            ingest: IngestRequest {
                items: vec![MemoryItem {
                    id: Uuid::from_u128(10),
                    version: 1,
                    kind: MemoryKind::Episodic,
                    scope: MemoryScope::Agent,
                    modality: Modality::Text,
                    content: ContentRef::Text {
                        text: content.to_owned(),
                    },
                    namespace: String::new(),
                    episode_id: Some("episode-1".into()),
                    subject: None,
                    source: "test".into(),
                    metadata: BTreeMap::new(),
                    abstraction_level: 0,
                    event_time: None,
                    ingest_time_ms: 1,
                    schema_version: 1,
                }],
                relations: Vec::new(),
                idempotency_key: format!("{id}-ingest"),
            },
        }
    }

    fn offline_sample(id: &str, terminal: bool, content: &str) -> OfflineDatasetSample {
        let mut item_id_bytes = [0_u8; 16];
        item_id_bytes.copy_from_slice(&Sha256::digest(id.as_bytes())[..16]);
        let item_id = Uuid::from_bytes(item_id_bytes);
        OfflineDatasetSample {
            sample: sample(id, terminal),
            build: RecordIngestRequest {
                records: vec![MemoryRecord {
                    item: MemoryItem {
                        id: item_id,
                        version: 1,
                        kind: MemoryKind::Episodic,
                        scope: MemoryScope::Agent,
                        modality: Modality::Text,
                        content: ContentRef::Text {
                            text: content.to_owned(),
                        },
                        namespace: String::new(),
                        episode_id: None,
                        subject: None,
                        source: "test".to_owned(),
                        metadata: BTreeMap::new(),
                        abstraction_level: 0,
                        event_time: None,
                        ingest_time_ms: 1,
                        schema_version: 1,
                    },
                    episode: None,
                    evidence: Vec::new(),
                    provenance: MemoryProvenance {
                        producer: "test".to_owned(),
                        source_id: id.to_owned(),
                        source_event_id: None,
                        parent_record_ids: Vec::new(),
                        method_version: Some("fixture".to_owned()),
                    },
                }],
                relations: Vec::new(),
                idempotency_key: format!("{id}-build"),
            },
        }
    }

    #[test]
    fn offline_sample_identity_ignores_run_scoped_bindings() {
        let sample = offline_sample("sample-0", true, "memory");
        let mut first_query = sample.sample.query.clone();
        first_query.namespace = "bench/test/run-a".to_owned();
        let mut first_build = sample.build.clone();
        first_build.idempotency_key = "bench/test/run-a/sample-0-build".to_owned();
        first_build.records[0].item.namespace = first_query.namespace.clone();

        let mut second_query = sample.sample.query.clone();
        second_query.namespace = "bench/test/run-b".to_owned();
        let mut second_build = sample.build.clone();
        second_build.idempotency_key = "bench/test/run-b/sample-0-build".to_owned();
        second_build.records[0].item.namespace = second_query.namespace.clone();

        assert_eq!(
            sample.identity_digest(&first_query, &first_build).unwrap(),
            sample
                .identity_digest(&second_query, &second_build)
                .unwrap()
        );
    }

    #[tokio::test]
    async fn offline_runner_builds_every_record_before_retrieval() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let context = context_for_lifecycle(
            &path,
            "continual",
            BenchmarkLifecycle::OfflineBuildThenQuery,
        );
        let namespace = context.namespace().to_owned();
        let run_id = context.operation_run_id();
        let runner = OfflineBuildThenQueryRunner::new(
            ContinualStubMethod {
                calls: Arc::clone(&calls),
                fail_ingest: false,
            },
            VecOfflineDataset::new(vec![
                offline_sample("sample-0", false, "first"),
                offline_sample("sample-1", true, "second"),
            ]),
            context,
        );

        let summary = runner.run().await.unwrap();
        assert_eq!(summary.step_count, 2);
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            &[
                ContinualCall {
                    kind: ContinualCallKind::Build,
                    namespace: namespace.clone(),
                    run_id,
                },
                ContinualCall {
                    kind: ContinualCallKind::Build,
                    namespace: namespace.clone(),
                    run_id,
                },
                ContinualCall {
                    kind: ContinualCallKind::Retrieve,
                    namespace: namespace.clone(),
                    run_id,
                },
                ContinualCall {
                    kind: ContinualCallKind::Retrieve,
                    namespace,
                    run_id,
                },
            ]
        );
        assert_eq!(read_builds(&path, "run-1").unwrap().len(), 2);
        let trace = read_run(&path, "run-1").unwrap();
        assert!(trace.steps.iter().all(|step| step.ingest.is_none()));
    }

    #[tokio::test]
    async fn offline_runner_preserves_build_failure_and_continues_retrieval() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = OfflineBuildThenQueryRunner::new(
            ContinualStubMethod {
                calls: Arc::clone(&calls),
                fail_ingest: true,
            },
            VecOfflineDataset::new(vec![
                offline_sample("sample-0", false, "first"),
                offline_sample("sample-1", true, "second"),
            ]),
            context_for_lifecycle(
                &path,
                "continual",
                BenchmarkLifecycle::OfflineBuildThenQuery,
            ),
        );

        let summary = runner.run().await.unwrap();
        assert_eq!(summary.ingest_error_count, 2);
        assert_eq!(summary.retrieval_error_count, 0);
        assert_eq!(
            calls
                .lock()
                .unwrap()
                .iter()
                .map(|call| &call.kind)
                .collect::<Vec<_>>(),
            vec![
                &ContinualCallKind::Build,
                &ContinualCallKind::Build,
                &ContinualCallKind::Retrieve,
                &ContinualCallKind::Retrieve,
            ]
        );
        let builds = read_builds(&path, "run-1").unwrap();
        assert_eq!(builds.len(), 2);
        assert!(builds
            .iter()
            .all(|build| build.error.as_ref().map(|error| error.stage)
                == Some(TraceErrorStage::Build)));
        let trace = read_run(&path, "run-1").unwrap();
        assert!(trace.complete);
        assert_eq!(trace.steps.len(), 2);
        assert!(trace.steps.iter().all(|step| {
            step.errors.len() == 1 && step.errors[0].stage == TraceErrorStage::Build
        }));
    }

    #[tokio::test]
    async fn offline_runner_preflights_source_before_provider_calls() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let first = offline_sample("sample-0", false, "first");
        let mut second = offline_sample("sample-1", true, "second");
        second.build.idempotency_key = first.build.idempotency_key.clone();
        let runner = OfflineBuildThenQueryRunner::new(
            ContinualStubMethod {
                calls: Arc::clone(&calls),
                fail_ingest: false,
            },
            VecOfflineDataset::new(vec![first, second]),
            context_for_lifecycle(
                &path,
                "continual",
                BenchmarkLifecycle::OfflineBuildThenQuery,
            ),
        );

        assert!(matches!(
            runner.run().await,
            Err(RunnerError::InvalidSample(message))
                if message.contains("idempotency_key")
        ));
        assert!(calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn offline_runner_rejects_record_ingest_without_provider_call() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = OfflineBuildThenQueryRunner::new(
            RecordIngestCapabilityFalse {
                inner: ContinualStubMethod {
                    calls: Arc::clone(&calls),
                    fail_ingest: false,
                },
            },
            VecOfflineDataset::new(vec![offline_sample("sample-0", true, "first")]),
            context_for_lifecycle(
                &path,
                "continual",
                BenchmarkLifecycle::OfflineBuildThenQuery,
            ),
        );

        assert!(matches!(
            runner.run().await,
            Err(RunnerError::CapabilityUnsupported { capability, .. })
                if capability == "record_ingest"
        ));
        assert!(calls.lock().unwrap().is_empty());
        assert!(read_builds(&path, "run-1").unwrap().is_empty());
    }

    #[tokio::test]
    async fn continual_runner_ingests_and_retrieves_with_one_operation_identity() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let context = context_for_method(&path, "continual");
        let operation_run_id = context.operation_run_id();
        let namespace = context.namespace().to_owned();
        let runner = ContinualEpisodeRunner::new(
            ContinualStubMethod {
                calls: Arc::clone(&calls),
                fail_ingest: false,
            },
            VecContinualDataset::new(vec![continual_sample("sample-0", true, "memory")]),
            context,
        );

        let summary = runner.run().await.unwrap();
        assert_eq!(summary.step_count, 1);
        assert_eq!(summary.ingest_error_count, 0);
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            &[
                ContinualCall {
                    kind: ContinualCallKind::Ingest,
                    namespace: namespace.clone(),
                    run_id: operation_run_id,
                },
                ContinualCall {
                    kind: ContinualCallKind::Retrieve,
                    namespace: namespace.clone(),
                    run_id: operation_run_id,
                },
            ]
        );

        let trace = read_run(&path, "run-1").unwrap();
        let ingest = trace.steps[0].ingest.as_ref().expect("ingest trace");
        assert_eq!(ingest.item_ids, vec![Uuid::from_u128(11).to_string()]);
        assert_eq!(ingest.relation_ids, vec![Uuid::from_u128(12).to_string()]);
        assert_eq!(ingest.snapshot_id.as_deref(), Some("snapshot-1"));
        assert!(ingest.completed_at_ms.is_some());
        assert!(ingest.error.is_none());
        assert_eq!(
            trace.steps[0]
                .query
                .as_ref()
                .expect("query trace")
                .namespace,
            namespace
        );
    }

    #[tokio::test]
    async fn continual_runner_records_ingest_failure_without_retrieving_stale_data() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = ContinualEpisodeRunner::new(
            ContinualStubMethod {
                calls: Arc::clone(&calls),
                fail_ingest: true,
            },
            VecContinualDataset::new(vec![continual_sample("sample-0", true, "memory")]),
            context_for_method(&path, "continual"),
        );

        let summary = runner.run().await.unwrap();
        assert_eq!(summary.ingest_error_count, 1);
        assert_eq!(calls.lock().unwrap().len(), 1);
        assert_eq!(calls.lock().unwrap()[0].kind, ContinualCallKind::Ingest);

        let trace = read_run(&path, "run-1").unwrap();
        assert_eq!(trace.steps[0].errors.len(), 1);
        assert_eq!(trace.steps[0].errors[0].stage, TraceErrorStage::Ingest);
        assert_eq!(
            trace.steps[0]
                .ingest
                .as_ref()
                .and_then(|ingest| ingest.error.as_ref())
                .map(|error| error.stage),
            Some(TraceErrorStage::Ingest)
        );
    }

    #[tokio::test]
    async fn continual_runner_records_artifact_failure_before_provider_ingest() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = ContinualEpisodeRunner::new(
            ContinualStubMethod {
                calls: Arc::clone(&calls),
                fail_ingest: false,
            },
            VecContinualDataset::new(vec![continual_sample("sample-0", true, "memory")]),
            context_for_method(&path, "continual"),
        )
        .with_artifact_validator(RejectingArtifactValidator);

        let summary = runner.run().await.unwrap();
        assert_eq!(summary.ingest_error_count, 1);
        assert!(calls.lock().unwrap().is_empty());

        let trace = read_run(&path, "run-1").unwrap();
        assert_eq!(trace.steps[0].errors.len(), 1);
        assert_eq!(trace.steps[0].errors[0].stage, TraceErrorStage::Ingest);
        assert!(trace.steps[0].observed_hits.is_empty());
    }

    #[tokio::test]
    async fn continual_runner_rejects_resume_when_ingest_payload_changes() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let first = ContinualEpisodeRunner::new(
            ContinualStubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                fail_ingest: false,
            },
            VecContinualDataset::new(vec![continual_sample("sample-0", false, "original")]),
            context_for_method(&path, "continual"),
        );
        assert!(matches!(
            first.run().await,
            Err(RunnerError::MissingTerminalSample)
        ));

        let resumed = ContinualEpisodeRunner::new(
            ContinualStubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                fail_ingest: false,
            },
            VecContinualDataset::new(vec![continual_sample("sample-0", false, "changed")]),
            BenchmarkRunContext::resume(&path, "run-1").unwrap(),
        );
        assert!(matches!(
            resumed.run().await,
            Err(RunnerError::ResumeSampleMismatch { sample_id, .. }) if sample_id == "sample-0"
        ));
    }

    #[tokio::test]
    async fn retrieval_runner_rejects_source_without_manifest_digest() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = RetrievalRunner::new(
            StubMethod {
                calls: Arc::clone(&calls),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![sample("sample-0", true)]),
            context_with_dataset_digest(&path, "stub"),
        );

        assert!(matches!(
            runner.run().await,
            Err(RunnerError::DatasetDigestUnavailable { expected })
                if expected == "sha256:expected"
        ));
        assert!(calls.lock().unwrap().is_empty());
        assert!(read_run(&path, "run-1").unwrap().steps.is_empty());
    }

    #[tokio::test]
    async fn continual_runner_rejects_source_without_manifest_digest() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = ContinualEpisodeRunner::new(
            ContinualStubMethod {
                calls: Arc::clone(&calls),
                fail_ingest: false,
            },
            VecContinualDataset::new(vec![continual_sample("sample-0", true, "memory")]),
            context_with_dataset_digest(&path, "continual"),
        );

        assert!(matches!(
            runner.run().await,
            Err(RunnerError::DatasetDigestUnavailable { expected })
                if expected == "sha256:expected"
        ));
        assert!(calls.lock().unwrap().is_empty());
        assert!(read_run(&path, "run-1").unwrap().steps.is_empty());
    }

    #[tokio::test]
    async fn runner_rejects_visibility_cutoff_unsupported_before_writing_first_step() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = RetrievalRunner::new(
            StubMethod {
                calls: Arc::clone(&calls),
                delay_ms: 0,
                supports_visibility_cutoff: false,
                truncated: false,
            },
            VecDataset::new(vec![sample("sample-0", true)]),
            context(&path),
        );

        assert!(matches!(
            runner.run().await,
            Err(RunnerError::VisibilityCutoffUnsupported {
                method_id,
                version,
                cutoff_ms: 100,
            }) if method_id == "stub" && version == "1"
        ));
        assert!(calls.lock().unwrap().is_empty());
        let trace = read_run(&path, "run-1").unwrap();
        assert!(trace.steps.is_empty());
        assert!(!trace.complete);
    }

    #[tokio::test]
    async fn retrieval_runner_rejects_unsupported_modality_before_provider_call() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut video_sample = sample("video-sample", true);
        video_sample.query.modalities = vec![Modality::Video];
        let runner = RetrievalRunner::new(
            StubMethod {
                calls: Arc::clone(&calls),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![video_sample]),
            context(&path),
        );

        assert!(matches!(
            runner.run().await,
            Err(RunnerError::CapabilityUnsupported { capability, .. })
                if capability == "modality=Video"
        ));
        assert!(calls.lock().unwrap().is_empty());
        assert!(read_run(&path, "run-1").unwrap().steps.is_empty());
    }

    #[tokio::test]
    async fn offline_runner_rejects_unsupported_modality_before_building() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut video_sample = offline_sample("video-sample", true, "video memory");
        video_sample.sample.query.modalities = vec![Modality::Video];
        video_sample.build.records[0].item.modality = Modality::Video;
        let runner = OfflineBuildThenQueryRunner::new(
            ContinualStubMethod {
                calls: Arc::clone(&calls),
                fail_ingest: false,
            },
            VecOfflineDataset::new(vec![video_sample]),
            context_for_lifecycle(
                &path,
                "continual",
                BenchmarkLifecycle::OfflineBuildThenQuery,
            ),
        );

        assert!(matches!(
            runner.run().await,
            Err(RunnerError::CapabilityUnsupported { capability, .. })
                if capability == "modality=Video"
        ));
        assert!(calls.lock().unwrap().is_empty());
        assert!(read_run(&path, "run-1").unwrap().steps.is_empty());
        assert!(read_builds(&path, "run-1").unwrap().is_empty());
    }

    #[tokio::test]
    async fn retrieval_runner_records_bound_queries_and_hits() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = RetrievalRunner::new(
            StubMethod {
                calls: Arc::clone(&calls),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![sample("sample-0", false), sample("sample-1", true)]),
            context(&path),
        );

        let summary = runner.run().await.unwrap();
        assert_eq!(summary.step_count, 2);
        assert_eq!(summary.retrieval_error_count, 0);
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            &["bench/test", "bench/test"]
        );

        let trace = read_run(&path, "run-1").unwrap();
        assert!(trace.complete);
        assert_eq!(
            trace.steps[0].observed_hits[0].record_id,
            Uuid::from_u128(7).to_string()
        );
        assert_eq!(
            trace.steps[1].terminal_reason,
            Some(TerminalReason::Success)
        );
        assert!(
            !trace.steps[0]
                .query
                .as_ref()
                .expect("retrieval query trace")
                .retrieval_truncated
        );
    }

    #[tokio::test]
    async fn retrieval_runner_persists_provider_truncation_in_query_trace() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let runner = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: true,
            },
            VecDataset::new(vec![sample("sample-0", true)]),
            context(&path),
        );

        let summary = runner.run().await.unwrap();
        assert_eq!(summary.retrieval_error_count, 0);
        let trace = read_run(&path, "run-1").unwrap();
        let query = trace.steps[0]
            .query
            .as_ref()
            .expect("retrieval query trace");
        assert!(query.retrieval_truncated);
        assert!(trace.steps[0].errors.is_empty());
    }

    #[tokio::test]
    async fn agent_runner_records_output_and_stage_specific_errors() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let runner = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![sample("sample-0", false), sample("sample-1", true)]),
            context(&path),
        );

        let summary = runner
            .run_with_agent(StubAgent {
                calls: Arc::clone(&calls),
            })
            .await
            .unwrap();
        assert_eq!(summary.retrieval_error_count, 0);
        assert_eq!(summary.agent_error_count, 2);
        assert_eq!(calls.lock().unwrap().as_slice(), &["sample-0", "sample-1"]);

        let trace = read_run(&path, "run-1").unwrap();
        assert_eq!(trace.steps[0].agent_answer.as_deref(), Some("agent answer"));
        assert_eq!(
            trace.steps[0]
                .action
                .as_ref()
                .map(|action| action.action_type.as_str()),
            Some("respond")
        );
        assert_eq!(trace.steps[0].telemetry.input_tokens, Some(11));
        assert_eq!(trace.steps[0].telemetry.output_tokens, Some(7));
        assert_eq!(trace.steps[0].telemetry.total_tokens, Some(18));
        assert_eq!(trace.steps[0].runtime_link, Some(test_runtime_link()));
        assert_eq!(trace.steps[1].errors[0].stage, TraceErrorStage::Tool);
        assert_eq!(trace.steps[1].errors[0].code, "tool_failed");
        assert_eq!(trace.steps[1].errors[1].stage, TraceErrorStage::Agent);
        assert_eq!(trace.steps[1].errors[1].code, "model_timeout");
        assert_eq!(
            trace.steps[1]
                .action
                .as_ref()
                .unwrap()
                .action_digest
                .as_deref(),
            Some("sha256:partial")
        );
        assert_eq!(trace.steps[1].runtime_link, Some(test_runtime_link()));
    }

    #[tokio::test]
    async fn runner_rejects_a_dataset_without_terminal_sample() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let runner = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![sample("sample-0", false)]),
            context(&path),
        );

        assert!(matches!(
            runner.run().await,
            Err(RunnerError::MissingTerminalSample)
        ));
    }

    #[tokio::test]
    async fn runner_resumes_after_persisted_prefix_without_reexecuting_it() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let first = RetrievalRunner::new(
            StubMethod {
                calls: Arc::clone(&calls),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![sample("sample-0", false)]),
            context(&path),
        );

        assert!(matches!(
            first.run().await,
            Err(RunnerError::MissingTerminalSample)
        ));
        let partial = read_run(&path, "run-1").unwrap();
        assert_eq!(
            partial
                .steps
                .iter()
                .map(|step| step.sample_id.as_str())
                .collect::<Vec<_>>(),
            vec!["sample-0"]
        );

        let resumed = RetrievalRunner::new(
            StubMethod {
                calls: Arc::clone(&calls),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![sample("sample-0", false), sample("sample-1", true)]),
            BenchmarkRunContext::resume(&path, "run-1").unwrap(),
        );
        let summary = resumed.run().await.unwrap();

        assert_eq!(summary.step_count, 2);
        let trace = read_run(&path, "run-1").unwrap();
        assert!(trace.complete);
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.steps[0].sample_id, "sample-0");
        assert_eq!(trace.steps[1].sample_id, "sample-1");
        assert_eq!(calls.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn runner_rejects_a_resumed_dataset_with_a_different_prefix() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let first = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![sample("sample-0", false)]),
            context(&path),
        );
        assert!(matches!(
            first.run().await,
            Err(RunnerError::MissingTerminalSample)
        ));

        let resumed = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![sample("sample-1", true), sample("sample-0", false)]),
            BenchmarkRunContext::resume(&path, "run-1").unwrap(),
        );
        assert!(matches!(
            resumed.run().await,
            Err(RunnerError::ResumeMismatch { expected, actual })
                if expected == "sample-0" && actual == "sample-1"
        ));

        let trace = read_run(&path, "run-1").unwrap();
        assert_eq!(trace.steps.len(), 1);
        assert!(!trace.complete);
    }

    #[tokio::test]
    async fn runner_rejects_a_resumed_sample_with_changed_query() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let mut original = sample("sample-0", false);
        original.query.input = Some(ContentRef::Text {
            text: "original".into(),
        });
        let first = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![original]),
            context(&path),
        );
        assert!(matches!(
            first.run().await,
            Err(RunnerError::MissingTerminalSample)
        ));

        let mut changed = sample("sample-0", false);
        changed.query.input = Some(ContentRef::Text {
            text: "changed".into(),
        });
        let resumed = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![changed, sample("sample-1", true)]),
            BenchmarkRunContext::resume(&path, "run-1").unwrap(),
        );

        assert!(matches!(
            resumed.run().await,
            Err(RunnerError::ResumeSampleMismatch { sample_id, .. })
                if sample_id == "sample-0"
        ));
    }

    #[tokio::test]
    async fn runner_rejects_restarting_a_complete_trace() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let first = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![sample("sample-0", true)]),
            context(&path),
        );
        first.run().await.unwrap();

        let resumed = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![sample("sample-0", true)]),
            BenchmarkRunContext::resume(&path, "run-1").unwrap(),
        );
        assert!(matches!(
            resumed.run().await,
            Err(RunnerError::TraceAlreadyComplete)
        ));
    }

    #[tokio::test]
    async fn runner_rejects_zero_hit_budget_before_writing_a_step() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let mut invalid = sample("sample-0", true);
        invalid.query.budget.max_hits = 0;
        let runner = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![invalid]),
            context(&path),
        );

        assert!(matches!(
            runner.run().await,
            Err(RunnerError::InvalidSample(message)) if message.contains("max_hits")
        ));
        assert!(read_run(&path, "run-1").unwrap().steps.is_empty());
    }

    #[tokio::test]
    async fn runner_rejects_duplicate_dataset_sample_ids() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let runner = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![sample("duplicate", false), sample("duplicate", true)]),
            context(&path),
        );

        assert!(matches!(
            runner.run().await,
            Err(RunnerError::InvalidSample(message)) if message.contains("appears more than once")
        ));
        assert!(!read_run(&path, "run-1").unwrap().complete);
    }

    #[tokio::test]
    async fn runner_records_provider_timeout_in_the_trace() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let mut timeout_sample = sample("slow-sample", true);
        timeout_sample.query.budget.max_latency_ms = Some(5);
        let runner = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 50,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![timeout_sample]),
            context(&path),
        );

        let summary = runner.run().await.unwrap();
        assert_eq!(summary.retrieval_error_count, 1);
        let trace = read_run(&path, "run-1").unwrap();
        assert_eq!(trace.steps[0].observed_hits.len(), 0);
        assert_eq!(trace.steps[0].errors[0].code, "timeout");
        assert!(trace.steps[0].errors[0].retryable);
    }

    #[tokio::test]
    async fn no_memory_baseline_uses_the_same_trace_contract() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let capabilities = NoMemoryMethod.manifest().capabilities;
        assert!(capabilities.modalities.contains(&Modality::Action));
        assert!(capabilities.modalities.contains(&Modality::Trajectory));
        let runner = RetrievalRunner::new(
            NoMemoryMethod,
            VecDataset::new(vec![sample("sample-0", false), sample("sample-1", true)]),
            context_for_method(&path, "no-memory"),
        );

        let summary = runner.run().await.unwrap();
        let trace = read_run(&path, "run-1").unwrap();
        assert_eq!(summary.step_count, 2);
        assert!(trace.steps.iter().all(|step| step.observed_hits.is_empty()));
        assert!(trace
            .steps
            .iter()
            .all(|step| step.telemetry.evidence_bytes == 0));
    }

    #[tokio::test]
    async fn no_memory_baseline_accepts_continual_writes_as_no_ops() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let runner = ContinualEpisodeRunner::new(
            NoMemoryMethod,
            VecContinualDataset::new(vec![continual_sample("sample-0", true, "memory")]),
            context_for_method(&path, "no-memory"),
        );

        let summary = runner.run().await.unwrap();
        assert_eq!(summary.ingest_error_count, 0);
        let trace = read_run(&path, "run-1").unwrap();
        let ingest = trace.steps[0].ingest.as_ref().expect("ingest trace");
        assert!(ingest.item_ids.is_empty());
        assert!(trace.steps[0].observed_hits.is_empty());
    }

    #[tokio::test]
    async fn continual_runner_writes_then_reads_with_projection_method() {
        let directory = tempdir().unwrap();
        let store =
            Arc::new(JsonlProjectionStore::open(directory.path().join("memory.jsonl")).unwrap());
        let method = ProjectionMemoryMethod::new(store, "projection", "1").unwrap();
        let path = directory.path().join("trace.jsonl");
        let mut sample = continual_sample("sample-0", true, "online memory");
        // The projection provider intentionally does not implement semantic
        // content search; an online benchmark can still query by modality.
        sample.sample.query.input = None;
        sample.sample.query.modalities = vec![Modality::Text];

        let summary = ContinualEpisodeRunner::new(
            method,
            VecContinualDataset::new(vec![sample]),
            context_for_method(&path, "projection"),
        )
        .run()
        .await
        .unwrap();

        assert_eq!(summary.ingest_error_count, 0);
        assert_eq!(summary.retrieval_error_count, 0);
        let trace = read_run(&path, "run-1").unwrap();
        assert_eq!(trace.steps[0].observed_hits.len(), 1);
        assert_eq!(
            trace.steps[0].observed_hits[0].record_id,
            Uuid::from_u128(10).to_string()
        );
        assert!(trace.steps[0].ingest.as_ref().unwrap().error.is_none());
    }

    #[tokio::test]
    async fn runner_rejects_samples_after_terminal_step() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let runner = RetrievalRunner::new(
            StubMethod {
                calls: Arc::new(Mutex::new(Vec::new())),
                delay_ms: 0,
                supports_visibility_cutoff: true,
                truncated: false,
            },
            VecDataset::new(vec![sample("terminal", true), sample("late", false)]),
            context(&path),
        );

        assert!(matches!(
            runner.run().await,
            Err(RunnerError::InvalidSample(message))
                if message.contains("after the terminal")
        ));
        let trace = read_run(&path, "run-1").unwrap();
        assert!(!trace.complete);
        assert!(trace.steps.is_empty());
    }

    #[tokio::test]
    async fn runner_reads_records_from_the_dataset_namespace() {
        let directory = tempdir().unwrap();
        let store =
            Arc::new(JsonlProjectionStore::open(directory.path().join("memory.jsonl")).unwrap());
        let item_id = Uuid::from_u128(99);
        store
            .write(
                StoreWriteRequest {
                    records: vec![MemoryRecord {
                        item: MemoryItem {
                            id: item_id,
                            version: 1,
                            kind: MemoryKind::Semantic,
                            scope: MemoryScope::Shared,
                            modality: Modality::Text,
                            content: ContentRef::Text {
                                text: "stored in the frozen dataset".into(),
                            },
                            namespace: "bench/test".into(),
                            episode_id: Some("episode-1".into()),
                            subject: None,
                            source: "fixture".into(),
                            metadata: BTreeMap::new(),
                            abstraction_level: 0,
                            event_time: None,
                            ingest_time_ms: 50,
                            schema_version: 1,
                        },
                        episode: None,
                        evidence: Vec::new(),
                        provenance: MemoryProvenance {
                            producer: "fixture".into(),
                            source_id: "fixture".into(),
                            source_event_id: None,
                            parent_record_ids: Vec::new(),
                            method_version: Some("1".into()),
                        },
                    }],
                    relations: Vec::new(),
                    idempotency_key: "seed".into(),
                    source_revision: Some(1),
                },
                StorageContext {
                    namespace: "bench/test".into(),
                    run_id: Uuid::from_u128(1),
                    snapshot_id: None,
                    timeout_ms: None,
                },
            )
            .await
            .unwrap();

        let method = ProjectionMemoryMethod::new(store, "projection", "1").unwrap();
        let run = BenchmarkRun::new(
            "projection-run",
            BenchmarkManifest::new(
                "retrieval",
                "bench/test",
                MethodManifest {
                    method_id: "projection".into(),
                    version: "1".into(),
                    config: BTreeMap::new(),
                },
                "dataset-1",
                Some(100),
            ),
            1,
        );
        let trace_path = directory.path().join("trace.jsonl");
        let context = BenchmarkRunContext::open(&trace_path, run, Uuid::from_u128(2)).unwrap();
        let sample = DatasetSample {
            sample_id: "sample-0".into(),
            observation: AgentObservation { parts: Vec::new() },
            query: MemoryQuery {
                input: None,
                namespace: String::new(),
                scope: None,
                subject: None,
                episode_id: None,
                kinds: Vec::new(),
                modalities: vec![Modality::Text],
                time_range: None,
                visibility_cutoff_ms: None,
                graph: None,
                budget: MemoryBudget::default(),
            },
            session_id: None,
            episode_id: None,
            terminal: true,
            terminal_reason: Some(TerminalReason::Success),
        };
        let summary = RetrievalRunner::new(method, VecDataset::new(vec![sample]), context)
            .run()
            .await
            .unwrap();

        assert_eq!(summary.retrieval_error_count, 0);
        let trace = read_run(&trace_path, "projection-run").unwrap();
        assert_eq!(
            trace.steps[0]
                .query
                .as_ref()
                .unwrap()
                .dataset_namespace
                .as_deref(),
            Some("bench/test")
        );
        assert_eq!(
            trace.steps[0].observed_hits[0].record_id,
            item_id.to_string()
        );
        assert_eq!(trace.steps[0].observed_hits[0].version, 1);
    }

    #[test]
    fn observed_hits_exports_unique_locator_ids_in_stable_order() {
        let locator = |id: u128, frame_start: Option<u64>| EvidenceLocator {
            artifact: ArtifactRef {
                artifact_id: Uuid::from_u128(id),
                uri: format!("cas://artifact-{id}"),
                media_type: "video/mp4".into(),
                sha256: format!("sha256:{id}"),
                bytes: 1,
                codec: None,
                schema: None,
                stream_key: None,
                storage_revision: None,
            },
            entity_path: None,
            time_range: None,
            frame_start,
            frame_end: frame_start,
            step_start: None,
            step_end: None,
            byte_start: None,
            byte_end: None,
        };
        let first = locator(2, Some(10));
        let response = RetrieveResponse {
            query_id: Uuid::from_u128(9),
            hits: vec![MemoryHit {
                item_id: Uuid::from_u128(7),
                source_record_ids: Vec::new(),
                version: 1,
                method_id: "stub".into(),
                rank: 1,
                score: None,
                evidence: Vec::new(),
                evidence_refs: vec![first.clone(), first, locator(2, Some(20)), locator(1, None)],
                projection: None,
                provenance: "stub".into(),
                snapshot_id: None,
                latency_ms: 0,
            }],
            snapshot_id: None,
            latency_ms: 0,
            truncated: false,
            telemetry: None,
        };

        let hits = observed_hits(&response, 8).unwrap();

        assert_eq!(hits[0].evidence_ids.len(), 3);
        assert!(hits[0]
            .evidence_ids
            .iter()
            .all(|id| id.starts_with("artifact:") && id.contains("#sha256:")));
        assert_ne!(
            evidence_locator_identity(&locator(2, Some(10))).unwrap(),
            evidence_locator_identity(&locator(2, Some(20))).unwrap()
        );
    }

    #[test]
    fn evidence_locator_identity_is_stable_for_the_same_locator() {
        let locator = EvidenceLocator {
            artifact: ArtifactRef {
                artifact_id: Uuid::from_u128(7),
                uri: "cas://artifact-7".into(),
                media_type: "video/mp4".into(),
                sha256: "sha256:7".into(),
                bytes: 1,
                codec: None,
                schema: None,
                stream_key: None,
                storage_revision: None,
            },
            entity_path: Some("robot/gripper/object-1".into()),
            time_range: None,
            frame_start: Some(10),
            frame_end: Some(20),
            step_start: Some(3),
            step_end: Some(5),
            byte_start: None,
            byte_end: None,
        };

        assert_eq!(
            evidence_locator_identity(&locator).unwrap(),
            evidence_locator_identity(&locator).unwrap()
        );
    }

    #[test]
    fn observed_hits_rejects_duplicate_ranks_and_non_finite_scores() {
        let hit = |rank: usize, score: Option<f32>| MemoryHit {
            item_id: Uuid::from_u128(rank as u128),
            source_record_ids: Vec::new(),
            version: 1,
            method_id: "stub".into(),
            rank,
            score,
            evidence: Vec::new(),
            evidence_refs: Vec::new(),
            projection: None,
            provenance: "stub".into(),
            snapshot_id: None,
            latency_ms: 0,
        };

        let duplicate_rank = RetrieveResponse {
            query_id: Uuid::from_u128(1),
            hits: vec![hit(1, Some(0.5)), hit(1, Some(0.4))],
            snapshot_id: None,
            latency_ms: 0,
            truncated: false,
            telemetry: None,
        };
        assert!(matches!(
            observed_hits(&duplicate_rank, 8),
            Err(RunnerError::InvalidResponse(message)) if message.contains("rank")
        ));

        let non_finite = RetrieveResponse {
            query_id: Uuid::from_u128(2),
            hits: vec![hit(1, Some(f32::NAN))],
            snapshot_id: None,
            latency_ms: 0,
            truncated: false,
            telemetry: None,
        };
        assert!(matches!(
            observed_hits(&non_finite, 8),
            Err(RunnerError::InvalidResponse(message)) if message.contains("non-finite")
        ));
    }

    #[test]
    fn observed_hits_rejects_duplicate_source_record_ids() {
        let response = RetrieveResponse {
            query_id: Uuid::from_u128(8),
            hits: vec![MemoryHit {
                item_id: Uuid::from_u128(1),
                source_record_ids: vec![Uuid::from_u128(2), Uuid::from_u128(2)],
                version: 1,
                method_id: "stub".into(),
                rank: 1,
                score: Some(0.5),
                evidence: Vec::new(),
                evidence_refs: Vec::new(),
                projection: None,
                provenance: "stub".into(),
                snapshot_id: None,
                latency_ms: 0,
            }],
            snapshot_id: None,
            latency_ms: 0,
            truncated: false,
            telemetry: None,
        };

        assert!(matches!(
            observed_hits(&response, 8),
            Err(RunnerError::InvalidResponse(message))
                if message.contains("source record id")
        ));
    }

    #[test]
    fn observed_hits_rejects_zero_record_versions() {
        let zero_version = MemoryHit {
            item_id: Uuid::from_u128(1),
            source_record_ids: Vec::new(),
            version: 0,
            method_id: "stub".into(),
            rank: 1,
            score: Some(0.5),
            evidence: Vec::new(),
            evidence_refs: Vec::new(),
            projection: None,
            provenance: "stub".into(),
            snapshot_id: None,
            latency_ms: 0,
        };
        let response = RetrieveResponse {
            query_id: Uuid::from_u128(3),
            hits: vec![zero_version],
            snapshot_id: None,
            latency_ms: 0,
            truncated: false,
            telemetry: None,
        };

        assert!(matches!(
            observed_hits(&response, 8),
            Err(RunnerError::InvalidResponse(message)) if message.contains("version")
        ));
    }

    #[test]
    fn observed_hits_rejects_results_over_the_query_budget() {
        let hit = |rank: usize| MemoryHit {
            item_id: Uuid::from_u128(rank as u128),
            source_record_ids: Vec::new(),
            version: 1,
            method_id: "stub".into(),
            rank,
            score: Some(0.5),
            evidence: Vec::new(),
            evidence_refs: Vec::new(),
            projection: None,
            provenance: "stub".into(),
            snapshot_id: None,
            latency_ms: 0,
        };
        let response = RetrieveResponse {
            query_id: Uuid::from_u128(4),
            hits: vec![hit(1), hit(2)],
            snapshot_id: None,
            latency_ms: 0,
            truncated: true,
            telemetry: None,
        };

        assert!(matches!(
            observed_hits(&response, 1),
            Err(RunnerError::InvalidResponse(message)) if message.contains("max_hits")
        ));
    }

    #[test]
    fn evidence_budget_rejects_over_budget_response_even_when_truncated() {
        let response = RetrieveResponse {
            query_id: Uuid::from_u128(6),
            hits: vec![MemoryHit {
                item_id: Uuid::from_u128(1),
                source_record_ids: Vec::new(),
                version: 1,
                method_id: "stub".into(),
                rank: 1,
                score: Some(0.5),
                evidence: Vec::new(),
                evidence_refs: Vec::new(),
                projection: Some(ContentRef::Text {
                    text: "evidence that exceeds the frozen query budget".into(),
                }),
                provenance: "stub".into(),
                snapshot_id: None,
                latency_ms: 0,
            }],
            snapshot_id: None,
            latency_ms: 0,
            truncated: true,
            telemetry: None,
        };

        assert!(matches!(
            validate_evidence_budget(&response, 1),
            Err(RunnerError::InvalidResponse(message))
                if message.contains("max_evidence_bytes")
        ));
    }

    #[test]
    fn evidence_budget_accepts_exact_limit_and_reports_returned_bytes() {
        let response = RetrieveResponse {
            query_id: Uuid::from_u128(7),
            hits: vec![MemoryHit {
                item_id: Uuid::from_u128(1),
                source_record_ids: Vec::new(),
                version: 1,
                method_id: "stub".into(),
                rank: 1,
                score: Some(0.5),
                evidence: Vec::new(),
                evidence_refs: Vec::new(),
                projection: Some(ContentRef::Text {
                    text: "bounded evidence".into(),
                }),
                provenance: "stub".into(),
                snapshot_id: None,
                latency_ms: 0,
            }],
            snapshot_id: None,
            latency_ms: 0,
            truncated: false,
            telemetry: None,
        };
        let returned_bytes = response_evidence_bytes(&response);

        assert_eq!(
            validate_evidence_budget(&response, returned_bytes as usize).unwrap(),
            returned_bytes
        );
    }

    #[test]
    fn observed_hits_rejects_non_contiguous_ranks() {
        let hit = |item_id: u128, rank: usize| MemoryHit {
            item_id: Uuid::from_u128(item_id),
            source_record_ids: Vec::new(),
            version: 1,
            method_id: "stub".into(),
            rank,
            score: Some(0.5),
            evidence: Vec::new(),
            evidence_refs: Vec::new(),
            projection: None,
            provenance: "stub".into(),
            snapshot_id: None,
            latency_ms: 0,
        };
        let response = RetrieveResponse {
            query_id: Uuid::from_u128(5),
            hits: vec![hit(1, 1), hit(2, 3)],
            snapshot_id: None,
            latency_ms: 0,
            truncated: false,
            telemetry: None,
        };

        assert!(matches!(
            observed_hits(&response, 4),
            Err(RunnerError::InvalidResponse(message)) if message.contains("contiguous")
        ));
    }

    #[test]
    fn dataset_digest_binding_is_bidirectional() {
        let digest = "sha256:dataset";
        assert!(validate_dataset_digest(None, None).is_ok());
        assert!(matches!(
            validate_dataset_digest(None, Some(digest)),
            Err(RunnerError::DatasetDigestMissing)
        ));
        assert!(matches!(
            validate_dataset_digest(Some(digest), None),
            Err(RunnerError::DatasetDigestUnavailable { expected })
                if expected == digest
        ));
        assert!(validate_dataset_digest(Some(digest), Some(digest)).is_ok());
        assert!(matches!(
            validate_dataset_digest(Some(digest), Some("sha256:other")),
            Err(RunnerError::DatasetDigestMismatch { expected, actual })
                if expected == digest && actual == "sha256:other"
        ));
    }
}
