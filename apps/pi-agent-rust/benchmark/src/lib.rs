//! Reproducible benchmark contracts for pluggable memory methods.
//!
//! The crate deliberately stops at recording. It does not call a model or
//! calculate evaluator metrics. A benchmark run is an append-only sequence of
//! JSONL records that can be replayed by a later evaluator.

mod dataset;
mod runner;

pub use dataset::{DatasetManifest, DatasetManifestError, DATASET_MANIFEST_SCHEMA_VERSION};
pub use runner::BenchmarkRunContext;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use thiserror::Error;
use uuid::Uuid;

use pi_agent_memory::{MemoryCapabilities, ProviderTelemetry};

pub const BENCHMARK_SCHEMA_VERSION: u32 = 1;

/// A memory method's identity and configuration at experiment time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MethodManifest {
    pub method_id: String,
    pub version: String,
    #[serde(default)]
    pub config: BTreeMap<String, String>,
}

impl MethodManifest {
    /// Records provider-discovered settings in the method identity.
    ///
    /// Provider settings use a reserved `provider.` prefix so they cannot be
    /// confused with benchmark adapter options. An empty map is explicit: it
    /// means the provider did not advertise reproducibility settings rather
    /// than that the caller guessed them locally.
    pub fn bind_provider_config(&mut self, provider_config: &BTreeMap<String, String>) {
        self.config.insert(
            "provider_config_source".to_owned(),
            if provider_config.is_empty() {
                "provider_manifest_missing".to_owned()
            } else {
                "provider_manifest".to_owned()
            },
        );
        for (key, value) in provider_config {
            self.config.insert(format!("provider.{key}"), value.clone());
        }
    }
}

/// Declares where Blob reference integrity was enforced for a benchmark run.
///
/// This is part of the run identity because a local content-addressed store,
/// a remote service resolver, and an unchecked reference boundary do not have
/// the same admission semantics. `Unspecified` is retained for manifests
/// written before this field existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactValidationPolicy {
    #[default]
    Unspecified,
    LocalStore,
    RemoteService,
    Unchecked,
}

/// Orders the memory operations that a benchmark run performs.
///
/// The default keeps manifests written before lifecycle support compatible:
/// they describe a retrieval-only run whose memory projection was prepared by
/// the caller. Offline build records are accepted only for the explicit
/// `OfflineBuildThenQuery` lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkLifecycle {
    #[default]
    RetrievalOnly,
    Continual,
    OfflineBuildThenQuery,
}

impl ArtifactValidationPolicy {
    pub fn is_unspecified(&self) -> bool {
        matches!(self, Self::Unspecified)
    }
}

/// Immutable inputs that define a benchmark comparison.
///
/// `visibility_cutoff_ms` is part of the manifest so a rerun cannot silently
/// read memories ingested after the dataset revision was frozen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkManifest {
    pub schema_version: u32,
    pub benchmark_id: String,
    pub namespace: String,
    pub method: MethodManifest,
    pub dataset_revision: String,
    #[serde(default)]
    pub dataset_digest: Option<String>,
    /// Structured dataset identity. The legacy revision and digest fields
    /// remain so older traces can still be read and compared.
    #[serde(default)]
    pub dataset: Option<DatasetManifest>,
    /// Lifecycle used to construct and query the memory projection.
    #[serde(default)]
    pub lifecycle: BenchmarkLifecycle,
    #[serde(
        default,
        skip_serializing_if = "ArtifactValidationPolicy::is_unspecified"
    )]
    pub artifact_validation_policy: ArtifactValidationPolicy,
    /// Digest of the evaluator reference sidecar bound to this run.
    #[serde(default)]
    pub reference_digest: Option<String>,
    /// Optional digest sealing both benchmark-level and method-level config.
    ///
    /// Older manifests omit this field. Call [`Self::seal_config`] when a
    /// producer wants later readers to detect configuration drift.
    #[serde(default)]
    pub config_digest: Option<String>,
    pub visibility_cutoff_ms: Option<u64>,
    #[serde(default)]
    pub config: BTreeMap<String, String>,
    #[serde(default)]
    pub metrics: Vec<MetricSpec>,
}

impl BenchmarkManifest {
    pub fn new(
        benchmark_id: impl Into<String>,
        namespace: impl Into<String>,
        method: MethodManifest,
        dataset_revision: impl Into<String>,
        visibility_cutoff_ms: Option<u64>,
    ) -> Self {
        Self {
            schema_version: BENCHMARK_SCHEMA_VERSION,
            benchmark_id: benchmark_id.into(),
            namespace: namespace.into(),
            method,
            dataset_revision: dataset_revision.into(),
            dataset_digest: None,
            dataset: None,
            lifecycle: BenchmarkLifecycle::RetrievalOnly,
            // New runs state their boundary explicitly. Older JSON manifests
            // that omit this field still deserialize as `Unspecified`.
            artifact_validation_policy: ArtifactValidationPolicy::Unchecked,
            reference_digest: None,
            config_digest: None,
            visibility_cutoff_ms,
            config: BTreeMap::new(),
            metrics: Vec::new(),
        }
    }

    /// Binds structured dataset identity to this benchmark manifest.
    pub fn bind_dataset(&mut self, dataset: DatasetManifest) -> Result<(), TraceStoreError> {
        dataset
            .validate()
            .map_err(|error| TraceStoreError::InvalidManifest(error.to_string()))?;

        if let Some(existing) = &self.dataset {
            if existing != &dataset {
                return Err(TraceStoreError::InvalidManifest(
                    "benchmark already binds a different dataset manifest".to_owned(),
                ));
            }
            return Ok(());
        }
        if !self.dataset_revision.is_empty() && self.dataset_revision != dataset.revision {
            return Err(TraceStoreError::InvalidManifest(
                "dataset manifest revision does not match dataset_revision".to_owned(),
            ));
        }
        if let Some(existing_digest) = &self.dataset_digest {
            if existing_digest != &dataset.content_digest {
                return Err(TraceStoreError::InvalidManifest(
                    "dataset manifest digest does not match dataset_digest".to_owned(),
                ));
            }
        }
        self.dataset_revision = dataset.revision.clone();
        self.dataset_digest = Some(dataset.content_digest.clone());
        self.dataset = Some(dataset);
        Ok(())
    }

    /// Checks the artifact boundary required before creating a new trace.
    ///
    /// This is deliberately separate from the low-level record validation:
    /// old traces may omit the policy field and must remain readable. New runs
    /// that declare binary-capable dataset modalities must say where Blob
    /// references were validated.
    pub fn validate_new_artifact_policy(&self) -> Result<(), TraceStoreError> {
        if self
            .dataset
            .as_ref()
            .is_some_and(DatasetManifest::contains_binary_modality)
            && self.artifact_validation_policy == ArtifactValidationPolicy::Unspecified
        {
            return Err(TraceStoreError::InvalidManifest(
                "binary-modality dataset requires an explicit artifact_validation_policy"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Returns a stable digest for all configuration that can change the
    /// method's result. `BTreeMap` keeps key ordering deterministic across
    /// processes and platforms.
    pub fn config_digest(&self) -> String {
        #[derive(Serialize)]
        struct Config<'a> {
            benchmark: &'a BTreeMap<String, String>,
            method: &'a BTreeMap<String, String>,
        }

        let bytes = serde_json::to_vec(&Config {
            benchmark: &self.config,
            method: &self.method.config,
        })
        .expect("benchmark config maps are always JSON serializable");
        format!("sha256:{:x}", Sha256::digest(bytes))
    }

    /// Seals the current configuration into the manifest.
    pub fn seal_config(&mut self) {
        self.config_digest = Some(self.config_digest());
    }

    /// Checks the optional configuration seal and basic key/value hygiene.
    /// This remains opt-in for the required reproducibility keys so existing
    /// manifests that predate model metadata remain readable.
    pub fn verify_config_digest(&self) -> Result<(), TraceStoreError> {
        for (key, value) in self.config.iter().chain(self.method.config.iter()) {
            if key.trim().is_empty() || value.trim().is_empty() {
                return Err(TraceStoreError::InvalidManifest(
                    "configuration keys and values must be non-empty".to_owned(),
                ));
            }
        }
        if let Some(capabilities_json) = self.method.config.get("capabilities_json") {
            serde_json::from_str::<MemoryCapabilities>(capabilities_json).map_err(|error| {
                TraceStoreError::InvalidManifest(format!(
                    "method config capabilities_json is not valid MemoryCapabilities JSON: {error}"
                ))
            })?;
        }
        if let Some(expected) = &self.config_digest {
            let actual = self.config_digest();
            if expected != &actual {
                return Err(TraceStoreError::InvalidManifest(format!(
                    "config_digest {expected:?} does not match computed digest {actual:?}"
                )));
            }
        }
        Ok(())
    }

    /// Validates the minimum metadata needed for a reproducible model run.
    /// `prompt_digest` is accepted instead of the raw prompt to keep large or
    /// private prompts out of the manifest.
    pub fn validate_reproducibility_config(&self) -> Result<(), TraceStoreError> {
        self.verify_config_digest()?;
        for key in ["model", "seed"] {
            if !self.config.contains_key(key) {
                return Err(TraceStoreError::InvalidManifest(format!(
                    "reproducibility config is missing {key:?}"
                )));
            }
        }
        if !self.config.contains_key("prompt") && !self.config.contains_key("prompt_digest") {
            return Err(TraceStoreError::InvalidManifest(
                "reproducibility config is missing \"prompt\" or \"prompt_digest\"".to_owned(),
            ));
        }
        Ok(())
    }
}

/// One concrete execution of a benchmark manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub run_id: String,
    pub manifest: BenchmarkManifest,
    pub started_at_ms: i64,
    pub finished_at_ms: Option<i64>,
    /// UUID forwarded to memory providers for this concrete execution.
    /// Older traces may omit it and are rejected by `BenchmarkRunContext`.
    #[serde(default)]
    pub operation_run_id: Option<Uuid>,
}

impl BenchmarkRun {
    pub fn new(run_id: impl Into<String>, manifest: BenchmarkManifest, started_at_ms: i64) -> Self {
        Self {
            run_id: run_id.into(),
            manifest,
            started_at_ms,
            finished_at_ms: None,
            operation_run_id: None,
        }
    }

    /// Namespace used by storage and artifact adapters for this concrete run.
    /// Keeping the run id in the namespace prevents two runs from sharing a
    /// projection even when they use the same benchmark dataset namespace. The
    /// run component is encoded so an opaque run id cannot introduce another
    /// path segment when it is passed to a filesystem-backed artifact store.
    pub fn isolated_namespace(&self) -> String {
        format!(
            "{}/{}",
            self.manifest.namespace.trim_end_matches('/'),
            encode_namespace_component(&self.run_id)
        )
    }
}

fn encode_namespace_component(value: &str) -> String {
    let mut encoded = String::from("r");
    for byte in value.bytes() {
        encoded.push(char::from(b"0123456789abcdef"[(byte >> 4) as usize]));
        encoded.push(char::from(b"0123456789abcdef"[(byte & 0x0f) as usize]));
    }
    encoded
}

/// Agent-visible content. Reference answers intentionally have no field in
/// this type and therefore cannot be serialized into a step observation by
/// this crate's API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentObservation {
    #[serde(default)]
    pub parts: Vec<ObservationPart>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum ObservationPart {
    Text(String),
    Artifact {
        artifact_id: String,
        modality: String,
    },
}

/// A memory hit exposed to the agent/evaluator trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservedHit {
    pub record_id: String,
    /// Canonical source records supporting this provider hit.
    ///
    /// Providers that preserve source identity populate this field with the
    /// original benchmark record identifiers.  It is intentionally separate
    /// from `record_id`, which remains the provider's own result identifier.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_record_ids: Vec<String>,
    /// Version returned by the memory method at retrieval time.
    #[serde(default)]
    pub version: u64,
    pub rank: u32,
    pub score: Option<f64>,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
}

/// The structural query recorded for one retrieval step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryTrace {
    /// Namespace identifying the concrete benchmark run in the trace.
    pub namespace: String,
    /// Namespace containing the immutable dataset read by this query.
    #[serde(default)]
    pub dataset_namespace: Option<String>,
    /// Digest of the complete serialized query after namespace binding.
    #[serde(default)]
    pub query_digest: Option<String>,
    /// Provider snapshot observed while executing this query.
    ///
    /// Older traces omit this field and are interpreted as having no snapshot
    /// contract.
    #[serde(default)]
    pub snapshot_id: Option<String>,
    pub input_digest: Option<String>,
    pub visibility_cutoff_ms: Option<u64>,
    pub requested_hits: u32,
    /// Whether the single memory provider returned only a bounded prefix.
    ///
    /// Older JSONL traces omit this field and are interpreted as complete.
    #[serde(default)]
    pub retrieval_truncated: bool,
    /// Provider-side resource measurements for this retrieval operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<ProviderTelemetry>,
}

/// The action emitted by the agent or environment for one step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionTrace {
    pub action_type: String,
    pub action_digest: Option<String>,
    /// Ordered tool calls that made up this action. Arguments and results are
    /// represented by digests so traces remain compact and auditable without
    /// persisting secrets or large tool payloads.
    #[serde(default)]
    pub calls: Vec<ActionCallTrace>,
}

/// One tool invocation contributing to an [`ActionTrace`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionCallTrace {
    pub step: usize,
    pub call_id: String,
    pub tool_name: String,
    pub arguments_digest: Option<String>,
    pub result_digest: Option<String>,
    /// `None` means the runtime has not emitted a terminal tool result yet.
    #[serde(default)]
    pub is_error: Option<bool>,
    /// `None` means the runtime has not emitted a terminal tool result yet.
    #[serde(default)]
    pub terminate: Option<bool>,
}

/// Reward kept separate from action and terminal state for embodied tasks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewardTrace {
    pub value: f64,
    #[serde(default)]
    pub components: BTreeMap<String, f64>,
}

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

/// Resource measurements needed to compare methods fairly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct StepTelemetry {
    /// Time spent admitting the current sample into an online memory.
    /// Retrieval-only traces deserialize this as zero for compatibility.
    #[serde(default)]
    pub ingest_latency_ms: u64,
    pub retrieval_latency_ms: u64,
    #[serde(default)]
    pub agent_latency_ms: u64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// Serialized evidence metadata exposed to the agent. Artifact payload
    /// bytes are not counted unless an adapter actually loads them.
    pub evidence_bytes: u64,
}

/// The execution stage that produced a step error.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceErrorStage {
    #[default]
    Retrieval,
    /// A failure while admitting the current sample into an online memory.
    Ingest,
    /// A failure while constructing an offline memory projection.
    Build,
    Agent,
    Tool,
    Dataset,
    Runtime,
}

/// An error observed while executing one benchmark step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceErrorEntry {
    /// Older traces omitted this field and are interpreted as retrieval errors.
    #[serde(default)]
    pub stage: TraceErrorStage,
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

/// The auditable result of the write performed before one continual-memory
/// retrieval step.
///
/// The request digest binds this receipt to the exact records submitted by
/// the runner.  Item and relation identifiers make the write replayable even
/// when a provider does not expose a human-readable log.  Older traces omit
/// this field because retrieval-only runs do not ingest data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestTrace {
    pub request_digest: String,
    #[serde(default)]
    pub item_ids: Vec<String>,
    #[serde(default)]
    pub relation_ids: Vec<String>,
    pub deduplicated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_id: Option<String>,
    /// Provider-side resource measurements for this ingest operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<ProviderTelemetry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<TraceErrorEntry>,
}

/// The auditable result of one request in an offline memory-build phase.
///
/// Build records are kept separate from [`StepTrace`] because their requests
/// happen before any query is executed. The request digest binds this receipt
/// to the exact records supplied by the build dataset; the source sample
/// digest makes interrupted builds safe to resume without silently changing
/// their input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildTrace {
    pub run_id: String,
    pub build_index: u64,
    pub sample_id: String,
    /// Required for newly written offline-build records. It remains optional
    /// in the wire type so future readers can still decode older records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_digest: Option<String>,
    pub namespace: String,
    pub request_digest: String,
    #[serde(default)]
    pub item_ids: Vec<String>,
    #[serde(default)]
    pub relation_ids: Vec<String>,
    pub deduplicated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_id: Option<String>,
    /// Provider-side resource measurements for this offline build operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<ProviderTelemetry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at_ms: Option<i64>,
    pub latency_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<TraceErrorEntry>,
}

/// Validates the durable shape of an online-memory write receipt.
///
/// The request itself is intentionally not persisted in a trace. Its digest
/// is the binding to the dataset input, while UUIDs and the error stage make
/// the receipt independently auditable after the provider is gone.
pub fn validate_ingest_trace(
    ingest: &IngestTrace,
    step_errors: &[TraceErrorEntry],
) -> Result<(), TraceStoreError> {
    if !is_sha256_digest(&ingest.request_digest) {
        return Err(TraceStoreError::InvalidStep(
            "ingest request_digest must be a sha256 digest".to_string(),
        ));
    }
    if ingest.item_ids.iter().any(|id| id.trim().is_empty())
        || ingest.relation_ids.iter().any(|id| id.trim().is_empty())
        || ingest
            .item_ids
            .iter()
            .chain(ingest.relation_ids.iter())
            .any(|id| Uuid::parse_str(id).is_err())
    {
        return Err(TraceStoreError::InvalidStep(
            "ingest receipt ids must be UUIDs".to_string(),
        ));
    }
    let mut receipt_ids = std::collections::BTreeSet::new();
    if ingest
        .item_ids
        .iter()
        .chain(ingest.relation_ids.iter())
        .any(|id| !receipt_ids.insert(id))
    {
        return Err(TraceStoreError::InvalidStep(
            "ingest receipt ids must be unique".to_string(),
        ));
    }
    if ingest
        .snapshot_id
        .as_deref()
        .is_some_and(|snapshot| snapshot.trim().is_empty())
    {
        return Err(TraceStoreError::InvalidStep(
            "ingest snapshot_id must be non-empty when present".to_string(),
        ));
    }
    if ingest
        .completed_at_ms
        .is_some_and(|timestamp| timestamp < 0)
    {
        return Err(TraceStoreError::InvalidStep(
            "ingest completed_at_ms must be non-negative".to_string(),
        ));
    }
    if let Some(error) = &ingest.error {
        if error.stage != TraceErrorStage::Ingest {
            return Err(TraceStoreError::InvalidStep(
                "ingest error must use the ingest stage".to_string(),
            ));
        }
        if error.code.trim().is_empty() || error.message.trim().is_empty() {
            return Err(TraceStoreError::InvalidStep(
                "ingest error code and message must be non-empty".to_string(),
            ));
        }
        if !ingest.item_ids.is_empty()
            || !ingest.relation_ids.is_empty()
            || ingest.snapshot_id.is_some()
        {
            return Err(TraceStoreError::InvalidStep(
                "failed ingest must not contain receipt ids or a snapshot".to_string(),
            ));
        }
        if !step_errors.iter().any(|candidate| candidate == error) {
            return Err(TraceStoreError::InvalidStep(
                "ingest error must also appear in step errors".to_string(),
            ));
        }
    } else if step_errors
        .iter()
        .any(|error| error.stage == TraceErrorStage::Ingest)
    {
        return Err(TraceStoreError::InvalidStep(
            "step contains an ingest error without an ingest trace".to_string(),
        ));
    }
    Ok(())
}

/// Validates one offline-build receipt before it is persisted.
///
/// Unlike an online [`IngestTrace`], a build receipt carries its own sample
/// identity and error record because it is not nested in a query step.
pub fn validate_build_trace(build: &BuildTrace, run: &BenchmarkRun) -> Result<(), TraceStoreError> {
    if build.run_id != run.run_id {
        return Err(TraceStoreError::RunMismatch {
            expected: run.run_id.clone(),
            actual: build.run_id.clone(),
        });
    }
    if build.sample_id.trim().is_empty() {
        return Err(TraceStoreError::InvalidStep(
            "build sample_id must be non-empty".to_string(),
        ));
    }
    let Some(sample_digest) = build.sample_digest.as_deref() else {
        return Err(TraceStoreError::InvalidStep(
            "build sample_digest must be a sha256 digest".to_string(),
        ));
    };
    if !is_sha256_digest(sample_digest) {
        return Err(TraceStoreError::InvalidStep(
            "build sample_digest must be a sha256 digest".to_string(),
        ));
    }
    let expected_namespace = run.isolated_namespace();
    if build.namespace != expected_namespace {
        return Err(TraceStoreError::NamespaceMismatch {
            expected: expected_namespace,
            actual: build.namespace.clone(),
        });
    }
    if !is_sha256_digest(&build.request_digest) {
        return Err(TraceStoreError::InvalidStep(
            "build request_digest must be a sha256 digest".to_string(),
        ));
    }
    if build.item_ids.iter().any(|id| id.trim().is_empty())
        || build.relation_ids.iter().any(|id| id.trim().is_empty())
        || build
            .item_ids
            .iter()
            .chain(build.relation_ids.iter())
            .any(|id| Uuid::parse_str(id).is_err())
    {
        return Err(TraceStoreError::InvalidStep(
            "build receipt ids must be UUIDs".to_string(),
        ));
    }
    let mut receipt_ids = std::collections::BTreeSet::new();
    if build
        .item_ids
        .iter()
        .chain(build.relation_ids.iter())
        .any(|id| !receipt_ids.insert(id))
    {
        return Err(TraceStoreError::InvalidStep(
            "build receipt ids must be unique".to_string(),
        ));
    }
    if build
        .snapshot_id
        .as_deref()
        .is_some_and(|snapshot| snapshot.trim().is_empty())
    {
        return Err(TraceStoreError::InvalidStep(
            "build snapshot_id must be non-empty when present".to_string(),
        ));
    }
    if build.completed_at_ms.is_some_and(|timestamp| timestamp < 0) {
        return Err(TraceStoreError::InvalidStep(
            "build completed_at_ms must be non-negative".to_string(),
        ));
    }
    if let Some(error) = &build.error {
        if error.stage != TraceErrorStage::Build {
            return Err(TraceStoreError::InvalidStep(
                "build error must use the build stage".to_string(),
            ));
        }
        if error.code.trim().is_empty() || error.message.trim().is_empty() {
            return Err(TraceStoreError::InvalidStep(
                "build error code and message must be non-empty".to_string(),
            ));
        }
        if !build.item_ids.is_empty()
            || !build.relation_ids.is_empty()
            || build.snapshot_id.is_some()
        {
            return Err(TraceStoreError::InvalidStep(
                "failed build must not contain receipt ids or a snapshot".to_string(),
            ));
        }
    }
    Ok(())
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

/// Correlates one benchmark step with the agent runtime execution that
/// produced its answer. Retrieval-only executors leave this field absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStepLink {
    pub runtime_run_id: Uuid,
    pub turn_id: Uuid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_decision: Option<RuntimeMemoryDecisionLink>,
    /// All memory decisions emitted while producing this step.
    ///
    /// `memory_decision` remains as the first decision for readers that only
    /// understand the original one-decision shape. New writers populate both
    /// fields and readers prefer this complete list when it is present.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory_decisions: Vec<RuntimeMemoryDecisionLink>,
}

/// The durable runtime event that admitted or skipped memory for a turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeMemoryDecisionLink {
    pub proposal_id: Uuid,
    pub event_sequence: u64,
    pub decision_kind: String,
}

/// A deterministic, append-only record for one dataset sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepTrace {
    pub run_id: String,
    pub step_index: u64,
    pub sample_id: String,
    /// Digest of the complete bound dataset sample consumed for this step.
    /// Older traces omit it and cannot be resumed safely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_digest: Option<String>,
    pub namespace: String,
    pub visibility_cutoff_ms: Option<u64>,
    pub observation: AgentObservation,
    /// Optional online-memory write performed before retrieval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ingest: Option<IngestTrace>,
    pub query: Option<QueryTrace>,
    #[serde(default)]
    pub observed_hits: Vec<ObservedHit>,
    pub agent_answer: Option<String>,
    pub action: Option<ActionTrace>,
    pub reward: Option<RewardTrace>,
    pub terminal_reason: Option<TerminalReason>,
    pub telemetry: StepTelemetry,
    #[serde(default)]
    pub errors: Vec<TraceErrorEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_link: Option<RuntimeStepLink>,
    pub terminal: bool,
}

/// A metric definition is recorded before evaluator code runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricSpec {
    pub name: String,
    pub version: String,
    pub unit: String,
    pub direction: MetricDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
}

/// A metric result can be appended after the run without rewriting steps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricValue {
    pub run_id: String,
    pub name: String,
    pub value: f64,
    /// Number of samples represented by this value. Zero means that the
    /// metric is not applicable for this run and `value` must be zero; this
    /// keeps dataset-specific subsets out of weighted aggregates.
    pub sample_count: u64,
}

/// Marks the append-only trace as complete after a terminal step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunFinished {
    pub run_id: String,
    pub finished_at_ms: i64,
}

/// JSONL record tags keep the file append-only while allowing a reader to
/// replay manifests, steps, and post-run metric values in original order.
// Keep `Step` by value to preserve the append-only JSONL contract and public
// record shape; this enum intentionally has one larger variant.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "record_type", content = "record")]
pub enum TraceRecord {
    Run(BenchmarkRun),
    Build(BuildTrace),
    Step(StepTrace),
    Metric(MetricValue),
    Finished(RunFinished),
}

impl TraceRecord {
    fn run_id(&self) -> &str {
        match self {
            Self::Run(run) => &run.run_id,
            Self::Build(build) => &build.run_id,
            Self::Step(step) => &step.run_id,
            Self::Metric(metric) => &metric.run_id,
            Self::Finished(finished) => &finished.run_id,
        }
    }
}

#[derive(Debug, Error)]
pub enum TraceStoreError {
    #[error("trace I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("trace JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
    #[error("record run_id {actual:?} does not match writer run_id {expected:?}")]
    RunMismatch { expected: String, actual: String },
    #[error("run {0:?} was not found in trace")]
    RunNotFound(String),
    #[error("trace contains more than one Run record for run {0:?}")]
    DuplicateRun(String),
    #[error("run {0:?} has not been started")]
    RunNotStarted(String),
    #[error("step index {actual} does not follow expected index {expected}")]
    StepOutOfOrder { expected: u64, actual: u64 },
    #[error("build index {actual} does not follow expected index {expected}")]
    BuildOutOfOrder { expected: u64, actual: u64 },
    #[error("trace cannot append a build record after the first step")]
    BuildAfterSteps,
    #[error("build records require the offline_build_then_query lifecycle")]
    BuildRequiresOfflineLifecycle,
    #[error("trace cannot append after a terminal step")]
    TraceAfterTerminal,
    #[error("trace run has already been finalized")]
    RunAlreadyFinished,
    #[error("run {0:?} cannot be finalized before a terminal step")]
    RunNotTerminal(String),
    #[error("trace namespace {actual:?} does not match run namespace {expected:?}")]
    NamespaceMismatch { expected: String, actual: String },
    #[error("trace visibility cutoff does not match the benchmark manifest")]
    CutoffMismatch,
    #[error("metric {0:?} is not declared by the benchmark manifest")]
    UnknownMetric(String),
    #[error("invalid benchmark metric: {0}")]
    InvalidMetric(String),
    #[error("terminal step must include a terminal reason")]
    MissingTerminalReason,
    #[error("invalid benchmark step: {0}")]
    InvalidStep(String),
    #[error("trace writer lock is already held: {0}")]
    LockHeld(PathBuf),
    #[error("invalid benchmark manifest: {0}")]
    InvalidManifest(String),
}

/// An append-only writer bound to one run. Binding prevents accidental mixing
/// of two experiments in the same trace file through one writer instance.
#[derive(Debug)]
pub struct JsonlTraceWriter {
    path: PathBuf,
    run_id: String,
    writer: BufWriter<File>,
    state: TraceState,
    _lock: TraceWriteLock,
}

#[derive(Debug)]
struct TraceWriteLock {
    #[allow(dead_code)]
    path: PathBuf,
    #[cfg(unix)]
    file: File,
}

impl Drop for TraceWriteLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            // The lock file intentionally remains as a stable coordination
            // path. The kernel lock is released automatically on process
            // exit, including an ungraceful crash.
            let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        }
        #[cfg(not(unix))]
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, Default)]
struct TraceState {
    run: Option<BenchmarkRun>,
    last_build_index: Option<u64>,
    last_step_index: Option<u64>,
    build_phase_closed: bool,
    terminal: bool,
    finished_at_ms: Option<i64>,
}

impl JsonlTraceWriter {
    pub fn open(
        path: impl AsRef<Path>,
        run_id: impl Into<String>,
    ) -> Result<Self, TraceStoreError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let run_id = run_id.into();
        let lock = acquire_trace_lock(&path)?;
        truncate_incomplete_tail(&path)?;
        let existing = if path.exists() {
            read_all(&path)?
        } else {
            Vec::new()
        };
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let mut writer = Self {
            path,
            run_id,
            writer: BufWriter::new(file),
            state: TraceState::default(),
            _lock: lock,
        };
        for record in existing {
            if record.run_id() == writer.run_id {
                validate_record(&writer.state, &writer.run_id, &record)?;
                apply_record(&mut writer.state, record);
            }
        }
        Ok(writer)
    }

    pub fn append(&mut self, record: &TraceRecord) -> Result<(), TraceStoreError> {
        if record.run_id() != self.run_id {
            return Err(TraceStoreError::RunMismatch {
                expected: self.run_id.clone(),
                actual: record.run_id().to_owned(),
            });
        }
        validate_record(&self.state, &self.run_id, record)?;
        serde_json::to_writer(&mut self.writer, record)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        apply_record(&mut self.state, record.clone());
        Ok(())
    }

    pub fn append_run(&mut self, run: &BenchmarkRun) -> Result<(), TraceStoreError> {
        self.append(&TraceRecord::Run(run.clone()))
    }

    pub fn append_step(&mut self, step: &StepTrace) -> Result<(), TraceStoreError> {
        self.append(&TraceRecord::Step(step.clone()))
    }

    pub fn append_build(&mut self, build: &BuildTrace) -> Result<(), TraceStoreError> {
        self.append(&TraceRecord::Build(build.clone()))
    }

    pub fn append_metric(&mut self, value: &MetricValue) -> Result<(), TraceStoreError> {
        self.append(&TraceRecord::Metric(value.clone()))
    }

    pub fn finalize(&mut self, finished_at_ms: i64) -> Result<(), TraceStoreError> {
        self.append(&TraceRecord::Finished(RunFinished {
            run_id: self.run_id.clone(),
            finished_at_ms,
        }))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn acquire_trace_lock(path: &Path) -> Result<TraceWriteLock, TraceStoreError> {
    let lock_path = PathBuf::from(format!("{}.lock", path.display()));
    #[cfg(unix)]
    {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result != 0 {
            let source = std::io::Error::last_os_error();
            if source.kind() == std::io::ErrorKind::WouldBlock {
                return Err(TraceStoreError::LockHeld(lock_path));
            }
            return Err(TraceStoreError::Io(source));
        }
        Ok(TraceWriteLock {
            path: lock_path,
            file,
        })
    }

    #[cfg(not(unix))]
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
            .map_err(|source| {
                if source.kind() == std::io::ErrorKind::AlreadyExists {
                    TraceStoreError::LockHeld(lock_path.clone())
                } else {
                    TraceStoreError::Io(source)
                }
            })?;
        if let Err(source) = writeln!(file, "pid={}", std::process::id()) {
            let _ = fs::remove_file(&lock_path);
            return Err(TraceStoreError::Io(source));
        }
        Ok(TraceWriteLock { path: lock_path })
    }
}

/// Removes a partial final JSONL record before a writer appends to the file.
///
/// A process can be interrupted after writing record bytes but before writing
/// the terminating newline. Readers intentionally ignore that tail, so a
/// reopened writer must remove it as well; otherwise the next append would be
/// concatenated with the partial JSON and make the whole tail unreadable.
fn truncate_incomplete_tail(path: &Path) -> Result<(), TraceStoreError> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => return Err(TraceStoreError::Io(source)),
    };
    if bytes.is_empty() || bytes.last() == Some(&b'\n') {
        return Ok(());
    }

    let tail_start = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |index| index + 1);
    let tail = bytes[tail_start..].trim_ascii();
    if !tail.is_empty() && serde_json::from_slice::<TraceRecord>(tail).is_ok() {
        let mut file = OpenOptions::new().append(true).open(path)?;
        file.write_all(b"\n")?;
        file.sync_data()?;
        return Ok(());
    }

    let len = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |index| index + 1);
    let file = OpenOptions::new().write(true).open(path)?;
    file.set_len(len as u64)?;
    file.sync_data()?;
    Ok(())
}

fn validate_record(
    state: &TraceState,
    run_id: &str,
    record: &TraceRecord,
) -> Result<(), TraceStoreError> {
    match record {
        TraceRecord::Run(run) => {
            validate_run(run)?;
            if state.run.is_some() {
                return Err(TraceStoreError::DuplicateRun(run_id.to_owned()));
            }
        }
        TraceRecord::Build(build) => {
            let run = state
                .run
                .as_ref()
                .ok_or_else(|| TraceStoreError::RunNotStarted(run_id.to_owned()))?;
            if state.finished_at_ms.is_some() || state.terminal {
                return Err(TraceStoreError::TraceAfterTerminal);
            }
            if run.manifest.lifecycle != BenchmarkLifecycle::OfflineBuildThenQuery {
                return Err(TraceStoreError::BuildRequiresOfflineLifecycle);
            }
            if state.build_phase_closed {
                return Err(TraceStoreError::BuildAfterSteps);
            }
            let expected = state.last_build_index.map_or(0, |index| index + 1);
            if build.build_index != expected {
                return Err(TraceStoreError::BuildOutOfOrder {
                    expected,
                    actual: build.build_index,
                });
            }
            validate_build_trace(build, run)?;
        }
        TraceRecord::Step(step) => {
            let run = state
                .run
                .as_ref()
                .ok_or_else(|| TraceStoreError::RunNotStarted(run_id.to_owned()))?;
            if state.finished_at_ms.is_some() || state.terminal {
                return Err(TraceStoreError::TraceAfterTerminal);
            }
            let expected = state.last_step_index.map_or(0, |index| index + 1);
            if step.step_index != expected {
                return Err(TraceStoreError::StepOutOfOrder {
                    expected,
                    actual: step.step_index,
                });
            }
            let expected_namespace = run.isolated_namespace();
            if step.namespace != expected_namespace {
                return Err(TraceStoreError::NamespaceMismatch {
                    expected: expected_namespace,
                    actual: step.namespace.clone(),
                });
            }
            if step.visibility_cutoff_ms != run.manifest.visibility_cutoff_ms {
                return Err(TraceStoreError::CutoffMismatch);
            }
            if let Some(query) = &step.query {
                if query.namespace != step.namespace {
                    return Err(TraceStoreError::NamespaceMismatch {
                        expected: step.namespace.clone(),
                        actual: query.namespace.clone(),
                    });
                }
                if let Some(dataset_namespace) = &query.dataset_namespace {
                    if dataset_namespace != &run.manifest.namespace {
                        return Err(TraceStoreError::NamespaceMismatch {
                            expected: run.manifest.namespace.clone(),
                            actual: dataset_namespace.clone(),
                        });
                    }
                }
                if query.visibility_cutoff_ms != step.visibility_cutoff_ms {
                    return Err(TraceStoreError::CutoffMismatch);
                }
            }
            if step.terminal && step.terminal_reason.is_none() {
                return Err(TraceStoreError::MissingTerminalReason);
            }
            if let Some(ingest) = &step.ingest {
                validate_ingest_trace(ingest, &step.errors)?;
            } else if step
                .errors
                .iter()
                .any(|error| error.stage == TraceErrorStage::Ingest)
            {
                return Err(TraceStoreError::InvalidStep(
                    "step contains an ingest error without an ingest trace".to_string(),
                ));
            }
        }
        TraceRecord::Metric(metric) => {
            let run = state
                .run
                .as_ref()
                .ok_or_else(|| TraceStoreError::RunNotStarted(run_id.to_owned()))?;
            if metric.run_id != run_id {
                return Err(TraceStoreError::RunMismatch {
                    expected: run_id.to_owned(),
                    actual: metric.run_id.clone(),
                });
            }
            if !metric.value.is_finite() || (metric.sample_count == 0 && metric.value != 0.0) {
                return Err(TraceStoreError::InvalidMetric(format!(
                    "metric {:?} must be finite and zero when sample_count is zero",
                    metric.name
                )));
            }
            if !run
                .manifest
                .metrics
                .iter()
                .any(|spec| spec.name == metric.name)
            {
                return Err(TraceStoreError::UnknownMetric(metric.name.clone()));
            }
        }
        TraceRecord::Finished(_) => {
            state
                .run
                .as_ref()
                .ok_or_else(|| TraceStoreError::RunNotStarted(run_id.to_owned()))?;
            if state.finished_at_ms.is_some() {
                return Err(TraceStoreError::RunAlreadyFinished);
            }
            if !state.terminal {
                return Err(TraceStoreError::RunNotTerminal(run_id.to_owned()));
            }
        }
    }
    Ok(())
}

fn validate_run(run: &BenchmarkRun) -> Result<(), TraceStoreError> {
    if run.run_id.trim().is_empty() {
        return Err(TraceStoreError::InvalidManifest(
            "run_id must be non-empty".to_string(),
        ));
    }
    let manifest = &run.manifest;
    if manifest.schema_version != BENCHMARK_SCHEMA_VERSION {
        return Err(TraceStoreError::InvalidManifest(format!(
            "schema_version {} is unsupported",
            manifest.schema_version
        )));
    }
    if manifest.benchmark_id.trim().is_empty() {
        return Err(TraceStoreError::InvalidManifest(
            "benchmark_id must be non-empty".to_string(),
        ));
    }
    validate_namespace(&manifest.namespace)?;
    if run.isolated_namespace().len() > 128 {
        return Err(TraceStoreError::InvalidManifest(
            "isolated namespace must be at most 128 bytes".to_string(),
        ));
    }
    if manifest.dataset_revision.trim().is_empty() {
        return Err(TraceStoreError::InvalidManifest(
            "dataset_revision must be non-empty".to_string(),
        ));
    }
    if let Some(dataset) = &manifest.dataset {
        dataset
            .validate()
            .map_err(|error| TraceStoreError::InvalidManifest(error.to_string()))?;
        if dataset.revision != manifest.dataset_revision {
            return Err(TraceStoreError::InvalidManifest(
                "dataset manifest revision does not match dataset_revision".to_owned(),
            ));
        }
        if manifest.dataset_digest.as_deref() != Some(dataset.content_digest.as_str()) {
            return Err(TraceStoreError::InvalidManifest(
                "dataset manifest digest does not match dataset_digest".to_owned(),
            ));
        }
    }
    if manifest.method.method_id.trim().is_empty() || manifest.method.version.trim().is_empty() {
        return Err(TraceStoreError::InvalidManifest(
            "method_id and method version must be non-empty".to_string(),
        ));
    }
    manifest.verify_config_digest()?;
    let mut metric_names = std::collections::BTreeSet::new();
    for metric in &manifest.metrics {
        if metric.name.trim().is_empty()
            || metric.version.trim().is_empty()
            || metric.unit.trim().is_empty()
        {
            return Err(TraceStoreError::InvalidManifest(
                "metric name, version, and unit must be non-empty".to_string(),
            ));
        }
        if !metric_names.insert(&metric.name) {
            return Err(TraceStoreError::InvalidManifest(format!(
                "metric {:?} is declared more than once",
                metric.name
            )));
        }
    }
    Ok(())
}

fn validate_namespace(namespace: &str) -> Result<(), TraceStoreError> {
    let valid = !namespace.is_empty()
        && namespace.len() <= 128
        && namespace.split('/').all(|segment| {
            !segment.is_empty()
                && segment != "."
                && segment != ".."
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        });
    if valid {
        Ok(())
    } else {
        Err(TraceStoreError::InvalidManifest(
            "namespace must be 1-128 safe ASCII path segments".to_string(),
        ))
    }
}

fn apply_record(state: &mut TraceState, record: TraceRecord) {
    match record {
        TraceRecord::Run(run) => state.run = Some(run),
        TraceRecord::Build(build) => state.last_build_index = Some(build.build_index),
        TraceRecord::Step(step) => {
            state.build_phase_closed = true;
            state.last_step_index = Some(step.step_index);
            state.terminal = step.terminal;
        }
        TraceRecord::Metric(_) => {}
        TraceRecord::Finished(finished) => state.finished_at_ms = Some(finished.finished_at_ms),
    }
}

/// All records belonging to one run, in append order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunTrace {
    pub run: BenchmarkRun,
    pub steps: Vec<StepTrace>,
    pub metrics: Vec<MetricValue>,
    pub finished_at_ms: Option<i64>,
    pub complete: bool,
}

pub fn read_all(path: impl AsRef<Path>) -> Result<Vec<TraceRecord>, TraceStoreError> {
    let bytes = fs::read(path)?;
    let mut records = Vec::new();
    for chunk in bytes.split_inclusive(|byte| *byte == b'\n') {
        if !chunk.ends_with(b"\n") {
            break;
        }
        let line = chunk[..chunk.len() - 1].trim_ascii();
        if line.is_empty() {
            continue;
        }
        records.push(serde_json::from_slice::<TraceRecord>(line)?);
    }
    Ok(records)
}

pub fn read_run(path: impl AsRef<Path>, run_id: &str) -> Result<RunTrace, TraceStoreError> {
    let mut state = TraceState::default();
    let mut run = None;
    let mut steps = Vec::new();
    let mut metrics = Vec::new();
    for record in read_all(path)? {
        if record.run_id() != run_id {
            continue;
        }
        validate_record(&state, run_id, &record)?;
        match &record {
            TraceRecord::Run(value) => run = Some(value.clone()),
            TraceRecord::Build(_) => {}
            TraceRecord::Step(value) => steps.push(value.clone()),
            TraceRecord::Metric(value) => metrics.push(value.clone()),
            TraceRecord::Finished(_) => {}
        }
        apply_record(&mut state, record);
    }
    let run = run.ok_or_else(|| TraceStoreError::RunNotFound(run_id.to_owned()))?;
    Ok(RunTrace {
        run,
        steps,
        metrics,
        finished_at_ms: state.finished_at_ms,
        complete: state.finished_at_ms.is_some(),
    })
}

/// Reads and validates the offline build records for one run.
///
/// Build records intentionally remain outside [`RunTrace`] until the evaluator
/// contract grows a build-phase view. This accessor keeps the records available
/// without changing the existing `RunTrace` struct shape for downstream users.
pub fn read_builds(
    path: impl AsRef<Path>,
    run_id: &str,
) -> Result<Vec<BuildTrace>, TraceStoreError> {
    let mut state = TraceState::default();
    let mut builds = Vec::new();
    for record in read_all(path)? {
        if record.run_id() != run_id {
            continue;
        }
        validate_record(&state, run_id, &record)?;
        if let TraceRecord::Build(build) = &record {
            builds.push(build.clone());
        }
        apply_record(&mut state, record);
    }
    if state.run.is_none() {
        return Err(TraceStoreError::RunNotFound(run_id.to_owned()));
    }
    Ok(builds)
}

/// Hashes canonical serde JSON lines, preserving record order. Struct fields
/// and `BTreeMap` keys are deterministic, so identical traces get identical
/// digests across restarts and machines.
pub fn trace_digest(records: &[TraceRecord]) -> Result<String, TraceStoreError> {
    let mut hasher = Sha256::new();
    for record in records {
        serde_json::to_writer(&mut hasher, record)?;
        hasher.update(b"\n");
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

pub fn file_digest(path: impl AsRef<Path>) -> Result<String, TraceStoreError> {
    trace_digest(&read_all(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn manifest() -> BenchmarkManifest {
        let mut method_config = BTreeMap::new();
        method_config.insert("top_k".into(), "3".into());
        let mut value = BenchmarkManifest::new(
            "retrieval-v1",
            "bench/news",
            MethodManifest {
                method_id: "projection".into(),
                version: "0.1.0".into(),
                config: method_config,
            },
            "dataset-42",
            Some(1_700_000_000_000),
        );
        value.metrics.push(MetricSpec {
            name: "recall@3".into(),
            version: "1".into(),
            unit: "ratio".into(),
            direction: MetricDirection::HigherIsBetter,
        });
        value
    }

    #[test]
    fn method_manifest_binds_provider_config_as_reproducibility_metadata() {
        let mut method = MethodManifest {
            method_id: "lightmem".to_owned(),
            version: "1".to_owned(),
            config: BTreeMap::new(),
        };
        method.bind_provider_config(&BTreeMap::from([
            ("profile".to_owned(), "lightweight".to_owned()),
            ("embedding_dims".to_owned(), "768".to_owned()),
        ]));

        assert_eq!(
            method.config.get("provider_config_source"),
            Some(&"provider_manifest".to_owned())
        );
        assert_eq!(
            method.config.get("provider.profile"),
            Some(&"lightweight".to_owned())
        );
        assert_eq!(
            method.config.get("provider.embedding_dims"),
            Some(&"768".to_owned())
        );
    }

    #[test]
    fn method_manifest_marks_missing_provider_config_explicitly() {
        let mut method = MethodManifest {
            method_id: "legacy".to_owned(),
            version: "1".to_owned(),
            config: BTreeMap::new(),
        };
        method.bind_provider_config(&BTreeMap::new());

        assert_eq!(
            method.config.get("provider_config_source"),
            Some(&"provider_manifest_missing".to_owned())
        );
        assert!(method
            .config
            .keys()
            .all(|key| key == "provider_config_source"));
    }

    fn run(run_id: &str) -> BenchmarkRun {
        BenchmarkRun::new(run_id, manifest(), 1_700_000_000_001)
    }

    fn offline_run(run_id: &str) -> BenchmarkRun {
        let mut value = run(run_id);
        value.manifest.lifecycle = BenchmarkLifecycle::OfflineBuildThenQuery;
        value
    }

    fn build(run_id: &str, index: u64, sample_id: &str) -> BuildTrace {
        BuildTrace {
            run_id: run_id.into(),
            build_index: index,
            sample_id: sample_id.into(),
            sample_digest: Some(format!("sha256:{}", "b".repeat(64))),
            namespace: offline_run(run_id).isolated_namespace(),
            request_digest: format!("sha256:{}", "a".repeat(64)),
            item_ids: vec![Uuid::from_u128(index as u128 + 1).to_string()],
            relation_ids: Vec::new(),
            deduplicated: false,
            snapshot_id: Some(format!("snapshot-{index}")),
            completed_at_ms: Some(1_700_000_000_010 + index as i64),
            latency_ms: 4,
            error: None,
            telemetry: None,
        }
    }

    fn step(run_id: &str, index: u64) -> StepTrace {
        let run = run(run_id);
        StepTrace {
            run_id: run_id.into(),
            step_index: index,
            sample_id: format!("sample-{index}"),
            sample_digest: None,
            namespace: run.isolated_namespace(),
            visibility_cutoff_ms: run.manifest.visibility_cutoff_ms,
            observation: AgentObservation {
                parts: vec![ObservationPart::Text("visible prompt".into())],
            },
            ingest: None,
            query: Some(QueryTrace {
                namespace: run.isolated_namespace(),
                dataset_namespace: Some(run.manifest.namespace.clone()),
                query_digest: None,
                snapshot_id: None,
                input_digest: Some("sha256:query".into()),
                visibility_cutoff_ms: Some(1_700_000_000_000),
                requested_hits: 3,
                retrieval_truncated: false,
                telemetry: None,
            }),
            observed_hits: vec![ObservedHit {
                record_id: "memory-1".into(),
                source_record_ids: Vec::new(),
                version: 1,
                rank: 1,
                score: Some(0.9),
                evidence_ids: vec!["evidence-1".into()],
            }],
            agent_answer: Some("visible answer".into()),
            action: Some(ActionTrace {
                action_type: "respond".into(),
                action_digest: Some("sha256:action".into()),
                calls: Vec::new(),
            }),
            reward: Some(RewardTrace {
                value: 1.0,
                components: BTreeMap::from([(String::from("correct"), 1.0)]),
            }),
            terminal_reason: Some(TerminalReason::Success),
            telemetry: StepTelemetry {
                ingest_latency_ms: 0,
                retrieval_latency_ms: 12,
                agent_latency_ms: 34,
                input_tokens: Some(32),
                output_tokens: Some(16),
                evidence_bytes: 128,
            },
            errors: Vec::new(),
            runtime_link: None,
            terminal: true,
        }
    }

    #[test]
    fn same_trace_has_same_digest() {
        let left = vec![
            TraceRecord::Run(run("run-a")),
            TraceRecord::Step(step("run-a", 0)),
        ];
        let right = vec![
            TraceRecord::Run(run("run-a")),
            TraceRecord::Step(step("run-a", 0)),
        ];
        assert_eq!(trace_digest(&left).unwrap(), trace_digest(&right).unwrap());
    }

    #[test]
    fn legacy_action_trace_without_calls_defaults_to_empty() {
        let action: ActionTrace =
            serde_json::from_str(r#"{"action_type":"respond","action_digest":"sha256:action"}"#)
                .unwrap();

        assert!(action.calls.is_empty());
    }

    #[test]
    fn legacy_query_trace_without_truncation_defaults_to_complete() {
        let query: QueryTrace = serde_json::from_str(
            r#"{
                "namespace":"bench/news/run",
                "dataset_namespace":"bench/news",
                "query_digest":null,
                "input_digest":"sha256:query",
                "visibility_cutoff_ms":1700000000000,
                "requested_hits":3
            }"#,
        )
        .unwrap();

        assert!(!query.retrieval_truncated);
    }

    #[test]
    fn provider_telemetry_roundtrips_in_query_trace() {
        let query = QueryTrace {
            namespace: "bench/news/run".into(),
            dataset_namespace: Some("bench/news".into()),
            query_digest: Some("sha256:query".into()),
            snapshot_id: Some("snapshot-1".into()),
            input_digest: None,
            visibility_cutoff_ms: None,
            requested_hits: 3,
            retrieval_truncated: false,
            telemetry: Some(ProviderTelemetry {
                counters: BTreeMap::from([("lightmem.embedding.total_calls".into(), 1)]),
                durations_ms: BTreeMap::new(),
                cost_usd_micros: Some(7),
            }),
        };

        let decoded: QueryTrace =
            serde_json::from_value(serde_json::to_value(&query).unwrap()).unwrap();

        assert_eq!(decoded, query);
    }

    #[test]
    fn legacy_step_telemetry_without_ingest_latency_defaults_to_zero() {
        let telemetry: StepTelemetry = serde_json::from_str(
            r#"{
                "retrieval_latency_ms":12,
                "agent_latency_ms":34,
                "input_tokens":null,
                "output_tokens":null,
                "evidence_bytes":0
            }"#,
        )
        .unwrap();

        assert_eq!(telemetry.ingest_latency_ms, 0);
    }

    #[test]
    fn lifecycle_defaults_for_old_manifests_and_serializes_explicitly() {
        let value = manifest();
        assert_eq!(value.lifecycle, BenchmarkLifecycle::RetrievalOnly);
        let encoded = serde_json::to_value(&value).unwrap();
        assert_eq!(encoded["lifecycle"], "retrieval_only");

        let mut legacy = encoded;
        legacy.as_object_mut().unwrap().remove("lifecycle");
        let decoded: BenchmarkManifest = serde_json::from_value(legacy).unwrap();
        assert_eq!(decoded.lifecycle, BenchmarkLifecycle::RetrievalOnly);

        let mut offline = value;
        offline.lifecycle = BenchmarkLifecycle::OfflineBuildThenQuery;
        assert_eq!(
            serde_json::to_value(offline).unwrap()["lifecycle"],
            "offline_build_then_query"
        );
    }

    #[test]
    fn build_records_are_ordered_persisted_and_replayed() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("offline.jsonl");
        let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
        writer.append_run(&offline_run("run-a")).unwrap();
        writer
            .append_build(&build("run-a", 0, "session-0"))
            .unwrap();
        writer
            .append_build(&build("run-a", 1, "session-1"))
            .unwrap();
        writer.append_step(&step("run-a", 0)).unwrap();
        writer.finalize(1_700_000_000_100).unwrap();
        drop(writer);

        let builds = read_builds(&path, "run-a").unwrap();
        assert_eq!(builds.len(), 2);
        assert_eq!(builds[0].sample_id, "session-0");
        assert_eq!(builds[1].build_index, 1);
        let records = read_all(&path).unwrap();
        assert!(matches!(records[1], TraceRecord::Build(_)));
        assert!(matches!(records[2], TraceRecord::Build(_)));
        assert_eq!(read_run(&path, "run-a").unwrap().steps.len(), 1);
    }

    #[test]
    fn build_records_require_offline_lifecycle_and_precede_steps() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("offline-order.jsonl");
        let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
        writer.append_run(&run("run-a")).unwrap();
        assert!(matches!(
            writer.append_build(&build("run-a", 0, "session-0")),
            Err(TraceStoreError::BuildRequiresOfflineLifecycle)
        ));
        drop(writer);

        let path = directory.path().join("offline-order-2.jsonl");
        let mut writer = JsonlTraceWriter::open(&path, "run-b").unwrap();
        writer.append_run(&offline_run("run-b")).unwrap();
        writer
            .append_build(&build("run-b", 0, "session-0"))
            .unwrap();
        assert!(matches!(
            writer.append_build(&build("run-b", 2, "session-2")),
            Err(TraceStoreError::BuildOutOfOrder {
                expected: 1,
                actual: 2
            })
        ));
        let mut query_step = step("run-b", 0);
        query_step.terminal = false;
        query_step.terminal_reason = None;
        writer.append_step(&query_step).unwrap();
        assert!(matches!(
            writer.append_build(&build("run-b", 1, "session-1")),
            Err(TraceStoreError::BuildAfterSteps)
        ));
    }

    #[test]
    fn build_validation_checks_digest_receipt_and_error_stage() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("offline-validation.jsonl");
        let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
        writer.append_run(&offline_run("run-a")).unwrap();

        let mut invalid = build("run-a", 0, "session-0");
        invalid.request_digest = "sha256:invalid".into();
        assert!(matches!(
            writer.append_build(&invalid),
            Err(TraceStoreError::InvalidStep(message))
                if message.contains("request_digest")
        ));

        invalid = build("run-a", 0, "session-0");
        invalid.item_ids.push(invalid.item_ids[0].clone());
        assert!(matches!(
            writer.append_build(&invalid),
            Err(TraceStoreError::InvalidStep(message))
                if message.contains("receipt ids")
        ));

        invalid = build("run-a", 0, "session-0");
        invalid.item_ids.clear();
        invalid.snapshot_id = None;
        invalid.error = Some(TraceErrorEntry {
            stage: TraceErrorStage::Ingest,
            code: "backend".into(),
            message: "failed".into(),
            retryable: true,
        });
        assert!(matches!(
            writer.append_build(&invalid),
            Err(TraceStoreError::InvalidStep(message))
                if message.contains("build stage")
        ));

        invalid.error.as_mut().unwrap().stage = TraceErrorStage::Build;
        writer.append_build(&invalid).unwrap();
        assert_eq!(read_builds(&path, "run-a").unwrap().len(), 1);
    }

    #[test]
    fn legacy_runtime_link_defaults_to_an_empty_decision_list() {
        let link: RuntimeStepLink = serde_json::from_str(&format!(
            r#"{{"runtime_run_id":"{}","turn_id":"{}","memory_decision":null}}"#,
            Uuid::from_u128(1),
            Uuid::from_u128(2),
        ))
        .unwrap();

        assert!(link.memory_decisions.is_empty());
    }

    #[test]
    fn restart_reads_append_only_trace() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        {
            let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
            writer.append_run(&run("run-a")).unwrap();
        }
        {
            let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
            writer.append_step(&step("run-a", 0)).unwrap();
            writer.finalize(1_700_000_000_100).unwrap();
        }
        let replay = read_run(&path, "run-a").unwrap();
        assert_eq!(replay.run.run_id, "run-a");
        assert_eq!(replay.steps.len(), 1);
        assert_eq!(
            replay.steps[0].visibility_cutoff_ms,
            Some(1_700_000_000_000)
        );
        assert!(replay.complete);
        assert_eq!(replay.finished_at_ms, Some(1_700_000_000_100));
    }

    #[test]
    fn runs_are_isolated_and_cutoff_is_preserved() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("shared.jsonl");
        let mut writer_a = JsonlTraceWriter::open(&path, "run-a").unwrap();
        writer_a.append_run(&run("run-a")).unwrap();
        writer_a.append_step(&step("run-a", 0)).unwrap();
        drop(writer_a);
        let mut writer_b = JsonlTraceWriter::open(&path, "run-b").unwrap();
        let mut run_b = run("run-b");
        run_b.manifest.visibility_cutoff_ms = Some(1_800_000_000_000);
        writer_b.append_run(&run_b).unwrap();
        drop(writer_b);

        let replay = read_run(&path, "run-a").unwrap();
        assert_eq!(replay.steps.len(), 1);
        assert_eq!(
            replay.run.manifest.visibility_cutoff_ms,
            Some(1_700_000_000_000)
        );
        assert!(matches!(
            read_run(&path, "missing"),
            Err(TraceStoreError::RunNotFound(_))
        ));
    }

    #[test]
    fn isolated_namespace_escapes_path_delimiters_in_run_id() {
        let value = run("run/a");
        assert_eq!(value.isolated_namespace(), "bench/news/r72756e2f61");
    }

    #[test]
    fn writer_rejects_mixed_runs() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
        let error = writer.append_run(&run("run-b")).unwrap_err();
        assert!(matches!(error, TraceStoreError::RunMismatch { .. }));
    }

    #[test]
    fn writer_rejects_invalid_manifest() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("invalid.jsonl");
        let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
        let mut invalid = run("run-a");
        invalid.manifest.namespace = "../escape".to_string();
        let error = writer.append_run(&invalid).unwrap_err();
        assert!(matches!(error, TraceStoreError::InvalidManifest(_)));
    }

    #[test]
    fn writer_validates_not_applicable_metric_values() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
        writer.append_run(&run("run-a")).unwrap();

        let invalid = MetricValue {
            run_id: "run-a".into(),
            name: "recall@3".into(),
            value: 1.0,
            sample_count: 0,
        };
        assert!(matches!(
            writer.append_metric(&invalid),
            Err(TraceStoreError::InvalidMetric(_))
        ));

        let valid = MetricValue {
            value: 0.0,
            ..invalid
        };
        writer.append_metric(&valid).unwrap();
    }

    #[test]
    fn writer_rejects_a_query_bound_to_another_dataset_namespace() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
        writer.append_run(&run("run-a")).unwrap();
        let mut invalid = step("run-a", 0);
        invalid.query.as_mut().unwrap().dataset_namespace = Some("other".into());

        let error = writer.append_step(&invalid).unwrap_err();
        assert!(matches!(error, TraceStoreError::NamespaceMismatch { .. }));
    }

    #[test]
    fn trace_writer_lock_rejects_a_second_writer() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
        let error = JsonlTraceWriter::open(&path, "run-b").unwrap_err();
        assert!(matches!(error, TraceStoreError::LockHeld(_)));
        drop(writer);
        assert!(JsonlTraceWriter::open(&path, "run-b").is_ok());
    }

    #[test]
    fn writer_rejects_invalid_step_lifecycle() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("lifecycle.jsonl");
        let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
        writer.append_run(&run("run-a")).unwrap();
        let mut out_of_order = step("run-a", 1);
        out_of_order.terminal = false;
        assert!(matches!(
            writer.append_step(&out_of_order),
            Err(TraceStoreError::StepOutOfOrder { .. })
        ));
        writer.append_step(&step("run-a", 0)).unwrap();
        writer.finalize(1_700_000_000_100).unwrap();
        assert!(matches!(
            writer.append_step(&step("run-a", 1)),
            Err(TraceStoreError::TraceAfterTerminal)
        ));
    }

    #[test]
    fn observation_serialization_has_no_reference_answer() {
        let encoded = serde_json::to_string(&step("run-a", 0)).unwrap();
        assert!(!encoded.contains("reference_answer"));
        assert!(encoded.contains("agent_answer"));
        assert!(encoded.contains("retrieval_latency_ms"));
    }

    #[test]
    fn incomplete_tail_is_not_replayed() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("tail.jsonl");
        let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
        writer.append_run(&run("run-a")).unwrap();
        drop(writer);
        let mut bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.pop(), Some(b'\n'));
        fs::write(&path, bytes).unwrap();
        assert!(read_all(&path).unwrap().is_empty());
    }

    #[test]
    fn reopening_truncates_incomplete_tail_before_appending() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("tail-recovery.jsonl");
        {
            let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
            writer.append_run(&run("run-a")).unwrap();
        }

        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(br#"{"type":"Step"#);
        fs::write(&path, bytes).unwrap();

        {
            let mut writer = JsonlTraceWriter::open(&path, "run-a").unwrap();
            writer.append_step(&step("run-a", 0)).unwrap();
        }

        let replay = read_run(&path, "run-a").unwrap();
        assert_eq!(replay.steps.len(), 1);
        assert_eq!(
            fs::read(&path)
                .unwrap()
                .iter()
                .filter(|byte| **byte == b'\n')
                .count(),
            2
        );
    }

    #[test]
    fn config_digest_is_stable_and_seal_detects_drift() {
        let mut left = manifest();
        left.config.insert("model".into(), "qwen3.8".into());
        left.config.insert("seed".into(), "7".into());

        let mut right = manifest();
        right.config.insert("seed".into(), "7".into());
        right.config.insert("model".into(), "qwen3.8".into());
        assert_eq!(left.config_digest(), right.config_digest());

        left.seal_config();
        assert!(left.verify_config_digest().is_ok());
        left.config.insert("temperature".into(), "0.2".into());
        assert!(matches!(
            left.verify_config_digest(),
            Err(TraceStoreError::InvalidManifest(message))
                if message.contains("config_digest")
        ));
    }

    #[test]
    fn capability_snapshot_is_validated_when_present() {
        let mut value = manifest();
        let capabilities = MemoryCapabilities {
            modalities: vec![pi_agent_memory::Modality::Text],
            memory_kinds: vec![pi_agent_memory::MemoryKind::Episodic],
            supports_content_search: true,
            supports_time_filter: false,
            supports_visibility_cutoff: false,
            supports_graph: false,
            supports_cross_modal: false,
            supports_update: false,
            supports_delete: false,
            supports_batch: true,
            supports_record_ingest: false,
            supports_record_get: false,
            deterministic: true,
            persistent: false,
        };
        value.method.config.insert(
            "capabilities_json".into(),
            serde_json::to_string(&capabilities).unwrap(),
        );

        assert!(value.verify_config_digest().is_ok());
    }

    #[test]
    fn malformed_capability_snapshot_is_rejected() {
        let mut value = manifest();
        value
            .method
            .config
            .insert("capabilities_json".into(), "not-json".into());

        assert!(matches!(
            value.verify_config_digest(),
            Err(TraceStoreError::InvalidManifest(message))
                if message.contains("capabilities_json")
        ));
    }

    #[test]
    fn reproducibility_config_requires_model_prompt_and_seed() {
        let mut value = manifest();
        assert!(value.validate_reproducibility_config().is_err());
        value.config.insert("model".into(), "qwen3.8".into());
        value
            .config
            .insert("prompt_digest".into(), "sha256:prompt".into());
        value.config.insert("seed".into(), "7".into());
        assert!(value.validate_reproducibility_config().is_ok());
    }

    #[test]
    fn old_manifest_without_config_digest_remains_compatible() {
        let mut encoded = serde_json::to_value(manifest()).unwrap();
        encoded.as_object_mut().unwrap().remove("config_digest");
        let decoded: BenchmarkManifest = serde_json::from_value(encoded).unwrap();
        assert!(decoded.config_digest.is_none());
        assert!(decoded.verify_config_digest().is_ok());
    }

    #[test]
    fn ingest_trace_validation_binds_receipt_shape_and_error_stage() {
        let valid = IngestTrace {
            request_digest: format!("sha256:{}", "a".repeat(64)),
            item_ids: vec![Uuid::from_u128(1).to_string()],
            relation_ids: Vec::new(),
            deduplicated: false,
            snapshot_id: Some("snapshot-1".into()),
            completed_at_ms: Some(1_700_000_000_000),
            error: None,
            telemetry: None,
        };
        assert!(validate_ingest_trace(&valid, &[]).is_ok());

        let mut invalid = valid.clone();
        invalid.item_ids = vec!["record-1".into()];
        assert!(matches!(
            validate_ingest_trace(&invalid, &[]),
            Err(TraceStoreError::InvalidStep(message))
                if message.contains("receipt ids")
        ));

        let error = TraceErrorEntry {
            stage: TraceErrorStage::Ingest,
            code: "timeout".into(),
            message: "write timed out".into(),
            retryable: true,
        };
        invalid = valid;
        invalid.item_ids.clear();
        invalid.snapshot_id = None;
        invalid.error = Some(error.clone());
        assert!(validate_ingest_trace(&invalid, &[error]).is_ok());
        let mut wrong_stage = invalid;
        wrong_stage.error.as_mut().unwrap().stage = TraceErrorStage::Retrieval;
        assert!(matches!(
            validate_ingest_trace(&wrong_stage, &[]),
            Err(TraceStoreError::InvalidStep(message))
                if message.contains("ingest stage")
        ));
    }
}
