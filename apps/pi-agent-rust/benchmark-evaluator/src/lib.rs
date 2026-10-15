//! Offline evaluation for benchmark traces.
//!
//! Reference data is deliberately supplied separately from the agent-visible
//! trace. This prevents a reference answer or relevance label from leaking
//! into a memory query or an observation while still making evaluation
//! reproducible.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

use pi_agent_benchmark::{
    read_builds, read_run, validate_build_trace, validate_ingest_trace, ArtifactValidationPolicy,
    BenchmarkLifecycle, BenchmarkManifest, BuildTrace, JsonlTraceWriter, MetricDirection,
    MetricSpec, MetricValue, RunTrace, RuntimeMemoryDecisionLink, RuntimeStepLink, TraceErrorStage,
    TraceStoreError,
};
use pi_agent_runtime::{
    EventSequence, JsonlLedger, LedgerEvent, RunId, MEMORY_INJECTED_EVENT, MEMORY_SKIPPED_EVENT,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

mod aggregate;
mod longmemeval;

pub use aggregate::{
    aggregate_traces, compare_traces, AggregateError, AggregateMetric, AggregateReport,
    ComparisonReport, AGGREGATE_SCHEMA_VERSION, COMPARISON_SCHEMA_VERSION,
};
pub use longmemeval::{
    longmemeval_is_abstention_question, longmemeval_judge_metric_specs, longmemeval_judge_prompt,
    parse_longmemeval_judge_response, score_answer, LongMemEvalAnswerError,
    LongMemEvalAnswerEvaluator, LongMemEvalAnswerReference, LongMemEvalAnswerScore,
    LongMemEvalJudgeError, LONGMEMEVAL_JUDGE_METRIC,
};

const METRIC_VERSION: &str = "1";
const REFERENCE_SIDECAR_FORMAT_VERSION: u32 = 1;

fn is_false(value: &bool) -> bool {
    !*value
}

/// Expected memory records for one dataset sample.
///
/// The identifier space is declared by [`ReferenceCatalogManifest::identity`]
/// rather than inferred from the provider's returned record IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleReference {
    pub sample_id: String,
    pub relevant_record_ids: BTreeSet<String>,
}

/// Identity space used by the reference relevance labels.
///
/// `ProviderRecordId` keeps the historical behavior where labels refer to the
/// identifier returned by a memory method. `CanonicalSourceId` is required for
/// methods that summarize, merge, or otherwise rewrite provider records: the
/// method must return the original benchmark record identities in each hit's
/// `source_record_ids` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceIdentity {
    #[default]
    ProviderRecordId,
    CanonicalSourceId,
}

impl ReferenceIdentity {
    fn is_default(&self) -> bool {
        matches!(self, Self::ProviderRecordId)
    }
}

/// The immutable dataset identity to which a reference catalog belongs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceCatalogManifest {
    pub benchmark_id: String,
    pub namespace: String,
    pub dataset_revision: String,
    pub dataset_digest: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "ArtifactValidationPolicy::is_unspecified"
    )]
    pub artifact_validation_policy: ArtifactValidationPolicy,
    pub visibility_cutoff_ms: Option<u64>,
    /// Identity space used by `SampleReference::relevant_record_ids`.
    #[serde(default, skip_serializing_if = "ReferenceIdentity::is_default")]
    pub identity: ReferenceIdentity,
    /// Allows dataset-specific samples whose retrieval relevance is not
    /// defined, such as LongMemEval abstention questions.
    #[serde(default, skip_serializing_if = "is_false")]
    pub allow_empty_relevance: bool,
}

impl ReferenceCatalogManifest {
    pub fn from_benchmark(manifest: &BenchmarkManifest) -> Self {
        Self {
            benchmark_id: manifest.benchmark_id.clone(),
            namespace: manifest.namespace.clone(),
            dataset_revision: manifest.dataset_revision.clone(),
            dataset_digest: manifest.dataset_digest.clone(),
            artifact_validation_policy: manifest.artifact_validation_policy,
            visibility_cutoff_ms: manifest.visibility_cutoff_ms,
            identity: ReferenceIdentity::ProviderRecordId,
            allow_empty_relevance: false,
        }
    }

    /// Checks the immutable benchmark identity while allowing the reference
    /// sidecar to choose its explicit record identity space.
    fn matches_benchmark(&self, manifest: &BenchmarkManifest) -> bool {
        let expected = Self::from_benchmark(manifest);
        self.benchmark_id == expected.benchmark_id
            && self.namespace == expected.namespace
            && self.dataset_revision == expected.dataset_revision
            && self.dataset_digest == expected.dataset_digest
            && self.artifact_validation_policy == expected.artifact_validation_policy
            && self.visibility_cutoff_ms == expected.visibility_cutoff_ms
    }
}

/// Immutable reference catalog loaded by an evaluator sidecar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceCatalog {
    pub manifest: ReferenceCatalogManifest,
    samples: BTreeMap<String, SampleReference>,
}

/// Stable on-disk representation of a reference catalog.
///
/// The sample list is intentionally encoded as a vector instead of a map so
/// duplicate sample IDs remain observable during loading and can be rejected
/// by [`ReferenceCatalog::new`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ReferenceCatalogSidecar {
    format_version: u32,
    manifest: ReferenceCatalogManifest,
    samples: Vec<SampleReferenceSidecar>,
    digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SampleReferenceSidecar {
    sample_id: String,
    relevant_record_ids: Vec<String>,
}

impl ReferenceCatalog {
    pub fn new(
        manifest: ReferenceCatalogManifest,
        references: Vec<SampleReference>,
    ) -> Result<Self, EvaluatorError> {
        if manifest.benchmark_id.trim().is_empty()
            || manifest.namespace.trim().is_empty()
            || manifest.dataset_revision.trim().is_empty()
        {
            return Err(EvaluatorError::InvalidReference(
                "catalog manifest identity fields must be non-empty".to_string(),
            ));
        }
        if references.is_empty() {
            return Err(EvaluatorError::InvalidReference(
                "catalog must contain at least one sample".to_string(),
            ));
        }
        let mut samples = BTreeMap::new();
        for reference in references {
            if reference.sample_id.trim().is_empty() {
                return Err(EvaluatorError::InvalidReference(
                    "sample_id must be non-empty".to_string(),
                ));
            }
            if (reference.relevant_record_ids.is_empty() && !manifest.allow_empty_relevance)
                || reference
                    .relevant_record_ids
                    .iter()
                    .any(|id| id.trim().is_empty())
            {
                return Err(EvaluatorError::InvalidReference(format!(
                    "sample {:?} must contain relevant record ids unless the catalog allows empty relevance",
                    reference.sample_id
                )));
            }
            if samples
                .insert(reference.sample_id.clone(), reference)
                .is_some()
            {
                return Err(EvaluatorError::InvalidReference(
                    "sample_id appears more than once".to_string(),
                ));
            }
        }
        Ok(Self { manifest, samples })
    }

    pub fn for_benchmark(
        manifest: &BenchmarkManifest,
        references: Vec<SampleReference>,
    ) -> Result<Self, EvaluatorError> {
        Self::new(
            ReferenceCatalogManifest::from_benchmark(manifest),
            references,
        )
    }

    /// Returns a stable digest for persisting the exact reference sidecar used.
    pub fn digest(&self) -> Result<String, EvaluatorError> {
        let bytes = serde_json::to_vec(self).map_err(|error| {
            EvaluatorError::InvalidReference(format!("catalog serialization failed: {error}"))
        })?;
        Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
    }

    /// Writes this catalog as an atomically replaced JSON sidecar.
    ///
    /// The temporary file is created next to the destination, flushed to the
    /// filesystem, and then renamed. A reader therefore observes either the
    /// previous complete sidecar or the new complete sidecar.
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), EvaluatorError> {
        let path = path.as_ref();
        let digest = self.digest()?;
        let sidecar = ReferenceCatalogSidecar {
            format_version: REFERENCE_SIDECAR_FORMAT_VERSION,
            manifest: self.manifest.clone(),
            samples: self
                .samples
                .values()
                .map(|reference| SampleReferenceSidecar {
                    sample_id: reference.sample_id.clone(),
                    relevant_record_ids: reference.relevant_record_ids.iter().cloned().collect(),
                })
                .collect(),
            digest,
        };
        let bytes = serde_json::to_vec_pretty(&sidecar).map_err(|error| {
            EvaluatorError::InvalidReferenceSidecar(format!(
                "catalog serialization failed: {error}"
            ))
        })?;
        atomic_write(path, &bytes)
    }

    /// Loads and validates a JSON sidecar, including its embedded digest.
    pub fn read_json(path: impl AsRef<Path>) -> Result<Self, EvaluatorError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| sidecar_io(path, "read", source))?;
        let sidecar: ReferenceCatalogSidecar = serde_json::from_slice(&bytes).map_err(|error| {
            EvaluatorError::InvalidReferenceSidecar(format!("JSON decode failed: {error}"))
        })?;
        if sidecar.format_version != REFERENCE_SIDECAR_FORMAT_VERSION {
            return Err(EvaluatorError::InvalidReferenceSidecar(format!(
                "unsupported format_version {}",
                sidecar.format_version
            )));
        }
        let references = sidecar
            .samples
            .into_iter()
            .map(|reference| {
                let mut relevant_record_ids = BTreeSet::new();
                for record_id in reference.relevant_record_ids {
                    if !relevant_record_ids.insert(record_id.clone()) {
                        return Err(EvaluatorError::InvalidReferenceSidecar(format!(
                            "sample {:?} contains duplicate relevant record id {:?}",
                            reference.sample_id, record_id
                        )));
                    }
                }
                Ok(SampleReference {
                    sample_id: reference.sample_id,
                    relevant_record_ids,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let catalog = Self::new(sidecar.manifest, references)?;
        let actual_digest = catalog.digest()?;
        if sidecar.digest != actual_digest {
            return Err(EvaluatorError::ReferenceSidecarDigestMismatch {
                expected: sidecar.digest,
                actual: actual_digest,
            });
        }
        Ok(catalog)
    }

    /// Loads a sidecar and additionally checks the digest recorded by a run.
    pub fn read_json_with_digest(
        path: impl AsRef<Path>,
        expected_digest: &str,
    ) -> Result<Self, EvaluatorError> {
        if expected_digest.trim().is_empty() {
            return Err(EvaluatorError::InvalidReferenceSidecar(
                "expected digest must be non-empty".to_string(),
            ));
        }
        let catalog = Self::read_json(path)?;
        let actual_digest = catalog.digest()?;
        if expected_digest != actual_digest {
            return Err(EvaluatorError::ReferenceSidecarDigestMismatch {
                expected: expected_digest.to_string(),
                actual: actual_digest,
            });
        }
        Ok(catalog)
    }

    /// Loads a sidecar and checks that it belongs to the expected dataset.
    pub fn read_json_for_manifest(
        path: impl AsRef<Path>,
        expected_manifest: &ReferenceCatalogManifest,
    ) -> Result<Self, EvaluatorError> {
        let catalog = Self::read_json(path)?;
        if &catalog.manifest != expected_manifest {
            return Err(EvaluatorError::ReferenceSidecarIdentityMismatch {
                expected: format!("{expected_manifest:?}"),
                actual: format!("{:?}", catalog.manifest),
            });
        }
        Ok(catalog)
    }

    fn get(&self, sample_id: &str) -> Option<&SampleReference> {
        self.samples.get(sample_id)
    }

    fn missing_sample(&self, seen: &BTreeSet<String>) -> Option<String> {
        self.samples
            .keys()
            .find(|sample_id| !seen.contains(*sample_id))
            .cloned()
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), EvaluatorError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().ok_or_else(|| {
        EvaluatorError::InvalidReferenceSidecar("sidecar path must name a file".to_string())
    })?;
    let temporary_path = parent.join(format!(
        ".{}.tmp-{}",
        file_name.to_string_lossy(),
        Uuid::new_v4()
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
            .map_err(|source| sidecar_io(path, "create temporary file", source))?;
        file.write_all(bytes)
            .map_err(|source| sidecar_io(path, "write", source))?;
        file.sync_all()
            .map_err(|source| sidecar_io(path, "sync", source))?;
        fs::rename(&temporary_path, path).map_err(|source| sidecar_io(path, "rename", source))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

fn sidecar_io(path: &Path, operation: &'static str, source: std::io::Error) -> EvaluatorError {
    EvaluatorError::ReferenceSidecarIo {
        operation,
        path: path.display().to_string(),
        source,
    }
}

/// Retrieval metrics calculated for one completed trace.
pub struct RetrievalEvaluator {
    top_k: u32,
    references: ReferenceCatalog,
}

/// Evaluates a continual-memory trace.
///
/// A continual run has one write receipt for every dataset sample.  Retrieval
/// metrics are still evaluated with the same rules as [`RetrievalEvaluator`],
/// while the additional ingest metrics make write failures and deduplication
/// visible in method comparisons.
pub struct ContinualEvaluator {
    retrieval: RetrievalEvaluator,
}

/// Evaluates a run whose memory projection is built completely before the
/// first query.  The build receipts are supplied separately because
/// [`RunTrace`] intentionally keeps query steps and build records as two
/// different parts of the append-only trace.
pub struct OfflineBuildThenQueryEvaluator {
    retrieval: RetrievalEvaluator,
}

/// Evaluates a trace produced by an agent adapter that is linked to the
/// runtime ledger.
///
/// The retrieval metrics are deliberately delegated to [`RetrievalEvaluator`]
/// so adding the runtime contract does not change the pure retrieval
/// evaluator or its metric declarations.  The wrapper only adds the
/// correlation checks needed to prove that a benchmark step used the same
/// runtime operation and turn that produced its answer.
pub struct RuntimeLinkedEvaluator {
    retrieval: RetrievalEvaluator,
}

impl RuntimeLinkedEvaluator {
    pub fn new(top_k: u32, references: ReferenceCatalog) -> Result<Self, EvaluatorError> {
        Ok(Self {
            retrieval: RetrievalEvaluator::new(top_k, references)?,
        })
    }

    pub fn metric_specs(&self) -> Vec<MetricSpec> {
        self.retrieval.metric_specs()
    }

    /// Evaluates retrieval metrics and then verifies the runtime correlation
    /// for every step in the completed trace.
    pub fn evaluate(&self, trace: &RunTrace) -> Result<EvaluationReport, EvaluatorError> {
        let report = self.retrieval.evaluate(trace)?;
        self.validate_runtime_links(trace)?;
        Ok(report)
    }

    /// Evaluates a trace and verifies every linked memory decision against the
    /// durable runtime ledger that produced it.
    pub fn evaluate_with_ledger(
        &self,
        trace: &RunTrace,
        ledger_path: impl AsRef<std::path::Path>,
    ) -> Result<EvaluationReport, EvaluatorError> {
        let report = self.evaluate(trace)?;
        self.validate_runtime_ledger(trace, ledger_path)?;
        Ok(report)
    }

    /// Validates only the optional runtime ledger correlation contract.
    pub fn validate_runtime_links(&self, trace: &RunTrace) -> Result<(), EvaluatorError> {
        let operation_run_id = trace
            .run
            .operation_run_id
            .ok_or(EvaluatorError::MissingRuntimeOperationRunId)?;
        if operation_run_id.is_nil() {
            return Err(EvaluatorError::InvalidRuntimeOperationRunId);
        }

        let mut turn_ids = BTreeSet::new();
        for step in &trace.steps {
            let link = step
                .runtime_link
                .as_ref()
                .ok_or_else(|| EvaluatorError::MissingRuntimeLink(step.sample_id.clone()))?;
            if link.runtime_run_id != operation_run_id {
                return Err(EvaluatorError::RuntimeRunIdMismatch {
                    sample_id: step.sample_id.clone(),
                    expected: operation_run_id.to_string(),
                    actual: link.runtime_run_id.to_string(),
                });
            }
            if link.turn_id.is_nil() {
                return Err(EvaluatorError::InvalidRuntimeTurnId(step.sample_id.clone()));
            }
            if !turn_ids.insert(link.turn_id) {
                return Err(EvaluatorError::DuplicateRuntimeTurnId(
                    step.sample_id.clone(),
                ));
            }
            for decision in runtime_memory_decisions(link) {
                if decision.proposal_id.is_nil() {
                    return Err(EvaluatorError::InvalidRuntimeMemoryDecision {
                        sample_id: step.sample_id.clone(),
                        field: "proposal_id".to_string(),
                    });
                }
                if decision.event_sequence == 0 {
                    return Err(EvaluatorError::InvalidRuntimeMemoryDecision {
                        sample_id: step.sample_id.clone(),
                        field: "event_sequence".to_string(),
                    });
                }
                if !matches!(decision.decision_kind.as_str(), "injected" | "skipped") {
                    return Err(EvaluatorError::InvalidRuntimeMemoryDecision {
                        sample_id: step.sample_id.clone(),
                        field: "decision_kind".to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_runtime_ledger(
        &self,
        trace: &RunTrace,
        ledger_path: impl AsRef<std::path::Path>,
    ) -> Result<(), EvaluatorError> {
        let operation_run_id = trace
            .run
            .operation_run_id
            .ok_or(EvaluatorError::MissingRuntimeOperationRunId)?;
        let mut ledger = JsonlLedger::open(ledger_path).map_err(EvaluatorError::RuntimeLedger)?;
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .map_err(EvaluatorError::RuntimeLedger)?;
        ledger.close().map_err(EvaluatorError::RuntimeLedger)?;

        let run_id = RunId::from_uuid(operation_run_id);
        for step in &trace.steps {
            let Some(link) = &step.runtime_link else {
                return Err(EvaluatorError::MissingRuntimeLink(step.sample_id.clone()));
            };
            for decision in runtime_memory_decisions(link) {
                let sequence = EventSequence::new(decision.event_sequence);
                let event = events
                    .iter()
                    .find(|event| event.sequence == sequence)
                    .ok_or(EvaluatorError::MissingRuntimeLedgerEvent {
                        sample_id: step.sample_id.clone(),
                        event_sequence: decision.event_sequence,
                    })?;
                validate_memory_ledger_event(
                    event,
                    run_id,
                    link.turn_id,
                    decision,
                    &step.sample_id,
                )?;
            }
        }
        Ok(())
    }
}

/// Returns the complete decision list for a runtime link while keeping traces
/// written before the list field was introduced readable.
fn runtime_memory_decisions(link: &RuntimeStepLink) -> Vec<&RuntimeMemoryDecisionLink> {
    if link.memory_decisions.is_empty() {
        link.memory_decision.iter().collect()
    } else {
        link.memory_decisions.iter().collect()
    }
}

fn validate_memory_ledger_event(
    event: &LedgerEvent,
    expected_run_id: RunId,
    expected_turn_id: Uuid,
    decision: &RuntimeMemoryDecisionLink,
    sample_id: &str,
) -> Result<(), EvaluatorError> {
    let expected_kind = match decision.decision_kind.as_str() {
        "injected" => MEMORY_INJECTED_EVENT,
        "skipped" => MEMORY_SKIPPED_EVENT,
        _ => {
            return Err(EvaluatorError::InvalidRuntimeMemoryDecision {
                sample_id: sample_id.to_string(),
                field: "decision_kind".to_string(),
            })
        }
    };
    if event.draft.kind.as_str() != expected_kind {
        return Err(EvaluatorError::RuntimeLedgerFieldMismatch {
            sample_id: sample_id.to_string(),
            event_sequence: decision.event_sequence,
            field: "event_kind".to_string(),
            expected: expected_kind.to_string(),
            actual: event.draft.kind.to_string(),
        });
    }
    if event.draft.run_id != expected_run_id {
        return Err(EvaluatorError::RuntimeLedgerFieldMismatch {
            sample_id: sample_id.to_string(),
            event_sequence: decision.event_sequence,
            field: "draft.run_id".to_string(),
            expected: expected_run_id.as_uuid().to_string(),
            actual: event.draft.run_id.as_uuid().to_string(),
        });
    }

    let payload = event.draft.payload.as_object().ok_or_else(|| {
        EvaluatorError::InvalidRuntimeLedgerPayload {
            event_sequence: decision.event_sequence,
            field: "payload".to_string(),
        }
    })?;
    let payload_run_id = parse_uuid_field(payload, "run_id", decision.event_sequence)?;
    if payload_run_id != expected_run_id.as_uuid() {
        return Err(EvaluatorError::RuntimeLedgerFieldMismatch {
            sample_id: sample_id.to_string(),
            event_sequence: decision.event_sequence,
            field: "payload.run_id".to_string(),
            expected: expected_run_id.as_uuid().to_string(),
            actual: payload_run_id.to_string(),
        });
    }
    let turn_id = parse_uuid_field(payload, "turn_id", decision.event_sequence)?;
    if turn_id != expected_turn_id {
        return Err(EvaluatorError::RuntimeLedgerFieldMismatch {
            sample_id: sample_id.to_string(),
            event_sequence: decision.event_sequence,
            field: "payload.turn_id".to_string(),
            expected: expected_turn_id.to_string(),
            actual: turn_id.to_string(),
        });
    }
    let status = payload
        .get("status")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| EvaluatorError::InvalidRuntimeLedgerPayload {
            event_sequence: decision.event_sequence,
            field: "status".to_string(),
        })?;
    let expected_status = if decision.decision_kind == "injected" {
        "injected"
    } else {
        "skipped"
    };
    if status != expected_status {
        return Err(EvaluatorError::RuntimeLedgerFieldMismatch {
            sample_id: sample_id.to_string(),
            event_sequence: decision.event_sequence,
            field: "status".to_string(),
            expected: expected_status.to_string(),
            actual: status.to_string(),
        });
    }
    let details = payload
        .get("details")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| EvaluatorError::InvalidRuntimeLedgerPayload {
            event_sequence: decision.event_sequence,
            field: "details".to_string(),
        })?;
    let proposal_id = parse_uuid_field(details, "proposal_id", decision.event_sequence)?;
    if proposal_id != decision.proposal_id {
        return Err(EvaluatorError::RuntimeLedgerFieldMismatch {
            sample_id: sample_id.to_string(),
            event_sequence: decision.event_sequence,
            field: "details.proposal_id".to_string(),
            expected: decision.proposal_id.to_string(),
            actual: proposal_id.to_string(),
        });
    }
    Ok(())
}

fn parse_uuid_field(
    payload: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    event_sequence: u64,
) -> Result<Uuid, EvaluatorError> {
    let value = payload
        .get(field)
        .ok_or_else(|| EvaluatorError::InvalidRuntimeLedgerPayload {
            event_sequence,
            field: field.to_string(),
        })?;
    serde_json::from_value(value.clone()).map_err(|_| EvaluatorError::InvalidRuntimeLedgerPayload {
        event_sequence,
        field: field.to_string(),
    })
}

impl RetrievalEvaluator {
    pub fn new(top_k: u32, references: ReferenceCatalog) -> Result<Self, EvaluatorError> {
        if top_k == 0 {
            return Err(EvaluatorError::InvalidConfig(
                "top_k must be greater than zero".to_string(),
            ));
        }
        Ok(Self { top_k, references })
    }

    /// Metric declarations to place in `BenchmarkManifest.metrics` before a run.
    pub fn metric_specs(&self) -> Vec<MetricSpec> {
        vec![
            MetricSpec {
                name: self.hit_rate_name(),
                version: METRIC_VERSION.to_string(),
                unit: "ratio".to_string(),
                direction: MetricDirection::HigherIsBetter,
            },
            MetricSpec {
                name: self.recall_name(),
                version: METRIC_VERSION.to_string(),
                unit: "ratio".to_string(),
                direction: MetricDirection::HigherIsBetter,
            },
            MetricSpec {
                name: "retrieval.error_rate".to_string(),
                version: METRIC_VERSION.to_string(),
                unit: "ratio".to_string(),
                direction: MetricDirection::LowerIsBetter,
            },
            MetricSpec {
                name: "retrieval.mean_latency_ms".to_string(),
                version: METRIC_VERSION.to_string(),
                unit: "milliseconds".to_string(),
                direction: MetricDirection::LowerIsBetter,
            },
        ]
    }

    pub fn evaluate(&self, trace: &RunTrace) -> Result<EvaluationReport, EvaluatorError> {
        if !trace.complete {
            return Err(EvaluatorError::IncompleteTrace);
        }
        if trace.steps.is_empty() {
            return Err(EvaluatorError::InvalidTrace(
                "completed trace must contain at least one step".to_string(),
            ));
        }
        let terminal_step = trace.steps.last();
        if trace.finished_at_ms.is_none()
            || terminal_step.is_none_or(|step| !step.terminal || step.terminal_reason.is_none())
            || trace.steps[..trace.steps.len().saturating_sub(1)]
                .iter()
                .any(|step| step.terminal || step.terminal_reason.is_some())
        {
            return Err(EvaluatorError::InvalidTrace(
                "completed trace must end with exactly one terminal step".to_string(),
            ));
        }
        if !self
            .references
            .manifest
            .matches_benchmark(&trace.run.manifest)
        {
            return Err(EvaluatorError::ManifestMismatch);
        }
        let reference_digest = self.references.digest()?;
        if trace.run.manifest.reference_digest.as_deref() != Some(reference_digest.as_str()) {
            return Err(EvaluatorError::ReferenceDigestMismatch);
        }
        for expected in self.metric_specs() {
            match trace
                .run
                .manifest
                .metrics
                .iter()
                .find(|declared| declared.name == expected.name)
            {
                None => {
                    return Err(EvaluatorError::MissingMetricDeclaration(expected.name));
                }
                Some(actual) if actual != &expected => {
                    return Err(EvaluatorError::MetricSpecMismatch(expected.name));
                }
                Some(_) => {}
            }
        }

        let mut seen_samples = BTreeSet::new();
        let mut hit_count = 0_u64;
        let mut recall_sum = 0.0_f64;
        let mut retrieval_sample_count = 0_u64;
        let mut error_count = 0_u64;
        let mut latency_sum = 0_u64;

        for (expected_step_index, step) in trace.steps.iter().enumerate() {
            if step.run_id != trace.run.run_id || step.step_index != expected_step_index as u64 {
                return Err(EvaluatorError::InvalidTrace(format!(
                    "step {:?} does not match the run sequence",
                    step.sample_id
                )));
            }
            if step.namespace != trace.run.isolated_namespace()
                || step.visibility_cutoff_ms != trace.run.manifest.visibility_cutoff_ms
            {
                return Err(EvaluatorError::InvalidTrace(format!(
                    "step {:?} has an invalid namespace or visibility cutoff",
                    step.sample_id
                )));
            }
            if step.sample_id.trim().is_empty() {
                return Err(EvaluatorError::InvalidTrace(
                    "sample_id must be non-empty".to_string(),
                ));
            }
            if !seen_samples.insert(step.sample_id.clone()) {
                return Err(EvaluatorError::InvalidTrace(format!(
                    "sample {:?} appears more than once",
                    step.sample_id
                )));
            }
            if let Some(ingest) = &step.ingest {
                validate_ingest_trace(ingest, &step.errors)?;
            } else if step
                .errors
                .iter()
                .any(|error| error.stage == TraceErrorStage::Ingest)
            {
                return Err(EvaluatorError::InvalidTrace(format!(
                    "sample {:?} contains an ingest error without an ingest trace",
                    step.sample_id
                )));
            }
            let reference = self
                .references
                .get(&step.sample_id)
                .ok_or_else(|| EvaluatorError::MissingReference(step.sample_id.clone()))?;
            let query = step.query.as_ref().ok_or_else(|| {
                EvaluatorError::InvalidTrace(format!(
                    "sample {:?} is missing its retrieval query",
                    step.sample_id
                ))
            })?;
            if query.namespace != step.namespace
                || query.dataset_namespace.as_deref() != Some(trace.run.manifest.namespace.as_str())
                || query.visibility_cutoff_ms != step.visibility_cutoff_ms
                || query.requested_hits == 0
            {
                return Err(EvaluatorError::InvalidTrace(format!(
                    "sample {:?} has an invalid retrieval query",
                    step.sample_id
                )));
            }
            let effective_k = self.top_k.min(query.requested_hits);
            let mut seen_record_ids = BTreeSet::new();
            let mut seen_ranks = BTreeSet::new();
            let mut retrieved = BTreeSet::new();
            for (hit_index, hit) in step.observed_hits.iter().enumerate() {
                let expected_rank = u32::try_from(hit_index + 1).map_err(|_| {
                    EvaluatorError::InvalidTrace(format!(
                        "sample {:?} contains too many observed hits",
                        step.sample_id
                    ))
                })?;
                if hit.record_id.trim().is_empty()
                    || hit.version == 0
                    || hit.rank == 0
                    || hit.rank != expected_rank
                    || hit.rank > query.requested_hits
                    || hit.score.is_some_and(|score| !score.is_finite())
                {
                    return Err(EvaluatorError::InvalidTrace(format!(
                        "sample {:?} contains an invalid observed hit",
                        step.sample_id
                    )));
                }
                if !seen_record_ids.insert(hit.record_id.clone()) {
                    return Err(EvaluatorError::InvalidTrace(format!(
                        "sample {:?} contains duplicate record id {:?}",
                        step.sample_id, hit.record_id
                    )));
                }
                let mut seen_source_record_ids = BTreeSet::new();
                for source_record_id in &hit.source_record_ids {
                    if source_record_id.trim().is_empty()
                        || !seen_source_record_ids.insert(source_record_id)
                    {
                        return Err(EvaluatorError::InvalidTrace(format!(
                            "sample {:?} contains an empty or duplicate source record id",
                            step.sample_id
                        )));
                    }
                }
                if self.references.manifest.identity == ReferenceIdentity::CanonicalSourceId
                    && hit.source_record_ids.is_empty()
                {
                    return Err(EvaluatorError::InvalidTrace(format!(
                        "sample {:?} contains a hit without canonical source record ids",
                        step.sample_id
                    )));
                }
                if !seen_ranks.insert(hit.rank) {
                    return Err(EvaluatorError::InvalidTrace(format!(
                        "sample {:?} contains duplicate hit rank {}",
                        step.sample_id, hit.rank
                    )));
                }
                let mut seen_evidence_ids = BTreeSet::new();
                for evidence_id in &hit.evidence_ids {
                    if evidence_id.trim().is_empty() || !seen_evidence_ids.insert(evidence_id) {
                        return Err(EvaluatorError::InvalidTrace(format!(
                            "sample {:?} contains an empty or duplicate evidence id",
                            step.sample_id
                        )));
                    }
                }
                if hit.rank <= effective_k {
                    match self.references.manifest.identity {
                        ReferenceIdentity::ProviderRecordId => {
                            retrieved.insert(hit.record_id.clone());
                        }
                        ReferenceIdentity::CanonicalSourceId => {
                            retrieved.extend(hit.source_record_ids.iter().cloned());
                        }
                    }
                }
            }
            if !reference.relevant_record_ids.is_empty() {
                let relevant_retrieved = retrieved
                    .intersection(&reference.relevant_record_ids)
                    .count();
                if relevant_retrieved > 0 {
                    hit_count += 1;
                }
                recall_sum +=
                    relevant_retrieved as f64 / reference.relevant_record_ids.len() as f64;
                retrieval_sample_count += 1;
            }
            if step
                .errors
                .iter()
                .any(|error| error.stage == TraceErrorStage::Retrieval)
            {
                error_count += 1;
            }
            latency_sum = latency_sum.saturating_add(step.telemetry.retrieval_latency_ms);
        }

        if let Some(sample_id) = self.references.missing_sample(&seen_samples) {
            return Err(EvaluatorError::MissingSample(sample_id));
        }

        let sample_count = trace.steps.len() as u64;
        let metrics = vec![
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: self.hit_rate_name(),
                value: if retrieval_sample_count == 0 {
                    0.0
                } else {
                    hit_count as f64 / retrieval_sample_count as f64
                },
                sample_count: retrieval_sample_count,
            },
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: self.recall_name(),
                value: if retrieval_sample_count == 0 {
                    0.0
                } else {
                    recall_sum / retrieval_sample_count as f64
                },
                sample_count: retrieval_sample_count,
            },
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: "retrieval.error_rate".to_string(),
                value: error_count as f64 / sample_count as f64,
                sample_count,
            },
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: "retrieval.mean_latency_ms".to_string(),
                value: latency_sum as f64 / sample_count as f64,
                sample_count,
            },
        ];
        Ok(EvaluationReport {
            run_id: trace.run.run_id.clone(),
            metrics,
        })
    }

    fn hit_rate_name(&self) -> String {
        match self.references.manifest.identity {
            ReferenceIdentity::ProviderRecordId => format!("retrieval.hit_rate@{}", self.top_k),
            ReferenceIdentity::CanonicalSourceId => {
                format!("retrieval.canonical_hit_rate@{}", self.top_k)
            }
        }
    }

    fn recall_name(&self) -> String {
        match self.references.manifest.identity {
            ReferenceIdentity::ProviderRecordId => format!("retrieval.recall@{}", self.top_k),
            ReferenceIdentity::CanonicalSourceId => {
                format!("retrieval.canonical_recall@{}", self.top_k)
            }
        }
    }
}

impl ContinualEvaluator {
    pub fn new(top_k: u32, references: ReferenceCatalog) -> Result<Self, EvaluatorError> {
        Ok(Self {
            retrieval: RetrievalEvaluator::new(top_k, references)?,
        })
    }

    /// Metric declarations required by a continual-memory benchmark.
    pub fn metric_specs(&self) -> Vec<MetricSpec> {
        let mut specs = self.retrieval.metric_specs();
        specs.extend([
            MetricSpec {
                name: "continual.ingest_success_rate".to_string(),
                version: METRIC_VERSION.to_string(),
                unit: "ratio".to_string(),
                direction: MetricDirection::HigherIsBetter,
            },
            MetricSpec {
                name: "continual.ingest_error_rate".to_string(),
                version: METRIC_VERSION.to_string(),
                unit: "ratio".to_string(),
                direction: MetricDirection::LowerIsBetter,
            },
            MetricSpec {
                name: "continual.ingest_dedup_rate".to_string(),
                version: METRIC_VERSION.to_string(),
                unit: "ratio".to_string(),
                direction: MetricDirection::HigherIsBetter,
            },
            MetricSpec {
                name: "continual.mean_ingest_latency_ms".to_string(),
                version: METRIC_VERSION.to_string(),
                unit: "milliseconds".to_string(),
                direction: MetricDirection::LowerIsBetter,
            },
        ]);
        specs
    }

    /// Evaluates retrieval and ingest contracts for every completed step.
    ///
    /// `ingest_dedup_rate` is the fraction of all dataset samples whose write
    /// receipt reports deduplication.  Failed writes therefore contribute to
    /// the denominator while contributing zero to the numerator, which keeps
    /// all three rates comparable over the same sample set.
    /// `mean_ingest_latency_ms` averages the ingest latency recorded for every
    /// step, including failed writes.
    pub fn evaluate(&self, trace: &RunTrace) -> Result<EvaluationReport, EvaluatorError> {
        let mut report = self.retrieval.evaluate(trace)?;
        for expected in self.metric_specs() {
            match trace
                .run
                .manifest
                .metrics
                .iter()
                .find(|declared| declared.name == expected.name)
            {
                None => return Err(EvaluatorError::MissingMetricDeclaration(expected.name)),
                Some(actual) if actual != &expected => {
                    return Err(EvaluatorError::MetricSpecMismatch(expected.name));
                }
                Some(_) => {}
            }
        }

        let mut success_count = 0_u64;
        let mut error_count = 0_u64;
        let mut deduplicated_count = 0_u64;
        let mut ingest_latency_sum = 0_u64;
        for step in &trace.steps {
            let ingest = step.ingest.as_ref().ok_or_else(|| {
                EvaluatorError::InvalidTrace(format!(
                    "sample {:?} is missing its ingest trace",
                    step.sample_id
                ))
            })?;
            validate_ingest_trace(ingest, &step.errors)?;
            if ingest.error.is_some() {
                error_count += 1;
            } else {
                success_count += 1;
            }
            if ingest.deduplicated {
                deduplicated_count += 1;
            }
            ingest_latency_sum =
                ingest_latency_sum.saturating_add(step.telemetry.ingest_latency_ms);
        }

        let sample_count = trace.steps.len() as u64;
        let denominator = sample_count as f64;
        report.metrics.extend([
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: "continual.ingest_success_rate".to_string(),
                value: success_count as f64 / denominator,
                sample_count,
            },
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: "continual.ingest_error_rate".to_string(),
                value: error_count as f64 / denominator,
                sample_count,
            },
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: "continual.ingest_dedup_rate".to_string(),
                value: deduplicated_count as f64 / denominator,
                sample_count,
            },
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: "continual.mean_ingest_latency_ms".to_string(),
                value: ingest_latency_sum as f64 / denominator,
                sample_count,
            },
        ]);
        Ok(report)
    }
}

impl OfflineBuildThenQueryEvaluator {
    pub fn new(top_k: u32, references: ReferenceCatalog) -> Result<Self, EvaluatorError> {
        Ok(Self {
            retrieval: RetrievalEvaluator::new(top_k, references)?,
        })
    }

    /// Metric declarations required by an offline build-then-query run.
    pub fn metric_specs(&self) -> Vec<MetricSpec> {
        let mut specs = self.retrieval.metric_specs();
        specs.extend([
            MetricSpec {
                name: "offline.build_success_rate".to_string(),
                version: METRIC_VERSION.to_string(),
                unit: "ratio".to_string(),
                direction: MetricDirection::HigherIsBetter,
            },
            MetricSpec {
                name: "offline.build_error_rate".to_string(),
                version: METRIC_VERSION.to_string(),
                unit: "ratio".to_string(),
                direction: MetricDirection::LowerIsBetter,
            },
            MetricSpec {
                name: "offline.build_dedup_rate".to_string(),
                version: METRIC_VERSION.to_string(),
                unit: "ratio".to_string(),
                direction: MetricDirection::HigherIsBetter,
            },
            MetricSpec {
                name: "offline.mean_build_latency_ms".to_string(),
                version: METRIC_VERSION.to_string(),
                unit: "milliseconds".to_string(),
                direction: MetricDirection::LowerIsBetter,
            },
        ]);
        specs
    }

    /// Evaluates query steps together with their persisted offline build
    /// receipts.  The caller should obtain `builds` with
    /// `pi_agent_benchmark::read_builds`; that API replays the JSONL record
    /// order and rejects a build appended after the query phase.
    pub fn evaluate(
        &self,
        trace: &RunTrace,
        builds: &[BuildTrace],
    ) -> Result<EvaluationReport, EvaluatorError> {
        if trace.run.manifest.lifecycle != BenchmarkLifecycle::OfflineBuildThenQuery {
            return Err(EvaluatorError::InvalidTrace(
                "offline evaluator requires offline_build_then_query lifecycle".to_string(),
            ));
        }
        if trace.steps.iter().any(|step| step.ingest.is_some()) {
            return Err(EvaluatorError::InvalidTrace(
                "offline query steps must not contain continual ingest traces".to_string(),
            ));
        }

        let mut report = self.retrieval.evaluate(trace)?;
        self.validate_metric_declarations(trace)?;
        self.validate_builds(trace, builds)?;

        let mut success_count = 0_u64;
        let mut deduplicated_count = 0_u64;
        let mut latency_sum = 0_u64;
        for build in builds {
            success_count += 1;
            if build.deduplicated {
                deduplicated_count += 1;
            }
            latency_sum = latency_sum.saturating_add(build.latency_ms);
        }

        let sample_count = builds.len() as u64;
        let denominator = sample_count as f64;
        report.metrics.extend([
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: "offline.build_success_rate".to_string(),
                value: success_count as f64 / denominator,
                sample_count,
            },
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: "offline.build_error_rate".to_string(),
                // A failed build aborts the runner before it can finalize a
                // complete offline run. Failed receipts are therefore
                // invalid here and this metric is necessarily zero.
                value: 0.0,
                sample_count,
            },
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: "offline.build_dedup_rate".to_string(),
                value: deduplicated_count as f64 / denominator,
                sample_count,
            },
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: "offline.mean_build_latency_ms".to_string(),
                value: latency_sum as f64 / denominator,
                sample_count,
            },
        ]);
        Ok(report)
    }

    /// Reads a complete JSONL trace and evaluates its build phase and query
    /// phase together.  This is the preferred entry point for persisted runs
    /// because `read_builds` validates that every build record precedes the
    /// first query step.
    pub fn evaluate_path(
        &self,
        path: impl AsRef<Path>,
        run_id: &str,
    ) -> Result<EvaluationReport, EvaluatorError> {
        let trace = read_run(path.as_ref(), run_id)?;
        let builds = read_builds(path, run_id)?;
        self.evaluate(&trace, &builds)
    }

    fn validate_metric_declarations(&self, trace: &RunTrace) -> Result<(), EvaluatorError> {
        for expected in self.metric_specs() {
            match trace
                .run
                .manifest
                .metrics
                .iter()
                .find(|declared| declared.name == expected.name)
            {
                None => return Err(EvaluatorError::MissingMetricDeclaration(expected.name)),
                Some(actual) if actual != &expected => {
                    return Err(EvaluatorError::MetricSpecMismatch(expected.name));
                }
                Some(_) => {}
            }
        }
        Ok(())
    }

    fn validate_builds(
        &self,
        trace: &RunTrace,
        builds: &[BuildTrace],
    ) -> Result<(), EvaluatorError> {
        if builds.is_empty() {
            return Err(EvaluatorError::InvalidTrace(
                "offline trace must contain at least one build record".to_string(),
            ));
        }
        if builds.len() != trace.steps.len() {
            return Err(EvaluatorError::InvalidTrace(format!(
                "offline build count {} does not match query step count {}",
                builds.len(),
                trace.steps.len()
            )));
        }

        let mut sample_ids = BTreeSet::new();
        for (expected_index, build) in builds.iter().enumerate() {
            validate_build_trace(build, &trace.run)?;
            if build.error.is_some() {
                return Err(EvaluatorError::InvalidTrace(format!(
                    "offline build sample {:?} failed; completed runs cannot contain failed builds",
                    build.sample_id
                )));
            }
            if build.build_index != expected_index as u64 {
                return Err(EvaluatorError::InvalidTrace(format!(
                    "offline build index {} is expected at position {}",
                    build.build_index, expected_index
                )));
            }
            let step = &trace.steps[expected_index];
            if build.sample_id != step.sample_id {
                return Err(EvaluatorError::InvalidTrace(format!(
                    "offline build sample {:?} does not match query sample {:?}",
                    build.sample_id, step.sample_id
                )));
            }
            if build.sample_digest.as_deref() != step.sample_digest.as_deref() {
                return Err(EvaluatorError::InvalidTrace(format!(
                    "offline build sample {:?} digest does not match its query step",
                    build.sample_id
                )));
            }
            if !sample_ids.insert(build.sample_id.clone()) {
                return Err(EvaluatorError::InvalidTrace(format!(
                    "offline build sample {:?} appears more than once",
                    build.sample_id
                )));
            }
            if build.sample_digest.is_none() {
                return Err(EvaluatorError::InvalidTrace(format!(
                    "offline build sample {:?} is missing its sample digest",
                    build.sample_id
                )));
            }
        }
        Ok(())
    }
}

/// Metric values ready to be appended to the trace.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationReport {
    pub run_id: String,
    pub metrics: Vec<MetricValue>,
}

impl EvaluationReport {
    /// Appends metrics idempotently. Existing equal values are left untouched;
    /// a different value for the same metric is treated as a conflict.
    pub fn append_to_trace(&self, path: impl AsRef<std::path::Path>) -> Result<(), EvaluatorError> {
        let mut writer = JsonlTraceWriter::open(path.as_ref(), self.run_id.clone())?;
        let existing = read_run(path, &self.run_id)?;
        let mut existing_names = BTreeSet::new();
        for metric in &existing.metrics {
            if !existing_names.insert(metric.name.clone()) {
                return Err(EvaluatorError::DuplicateMetric(metric.name.clone()));
            }
        }
        let mut new_names = BTreeSet::new();
        for metric in &self.metrics {
            if !new_names.insert(metric.name.clone()) {
                return Err(EvaluatorError::DuplicateMetric(metric.name.clone()));
            }
            if let Some(previous) = existing
                .metrics
                .iter()
                .find(|value| value.name == metric.name)
            {
                if previous != metric {
                    return Err(EvaluatorError::MetricConflict(metric.name.clone()));
                }
            }
        }
        // Complete the conflict/duplicate preflight before writing any metric.
        for metric in &self.metrics {
            if existing_names.contains(&metric.name) {
                continue;
            }
            writer.append_metric(metric)?;
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum EvaluatorError {
    #[error("invalid evaluator configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid reference catalog: {0}")]
    InvalidReference(String),
    #[error("invalid reference catalog sidecar: {0}")]
    InvalidReferenceSidecar(String),
    #[error("reference catalog sidecar {operation} failed at {path:?}: {source}")]
    ReferenceSidecarIo {
        operation: &'static str,
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("reference catalog sidecar digest mismatch: expected {expected:?}, actual {actual:?}")]
    ReferenceSidecarDigestMismatch { expected: String, actual: String },
    #[error("reference catalog sidecar identity mismatch: expected {expected}, actual {actual}")]
    ReferenceSidecarIdentityMismatch { expected: String, actual: String },
    #[error("trace is not finalized")]
    IncompleteTrace,
    #[error("reference is missing for sample {0:?}")]
    MissingReference(String),
    #[error("trace is missing dataset sample {0:?}")]
    MissingSample(String),
    #[error("reference catalog does not match the benchmark manifest")]
    ManifestMismatch,
    #[error("benchmark manifest is not bound to this reference catalog digest")]
    ReferenceDigestMismatch,
    #[error("metric {0:?} is not declared by the benchmark manifest")]
    MissingMetricDeclaration(String),
    #[error("metric {0:?} has a different declaration in the benchmark manifest")]
    MetricSpecMismatch(String),
    #[error("invalid trace: {0}")]
    InvalidTrace(String),
    #[error("benchmark run is missing its runtime operation id")]
    MissingRuntimeOperationRunId,
    #[error("benchmark run has a nil runtime operation id")]
    InvalidRuntimeOperationRunId,
    #[error("runtime link is missing for sample {0:?}")]
    MissingRuntimeLink(String),
    #[error("runtime run id mismatch for sample {sample_id:?}: expected {expected}, got {actual}")]
    RuntimeRunIdMismatch {
        sample_id: String,
        expected: String,
        actual: String,
    },
    #[error("runtime turn id is nil for sample {0:?}")]
    InvalidRuntimeTurnId(String),
    #[error("runtime turn id is duplicated for sample {0:?}")]
    DuplicateRuntimeTurnId(String),
    #[error("runtime memory decision for sample {sample_id:?} has invalid {field}")]
    InvalidRuntimeMemoryDecision { sample_id: String, field: String },
    #[error("runtime ledger is missing event {event_sequence} for sample {sample_id:?}")]
    MissingRuntimeLedgerEvent {
        sample_id: String,
        event_sequence: u64,
    },
    #[error(
        "runtime ledger field {field:?} mismatch for sample {sample_id:?} at event {event_sequence}: expected {expected}, got {actual}"
    )]
    RuntimeLedgerFieldMismatch {
        sample_id: String,
        event_sequence: u64,
        field: String,
        expected: String,
        actual: String,
    },
    #[error("invalid runtime ledger payload field {field:?} at event {event_sequence}")]
    InvalidRuntimeLedgerPayload { event_sequence: u64, field: String },
    #[error(transparent)]
    RuntimeLedger(#[from] pi_agent_runtime::JsonlLedgerError),
    #[error("metric {0:?} already exists with a different value")]
    MetricConflict(String),
    #[error("metric {0:?} appears more than once")]
    DuplicateMetric(String),
    #[error(transparent)]
    Trace(#[from] TraceStoreError),
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
    };

    use pi_agent_benchmark::{
        AgentObservation, ArtifactValidationPolicy, BenchmarkLifecycle, BenchmarkManifest,
        BenchmarkRun, BuildTrace, IngestTrace, JsonlTraceWriter, MethodManifest, ObservationPart,
        ObservedHit, QueryTrace, RuntimeMemoryDecisionLink, RuntimeStepLink, StepTelemetry,
        StepTrace, TerminalReason, TraceErrorEntry,
    };
    use pi_agent_runtime::{EventDraft, EventKind, JsonlLedger, Revision, RunId};
    use serde_json::json;
    use tempfile::tempdir;
    use uuid::Uuid;

    use super::*;

    fn trace(metrics: Vec<MetricSpec>) -> RunTrace {
        let mut manifest = BenchmarkManifest::new(
            "retrieval",
            "bench/test",
            MethodManifest {
                method_id: "projection".into(),
                version: "1".into(),
                config: BTreeMap::new(),
            },
            "dataset-1",
            Some(100),
        );
        manifest.metrics = metrics;
        let mut run = BenchmarkRun::new("run-1", manifest, 1);
        let catalog = ReferenceCatalog::for_benchmark(&run.manifest, references()).unwrap();
        run.manifest.reference_digest = Some(catalog.digest().unwrap());
        let run_id = run.run_id.clone();
        let namespace = run.isolated_namespace();
        let step = |sample_id: &str, hit_id: &str, error: bool| StepTrace {
            run_id: run_id.clone(),
            step_index: if sample_id == "sample-0" { 0 } else { 1 },
            sample_id: sample_id.into(),
            sample_digest: None,
            namespace: namespace.clone(),
            visibility_cutoff_ms: Some(100),
            observation: AgentObservation {
                parts: vec![ObservationPart::Text("visible".into())],
            },
            ingest: None,
            query: Some(QueryTrace {
                namespace: namespace.clone(),
                dataset_namespace: Some("bench/test".into()),
                query_digest: Some("sha256:query".into()),
                snapshot_id: None,
                input_digest: None,
                visibility_cutoff_ms: Some(100),
                requested_hits: 2,
                retrieval_truncated: false,
                telemetry: None,
            }),
            observed_hits: vec![ObservedHit {
                record_id: hit_id.into(),
                source_record_ids: Vec::new(),
                version: 1,
                rank: 1,
                score: Some(0.9),
                evidence_ids: Vec::new(),
            }],
            agent_answer: None,
            action: None,
            reward: None,
            terminal_reason: if sample_id == "sample-1" {
                Some(TerminalReason::Success)
            } else {
                None
            },
            telemetry: StepTelemetry {
                retrieval_latency_ms: if sample_id == "sample-0" { 10 } else { 30 },
                ..StepTelemetry::default()
            },
            errors: if error {
                vec![TraceErrorEntry {
                    stage: TraceErrorStage::Retrieval,
                    code: "timeout".into(),
                    message: "timed out".into(),
                    retryable: true,
                }]
            } else {
                Vec::new()
            },
            runtime_link: None,
            terminal: sample_id == "sample-1",
        };
        RunTrace {
            run,
            steps: vec![
                step("sample-0", "memory-1", false),
                step("sample-1", "memory-9", true),
            ],
            metrics: Vec::new(),
            finished_at_ms: Some(200),
            complete: true,
        }
    }

    fn evaluator() -> RetrievalEvaluator {
        let manifest = BenchmarkManifest::new(
            "retrieval",
            "bench/test",
            MethodManifest {
                method_id: "projection".into(),
                version: "1".into(),
                config: BTreeMap::new(),
            },
            "dataset-1",
            Some(100),
        );
        let references = ReferenceCatalog::for_benchmark(&manifest, references()).unwrap();
        RetrievalEvaluator::new(1, references).unwrap()
    }

    fn continual_evaluator() -> ContinualEvaluator {
        let manifest = BenchmarkManifest::new(
            "retrieval",
            "bench/test",
            MethodManifest {
                method_id: "projection".into(),
                version: "1".into(),
                config: BTreeMap::new(),
            },
            "dataset-1",
            Some(100),
        );
        let references = ReferenceCatalog::for_benchmark(&manifest, references()).unwrap();
        ContinualEvaluator::new(1, references).unwrap()
    }

    fn continual_trace(evaluator: &ContinualEvaluator) -> RunTrace {
        let mut value = trace(evaluator.metric_specs());
        value.steps[0].ingest = Some(IngestTrace {
            request_digest: format!("sha256:{}", "a".repeat(64)),
            item_ids: vec![Uuid::from_u128(1).to_string()],
            relation_ids: vec![Uuid::from_u128(2).to_string()],
            deduplicated: true,
            snapshot_id: Some("snapshot-1".into()),
            completed_at_ms: Some(1),
            error: None,
            telemetry: None,
        });
        let ingest_error = TraceErrorEntry {
            stage: TraceErrorStage::Ingest,
            code: "backend".into(),
            message: "write failed".into(),
            retryable: true,
        };
        value.steps[1].ingest = Some(IngestTrace {
            request_digest: format!("sha256:{}", "b".repeat(64)),
            item_ids: Vec::new(),
            relation_ids: Vec::new(),
            deduplicated: false,
            snapshot_id: None,
            completed_at_ms: Some(2),
            error: Some(ingest_error.clone()),
            telemetry: None,
        });
        value.steps[1].errors = vec![ingest_error];
        value.steps[0].telemetry.ingest_latency_ms = 10;
        value.steps[1].telemetry.ingest_latency_ms = 30;
        value
    }

    fn offline_trace(evaluator: &OfflineBuildThenQueryEvaluator) -> (RunTrace, Vec<BuildTrace>) {
        let mut value = trace(evaluator.metric_specs());
        value.run.manifest.lifecycle = BenchmarkLifecycle::OfflineBuildThenQuery;
        let run_id = value.run.run_id.clone();
        let namespace = value.run.isolated_namespace();
        let first_build = BuildTrace {
            run_id: run_id.clone(),
            build_index: 0,
            sample_id: "sample-0".into(),
            sample_digest: Some(format!("sha256:{}", "0".repeat(64))),
            namespace: namespace.clone(),
            request_digest: format!("sha256:{}", "a".repeat(64)),
            item_ids: vec![Uuid::from_u128(1).to_string()],
            relation_ids: Vec::new(),
            deduplicated: true,
            snapshot_id: Some("snapshot-0".into()),
            completed_at_ms: Some(1),
            latency_ms: 10,
            error: None,
            telemetry: None,
        };
        let second_build = BuildTrace {
            run_id,
            build_index: 1,
            sample_id: "sample-1".into(),
            sample_digest: Some(format!("sha256:{}", "1".repeat(64))),
            namespace,
            request_digest: format!("sha256:{}", "b".repeat(64)),
            item_ids: vec![Uuid::from_u128(2).to_string()],
            relation_ids: Vec::new(),
            deduplicated: false,
            snapshot_id: Some("snapshot-1".into()),
            completed_at_ms: Some(2),
            latency_ms: 30,
            error: None,
            telemetry: None,
        };
        value.steps[0].sample_digest = first_build.sample_digest.clone();
        value.steps[1].sample_digest = second_build.sample_digest.clone();
        (value, vec![first_build, second_build])
    }

    fn offline_evaluator() -> OfflineBuildThenQueryEvaluator {
        let manifest = BenchmarkManifest::new(
            "retrieval",
            "bench/test",
            MethodManifest {
                method_id: "projection".into(),
                version: "1".into(),
                config: BTreeMap::new(),
            },
            "dataset-1",
            Some(100),
        );
        let references = ReferenceCatalog::for_benchmark(&manifest, references()).unwrap();
        OfflineBuildThenQueryEvaluator::new(1, references).unwrap()
    }

    fn runtime_evaluator() -> RuntimeLinkedEvaluator {
        let manifest = BenchmarkManifest::new(
            "retrieval",
            "bench/test",
            MethodManifest {
                method_id: "projection".into(),
                version: "1".into(),
                config: BTreeMap::new(),
            },
            "dataset-1",
            Some(100),
        );
        let references = ReferenceCatalog::for_benchmark(&manifest, references()).unwrap();
        RuntimeLinkedEvaluator::new(1, references).unwrap()
    }

    fn linked_trace(evaluator: &RuntimeLinkedEvaluator) -> RunTrace {
        let mut value = trace(evaluator.metric_specs());
        let operation_run_id = Uuid::from_u128(42);
        value.run.operation_run_id = Some(operation_run_id);
        for (index, step) in value.steps.iter_mut().enumerate() {
            step.runtime_link = Some(RuntimeStepLink {
                runtime_run_id: operation_run_id,
                turn_id: Uuid::from_u128(100 + index as u128),
                memory_decision: Some(RuntimeMemoryDecisionLink {
                    proposal_id: Uuid::from_u128(200 + index as u128),
                    event_sequence: index as u64 + 1,
                    decision_kind: "injected".into(),
                }),
                memory_decisions: Vec::new(),
            });
        }
        value
    }

    fn linked_trace_with_multiple_decisions(evaluator: &RuntimeLinkedEvaluator) -> RunTrace {
        let mut value = linked_trace(evaluator);
        let first = value.steps[0]
            .runtime_link
            .as_mut()
            .expect("linked trace should have a runtime link");
        let first_decision = first
            .memory_decision
            .clone()
            .expect("linked trace should have a memory decision");
        first.memory_decisions = vec![
            first_decision,
            RuntimeMemoryDecisionLink {
                proposal_id: Uuid::from_u128(201),
                event_sequence: 2,
                decision_kind: "skipped".into(),
            },
        ];
        value.steps[1]
            .runtime_link
            .as_mut()
            .expect("linked trace should have a runtime link")
            .memory_decision
            .as_mut()
            .expect("linked trace should have a memory decision")
            .event_sequence = 3;
        value
    }

    fn write_runtime_ledger(
        trace: &RunTrace,
        path: &std::path::Path,
        first_turn_override: Option<Uuid>,
    ) {
        let run_id = RunId::from_uuid(trace.run.operation_run_id.unwrap());
        let mut ledger = JsonlLedger::open(path).unwrap();
        let mut event_index = 0u64;
        for step in &trace.steps {
            let link = step.runtime_link.as_ref().unwrap();
            for decision in runtime_memory_decisions(link) {
                let turn_id = if event_index == 0 {
                    first_turn_override.unwrap_or(link.turn_id)
                } else {
                    link.turn_id
                };
                let event_kind = if decision.decision_kind == "injected" {
                    "memory_injected"
                } else {
                    "memory_skipped"
                };
                let payload = json!({
                    "status": decision.decision_kind,
                    "run_id": run_id,
                    "turn_id": turn_id,
                    "details": { "proposal_id": decision.proposal_id },
                });
                ledger
                    .append(vec![EventDraft::new(
                        run_id,
                        Revision::INITIAL,
                        EventKind::new(event_kind).unwrap(),
                        payload,
                        1_000 + event_index,
                    )])
                    .unwrap();
                event_index += 1;
            }
        }
        ledger.close().unwrap();
    }

    fn references() -> Vec<SampleReference> {
        vec![
            SampleReference {
                sample_id: "sample-0".into(),
                relevant_record_ids: BTreeSet::from(["memory-1".into()]),
            },
            SampleReference {
                sample_id: "sample-1".into(),
                relevant_record_ids: BTreeSet::from(["memory-2".into()]),
            },
        ]
    }

    #[test]
    fn computes_hit_recall_error_and_latency_metrics() {
        let evaluator = evaluator();
        let report = evaluator
            .evaluate(&trace(evaluator.metric_specs()))
            .unwrap();

        assert_eq!(report.metrics[0].name, "retrieval.hit_rate@1");
        assert_eq!(report.metrics[0].value, 0.5);
        assert_eq!(report.metrics[1].value, 0.5);
        assert_eq!(report.metrics[2].value, 0.5);
        assert_eq!(report.metrics[3].value, 20.0);
    }

    #[test]
    fn empty_relevance_requires_an_explicit_catalog_policy() {
        let manifest = BenchmarkManifest::new(
            "retrieval",
            "bench/test",
            MethodManifest {
                method_id: "projection".into(),
                version: "1".into(),
                config: BTreeMap::new(),
            },
            "dataset-1",
            Some(100),
        );
        let reference = SampleReference {
            sample_id: "sample-abs".into(),
            relevant_record_ids: BTreeSet::new(),
        };
        assert!(matches!(
            ReferenceCatalog::for_benchmark(&manifest, vec![reference.clone()]),
            Err(EvaluatorError::InvalidReference(message))
                if message.contains("allows empty relevance")
        ));

        let mut reference_manifest = ReferenceCatalogManifest::from_benchmark(&manifest);
        reference_manifest.allow_empty_relevance = true;
        let catalog = ReferenceCatalog::new(reference_manifest, vec![reference]).unwrap();
        assert!(catalog
            .samples
            .values()
            .next()
            .unwrap()
            .relevant_record_ids
            .is_empty());
    }

    #[test]
    fn retrieval_metrics_exclude_empty_relevance_but_keep_latency_and_errors() {
        let mut value = trace(Vec::new());
        let mut reference_manifest = ReferenceCatalogManifest::from_benchmark(&value.run.manifest);
        reference_manifest.allow_empty_relevance = true;
        let catalog = ReferenceCatalog::new(
            reference_manifest,
            vec![
                SampleReference {
                    sample_id: "sample-0".into(),
                    relevant_record_ids: BTreeSet::new(),
                },
                SampleReference {
                    sample_id: "sample-1".into(),
                    relevant_record_ids: BTreeSet::from(["memory-2".into()]),
                },
            ],
        )
        .unwrap();
        let evaluator = RetrievalEvaluator::new(1, catalog.clone()).unwrap();
        value.run.manifest.metrics = evaluator.metric_specs();
        value.run.manifest.reference_digest = Some(catalog.digest().unwrap());

        let report = evaluator.evaluate(&value).unwrap();
        assert_eq!(report.metrics[0].sample_count, 1);
        assert_eq!(report.metrics[1].sample_count, 1);
        assert_eq!(report.metrics[2].sample_count, 2);
        assert_eq!(report.metrics[3].sample_count, 2);
    }

    #[test]
    fn canonical_identity_evaluates_source_ids_and_rejects_missing_lineage() {
        let mut value = trace(Vec::new());
        let mut manifest = ReferenceCatalogManifest::from_benchmark(&value.run.manifest);
        manifest.identity = ReferenceIdentity::CanonicalSourceId;
        let catalog = ReferenceCatalog::new(
            manifest,
            vec![
                SampleReference {
                    sample_id: "sample-0".into(),
                    relevant_record_ids: BTreeSet::from(["source-1".into()]),
                },
                SampleReference {
                    sample_id: "sample-1".into(),
                    relevant_record_ids: BTreeSet::from(["source-2".into()]),
                },
            ],
        )
        .unwrap();
        let evaluator = RetrievalEvaluator::new(1, catalog).unwrap();
        value.run.manifest.metrics = evaluator.metric_specs();
        value.run.manifest.reference_digest = Some(evaluator.references.digest().unwrap());
        value.steps[0].observed_hits[0].source_record_ids = vec!["source-1".into()];
        value.steps[1].observed_hits[0].source_record_ids = vec!["source-2".into()];

        let report = evaluator.evaluate(&value).unwrap();
        assert_eq!(report.metrics[0].name, "retrieval.canonical_hit_rate@1");
        assert_eq!(report.metrics[1].name, "retrieval.canonical_recall@1");
        assert_eq!(report.metrics[0].value, 1.0);
        assert_eq!(report.metrics[1].value, 1.0);

        value.steps[0].observed_hits[0].source_record_ids.clear();
        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::InvalidTrace(message))
                if message.contains("without canonical source record ids")
        ));
    }

    #[test]
    fn continual_evaluator_merges_ingest_metrics_with_retrieval_metrics() {
        let evaluator = continual_evaluator();
        let report = evaluator.evaluate(&continual_trace(&evaluator)).unwrap();

        assert_eq!(report.metrics.len(), 8);
        assert_eq!(report.metrics[4].name, "continual.ingest_success_rate");
        assert_eq!(report.metrics[4].value, 0.5);
        assert_eq!(report.metrics[5].name, "continual.ingest_error_rate");
        assert_eq!(report.metrics[5].value, 0.5);
        assert_eq!(report.metrics[6].name, "continual.ingest_dedup_rate");
        assert_eq!(report.metrics[6].value, 0.5);
        assert_eq!(report.metrics[7].name, "continual.mean_ingest_latency_ms");
        assert_eq!(report.metrics[7].value, 20.0);
        assert!(report.metrics.iter().all(|metric| metric.sample_count == 2));
    }

    #[test]
    fn continual_evaluator_requires_an_ingest_trace_for_every_step() {
        let evaluator = continual_evaluator();
        let mut value = continual_trace(&evaluator);
        value.steps[0].ingest = None;

        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::InvalidTrace(message)) if message.contains("missing its ingest trace")
        ));
    }

    #[test]
    fn continual_evaluator_reuses_ingest_receipt_validation() {
        let evaluator = continual_evaluator();
        let mut value = continual_trace(&evaluator);
        value.steps[0].ingest.as_mut().unwrap().request_digest = "sha256:invalid".into();

        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::Trace(TraceStoreError::InvalidStep(message)))
                if message.contains("request_digest")
        ));
    }

    #[test]
    fn offline_evaluator_reports_build_success_error_dedup_and_latency() {
        let evaluator = offline_evaluator();
        let (trace, builds) = offline_trace(&evaluator);
        let report = evaluator.evaluate(&trace, &builds).unwrap();

        assert_eq!(report.metrics.len(), 8);
        assert_eq!(report.metrics[4].name, "offline.build_success_rate");
        assert_eq!(report.metrics[4].value, 1.0);
        assert_eq!(report.metrics[5].name, "offline.build_error_rate");
        assert_eq!(report.metrics[5].value, 0.0);
        assert_eq!(report.metrics[6].name, "offline.build_dedup_rate");
        assert_eq!(report.metrics[6].value, 0.5);
        assert_eq!(report.metrics[7].name, "offline.mean_build_latency_ms");
        assert_eq!(report.metrics[7].value, 20.0);
        assert!(report.metrics.iter().all(|metric| metric.sample_count == 2));
    }

    #[test]
    fn offline_evaluator_rejects_out_of_order_or_mismatched_builds() {
        let evaluator = offline_evaluator();
        let (trace, mut builds) = offline_trace(&evaluator);
        builds[1].build_index = 3;
        assert!(matches!(
            evaluator.evaluate(&trace, &builds),
            Err(EvaluatorError::InvalidTrace(message))
                if message.contains("build index")
        ));

        let (trace, mut builds) = offline_trace(&evaluator);
        builds[0].sample_id = "sample-1".into();
        assert!(matches!(
            evaluator.evaluate(&trace, &builds),
            Err(EvaluatorError::InvalidTrace(message))
                if message.contains("does not match query sample")
        ));

        let (mut trace, builds) = offline_trace(&evaluator);
        trace.steps[0].sample_digest = Some("sha256:different".into());
        assert!(matches!(
            evaluator.evaluate(&trace, &builds),
            Err(EvaluatorError::InvalidTrace(message))
                if message.contains("digest does not match")
        ));
    }

    #[test]
    fn offline_evaluator_rejects_failed_builds_in_a_completed_trace() {
        let evaluator = offline_evaluator();
        let (trace, mut builds) = offline_trace(&evaluator);
        builds[1].item_ids.clear();
        builds[1].snapshot_id = None;
        builds[1].error = Some(TraceErrorEntry {
            stage: TraceErrorStage::Build,
            code: "backend".into(),
            message: "build failed".into(),
            retryable: true,
        });

        assert!(matches!(
            evaluator.evaluate(&trace, &builds),
            Err(EvaluatorError::InvalidTrace(message))
                if message.contains("completed runs cannot contain failed builds")
        ));
    }

    #[test]
    fn offline_evaluator_rejects_continual_ingest_on_query_steps() {
        let evaluator = offline_evaluator();
        let (mut trace, builds) = offline_trace(&evaluator);
        trace.steps[0].ingest = Some(IngestTrace {
            request_digest: format!("sha256:{}", "c".repeat(64)),
            item_ids: vec![Uuid::from_u128(3).to_string()],
            relation_ids: Vec::new(),
            deduplicated: false,
            snapshot_id: Some("snapshot-query".into()),
            completed_at_ms: Some(3),
            error: None,
            telemetry: None,
        });

        assert!(matches!(
            evaluator.evaluate(&trace, &builds),
            Err(EvaluatorError::InvalidTrace(message))
                if message.contains("must not contain continual ingest traces")
        ));
    }

    #[test]
    fn offline_evaluator_reads_builds_from_persisted_trace() {
        let evaluator = offline_evaluator();
        let (trace, builds) = offline_trace(&evaluator);
        let directory = tempdir().unwrap();
        let path = directory.path().join("offline.jsonl");
        let mut writer = JsonlTraceWriter::open(&path, trace.run.run_id.clone()).unwrap();
        writer.append_run(&trace.run).unwrap();
        for build in &builds {
            writer.append_build(build).unwrap();
        }
        for step in &trace.steps {
            writer.append_step(step).unwrap();
        }
        writer.finalize(trace.finished_at_ms.unwrap()).unwrap();

        let report = evaluator.evaluate_path(&path, &trace.run.run_id).unwrap();
        assert_eq!(report.metrics[4].value, 1.0);
    }

    #[test]
    fn offline_evaluator_requires_explicit_lifecycle() {
        let evaluator = offline_evaluator();
        let mut value = trace(evaluator.metric_specs());
        let builds = Vec::new();
        assert!(matches!(
            evaluator.evaluate(&value, &builds),
            Err(EvaluatorError::InvalidTrace(message))
                if message.contains("offline_build_then_query")
        ));
        value.run.manifest.lifecycle = BenchmarkLifecycle::Continual;
        assert!(matches!(
            evaluator.evaluate(&value, &builds),
            Err(EvaluatorError::InvalidTrace(message))
                if message.contains("offline_build_then_query")
        ));
    }

    #[test]
    fn errors_are_counted_without_turning_into_a_hit() {
        let evaluator = evaluator();
        let mut value = trace(evaluator.metric_specs());
        value.steps[1].errors = vec![TraceErrorEntry {
            stage: TraceErrorStage::Retrieval,
            code: "backend".into(),
            message: "failed".into(),
            retryable: true,
        }];
        value.steps[1].observed_hits.clear();
        let report = evaluator.evaluate(&value).unwrap();

        assert_eq!(report.metrics[0].value, 0.5);
        assert_eq!(report.metrics[2].value, 0.5);
    }

    #[test]
    fn agent_errors_do_not_change_retrieval_error_rate() {
        let evaluator = evaluator();
        let mut value = trace(evaluator.metric_specs());
        value.steps[1].errors = vec![TraceErrorEntry {
            stage: TraceErrorStage::Agent,
            code: "model_timeout".into(),
            message: "model did not finish".into(),
            retryable: true,
        }];
        let report = evaluator.evaluate(&value).unwrap();

        assert_eq!(report.metrics[2].value, 0.0);
    }

    #[test]
    fn runtime_linked_evaluator_requires_a_link_for_every_step() {
        let evaluator = runtime_evaluator();
        let mut value = trace(evaluator.metric_specs());
        value.run.operation_run_id = Some(Uuid::from_u128(42));

        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::MissingRuntimeLink(sample_id)) if sample_id == "sample-0"
        ));
    }

    #[test]
    fn runtime_linked_evaluator_rejects_a_run_id_mismatch() {
        let evaluator = runtime_evaluator();
        let mut value = linked_trace(&evaluator);
        value.steps[1].runtime_link.as_mut().unwrap().runtime_run_id = Uuid::from_u128(43);

        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::RuntimeRunIdMismatch { sample_id, .. }) if sample_id == "sample-1"
        ));
    }

    #[test]
    fn runtime_linked_evaluator_accepts_unique_turns_and_memory_decisions() {
        let evaluator = runtime_evaluator();
        let value = linked_trace(&evaluator);

        let report = evaluator.evaluate(&value).unwrap();
        assert_eq!(report.metrics[0].value, 0.5);
    }

    #[test]
    fn runtime_linked_evaluator_validates_all_memory_decisions() {
        let evaluator = runtime_evaluator();
        let value = linked_trace_with_multiple_decisions(&evaluator);

        evaluator.validate_runtime_links(&value).unwrap();
        assert_eq!(
            value.steps[0]
                .runtime_link
                .as_ref()
                .unwrap()
                .memory_decisions
                .len(),
            2
        );
    }

    #[test]
    fn runtime_linked_evaluator_audits_the_durable_ledger() {
        let evaluator = runtime_evaluator();
        let value = linked_trace(&evaluator);
        let directory = tempdir().unwrap();
        let ledger_path = directory.path().join("runtime.jsonl");
        write_runtime_ledger(&value, &ledger_path, None);

        let report = evaluator
            .evaluate_with_ledger(&value, &ledger_path)
            .unwrap();
        assert_eq!(report.metrics[0].value, 0.5);
    }

    #[test]
    fn runtime_ledger_tampering_is_rejected() {
        let evaluator = runtime_evaluator();
        let value = linked_trace(&evaluator);
        let directory = tempdir().unwrap();
        let ledger_path = directory.path().join("runtime.jsonl");
        write_runtime_ledger(&value, &ledger_path, Some(Uuid::from_u128(999)));

        assert!(matches!(
            evaluator.evaluate_with_ledger(&value, &ledger_path),
            Err(EvaluatorError::RuntimeLedgerFieldMismatch {
                field,
                sample_id,
                ..
            }) if field == "payload.turn_id" && sample_id == "sample-0"
        ));
    }

    #[test]
    fn runtime_ledger_audits_all_memory_decisions() {
        let evaluator = runtime_evaluator();
        let value = linked_trace_with_multiple_decisions(&evaluator);
        let directory = tempdir().unwrap();
        let ledger_path = directory.path().join("runtime.jsonl");
        write_runtime_ledger(&value, &ledger_path, None);

        evaluator
            .evaluate_with_ledger(&value, &ledger_path)
            .unwrap();

        let mut tampered = value;
        tampered.steps[0]
            .runtime_link
            .as_mut()
            .unwrap()
            .memory_decisions[1]
            .proposal_id = Uuid::from_u128(998);
        assert!(matches!(
            evaluator.evaluate_with_ledger(&tampered, &ledger_path),
            Err(EvaluatorError::RuntimeLedgerFieldMismatch {
                field,
                sample_id,
                event_sequence: 2,
                ..
            }) if field == "details.proposal_id" && sample_id == "sample-0"
        ));
    }

    #[test]
    fn trace_tampering_is_rejected_against_the_durable_ledger() {
        let evaluator = runtime_evaluator();
        let mut value = linked_trace(&evaluator);
        let directory = tempdir().unwrap();
        let ledger_path = directory.path().join("runtime.jsonl");
        write_runtime_ledger(&value, &ledger_path, None);
        value.steps[0]
            .runtime_link
            .as_mut()
            .unwrap()
            .memory_decision
            .as_mut()
            .unwrap()
            .proposal_id = Uuid::from_u128(998);

        assert!(matches!(
            evaluator.evaluate_with_ledger(&value, &ledger_path),
            Err(EvaluatorError::RuntimeLedgerFieldMismatch {
                field,
                sample_id,
                ..
            }) if field == "details.proposal_id" && sample_id == "sample-0"
        ));
    }

    #[test]
    fn runtime_linked_evaluator_validates_memory_decision_fields() {
        let evaluator = runtime_evaluator();
        let mut value = linked_trace(&evaluator);
        value.steps[0]
            .runtime_link
            .as_mut()
            .unwrap()
            .memory_decision
            .as_mut()
            .unwrap()
            .decision_kind = "unknown".into();

        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::InvalidRuntimeMemoryDecision { sample_id, field })
                if sample_id == "sample-0" && field == "decision_kind"
        ));
    }

    #[test]
    fn appends_metrics_idempotently_to_a_trace() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let evaluator = evaluator();
        let value = trace(evaluator.metric_specs());
        let mut writer = JsonlTraceWriter::open(&path, "run-1").unwrap();
        writer.append_run(&value.run).unwrap();
        for step in &value.steps {
            writer.append_step(step).unwrap();
        }
        writer.finalize(200).unwrap();
        drop(writer);

        let report = evaluator.evaluate(&value).unwrap();
        report.append_to_trace(&path).unwrap();
        report.append_to_trace(&path).unwrap();
        assert_eq!(read_run(&path, "run-1").unwrap().metrics.len(), 4);
    }

    #[test]
    fn rejects_manifest_and_hit_contract_mismatches() {
        let evaluator = evaluator();
        let mut value = trace(evaluator.metric_specs());
        value.run.manifest.dataset_revision = "dataset-2".into();
        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::ManifestMismatch)
        ));

        let mut value = trace(evaluator.metric_specs());
        value.steps[0].observed_hits[0].rank = 0;
        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::InvalidTrace(_))
        ));

        let mut value = trace(evaluator.metric_specs());
        value.steps[0].observed_hits[0].rank = 2;
        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::InvalidTrace(message)) if message.contains("observed hit")
        ));
    }

    #[test]
    fn rejects_empty_or_duplicate_multimodal_evidence_ids() {
        let evaluator = evaluator();
        let mut value = trace(evaluator.metric_specs());
        value.steps[0].observed_hits[0].evidence_ids = vec!["artifact-video-1".into()];
        assert!(evaluator.evaluate(&value).is_ok());

        value.steps[0].observed_hits[0].evidence_ids =
            vec!["artifact-video-1".into(), "artifact-video-1".into()];
        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::InvalidTrace(message)) if message.contains("evidence id")
        ));

        value.steps[0].observed_hits[0].evidence_ids = vec![" ".into()];
        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::InvalidTrace(message)) if message.contains("evidence id")
        ));
    }

    #[test]
    fn allows_shared_artifacts_at_different_locations_across_hits() {
        let evaluator = evaluator();
        let mut value = trace(evaluator.metric_specs());
        value.steps[0].observed_hits.push(ObservedHit {
            record_id: "memory-9".into(),
            source_record_ids: Vec::new(),
            version: 1,
            rank: 2,
            score: Some(0.8),
            evidence_ids: vec!["artifact:video-1#sha256:frame-20".into()],
        });
        value.steps[0].observed_hits[0].evidence_ids =
            vec!["artifact:video-1#sha256:frame-10".into()];

        assert!(evaluator.evaluate(&value).is_ok());
    }

    #[test]
    fn rejects_zero_observed_hit_versions() {
        let evaluator = evaluator();
        let mut value = trace(evaluator.metric_specs());
        value.steps[0].observed_hits[0].version = 0;

        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::InvalidTrace(message)) if message.contains("observed hit")
        ));
    }

    #[test]
    fn requires_every_reference_sample_in_a_completed_trace() {
        let evaluator = evaluator();
        let mut value = trace(evaluator.metric_specs());
        value.steps.truncate(1);
        value.steps[0].terminal = true;
        value.steps[0].terminal_reason = Some(TerminalReason::Success);

        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::MissingSample(sample_id)) if sample_id == "sample-1"
        ));
    }

    #[test]
    fn completed_trace_requires_a_terminal_reason() {
        let evaluator = evaluator();
        let mut value = trace(evaluator.metric_specs());
        value.steps[1].terminal_reason = None;

        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::InvalidTrace(message)) if message.contains("terminal step")
        ));
    }

    #[test]
    fn repeated_evaluation_of_the_same_trace_is_deterministic() {
        let evaluator = evaluator();
        let value = trace(evaluator.metric_specs());

        assert_eq!(
            evaluator.evaluate(&value).unwrap(),
            evaluator.evaluate(&value).unwrap()
        );
    }

    #[test]
    fn reference_catalog_digest_is_independent_of_input_order() {
        let manifest = BenchmarkManifest::new(
            "retrieval",
            "bench/test",
            MethodManifest {
                method_id: "projection".into(),
                version: "1".into(),
                config: BTreeMap::new(),
            },
            "dataset-1",
            Some(100),
        );
        let mut references = references();
        let left = ReferenceCatalog::for_benchmark(&manifest, references.clone()).unwrap();
        references.reverse();
        let right = ReferenceCatalog::for_benchmark(&manifest, references).unwrap();

        assert_eq!(left.digest().unwrap(), right.digest().unwrap());
    }

    #[test]
    fn reference_catalog_sidecar_round_trips_and_checks_digest() {
        let mut manifest = BenchmarkManifest::new(
            "retrieval",
            "bench/test",
            MethodManifest {
                method_id: "projection".into(),
                version: "1".into(),
                config: BTreeMap::new(),
            },
            "dataset-1",
            Some(100),
        );
        manifest.artifact_validation_policy = ArtifactValidationPolicy::RemoteService;
        let catalog = ReferenceCatalog::for_benchmark(&manifest, references()).unwrap();
        let directory = tempdir().unwrap();
        let path = directory.path().join("references.json");

        catalog.write_json(&path).unwrap();
        let loaded = ReferenceCatalog::read_json(&path).unwrap();
        assert_eq!(loaded, catalog);
        assert_eq!(
            loaded.manifest.artifact_validation_policy,
            ArtifactValidationPolicy::RemoteService
        );
        assert_eq!(
            ReferenceCatalog::read_json_with_digest(&path, &catalog.digest().unwrap()).unwrap(),
            catalog
        );
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn legacy_reference_sidecar_without_validation_policy_keeps_digest() {
        let mut encoded_manifest = serde_json::to_value(BenchmarkManifest::new(
            "retrieval",
            "bench/test",
            MethodManifest {
                method_id: "projection".into(),
                version: "1".into(),
                config: BTreeMap::new(),
            },
            "dataset-1",
            Some(100),
        ))
        .unwrap();
        encoded_manifest
            .as_object_mut()
            .unwrap()
            .remove("artifact_validation_policy");
        let manifest: BenchmarkManifest = serde_json::from_value(encoded_manifest).unwrap();
        let catalog = ReferenceCatalog::for_benchmark(&manifest, references()).unwrap();
        let encoded = serde_json::to_value(&catalog).unwrap();
        assert!(encoded["manifest"]
            .get("artifact_validation_policy")
            .is_none());

        let directory = tempdir().unwrap();
        let path = directory.path().join("references.json");
        catalog.write_json(&path).unwrap();
        assert_eq!(ReferenceCatalog::read_json(&path).unwrap(), catalog);
    }

    #[test]
    fn reference_catalog_sidecar_rejects_corrupt_json() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("references.json");
        fs::write(&path, b"{not-json").unwrap();

        assert!(matches!(
            ReferenceCatalog::read_json(&path),
            Err(EvaluatorError::InvalidReferenceSidecar(message))
                if message.contains("JSON decode failed")
        ));
    }

    #[test]
    fn reference_catalog_sidecar_rejects_digest_and_identity_mismatches() {
        let manifest = BenchmarkManifest::new(
            "retrieval",
            "bench/test",
            MethodManifest {
                method_id: "projection".into(),
                version: "1".into(),
                config: BTreeMap::new(),
            },
            "dataset-1",
            Some(100),
        );
        let catalog = ReferenceCatalog::for_benchmark(&manifest, references()).unwrap();
        let directory = tempdir().unwrap();
        let path = directory.path().join("references.json");
        catalog.write_json(&path).unwrap();

        assert!(matches!(
            ReferenceCatalog::read_json_with_digest(&path, "sha256:wrong"),
            Err(EvaluatorError::ReferenceSidecarDigestMismatch { expected, .. })
                if expected == "sha256:wrong"
        ));

        let mut wrong_manifest = ReferenceCatalogManifest::from_benchmark(&manifest);
        wrong_manifest.benchmark_id = "other-benchmark".into();
        assert!(matches!(
            ReferenceCatalog::read_json_for_manifest(&path, &wrong_manifest),
            Err(EvaluatorError::ReferenceSidecarIdentityMismatch { .. })
        ));

        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["digest"] = serde_json::Value::String("sha256:tampered".into());
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(matches!(
            ReferenceCatalog::read_json(&path),
            Err(EvaluatorError::ReferenceSidecarDigestMismatch { expected, .. })
                if expected == "sha256:tampered"
        ));
    }

    #[test]
    fn reference_catalog_sidecar_revalidates_empty_and_duplicate_samples() {
        let manifest = BenchmarkManifest::new(
            "retrieval",
            "bench/test",
            MethodManifest {
                method_id: "projection".into(),
                version: "1".into(),
                config: BTreeMap::new(),
            },
            "dataset-1",
            Some(100),
        );
        let catalog = ReferenceCatalog::for_benchmark(&manifest, references()).unwrap();
        let directory = tempdir().unwrap();
        let path = directory.path().join("references.json");
        catalog.write_json(&path).unwrap();

        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["samples"] = serde_json::json!([]);
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(matches!(
            ReferenceCatalog::read_json(&path),
            Err(EvaluatorError::InvalidReference(message))
                if message.contains("at least one sample")
        ));

        catalog.write_json(&path).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let first = value["samples"][0].clone();
        value["samples"].as_array_mut().unwrap().push(first);
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(matches!(
            ReferenceCatalog::read_json(&path),
            Err(EvaluatorError::InvalidReference(message))
                if message.contains("appears more than once")
        ));
    }

    #[test]
    fn preflights_metric_conflicts_before_appending_anything() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let evaluator = evaluator();
        let value = trace(evaluator.metric_specs());
        let report = evaluator.evaluate(&value).unwrap();

        let mut writer = JsonlTraceWriter::open(&path, "run-1").unwrap();
        writer.append_run(&value.run).unwrap();
        for step in &value.steps {
            writer.append_step(step).unwrap();
        }
        writer.finalize(200).unwrap();
        writer
            .append_metric(&MetricValue {
                run_id: "run-1".into(),
                name: "retrieval.recall@1".into(),
                value: 99.0,
                sample_count: 2,
            })
            .unwrap();
        drop(writer);

        assert!(matches!(
            report.append_to_trace(&path),
            Err(EvaluatorError::MetricConflict(name)) if name == "retrieval.recall@1"
        ));
        assert_eq!(read_run(&path, "run-1").unwrap().metrics.len(), 1);
    }

    #[test]
    fn evaluator_rejects_malformed_ingest_even_without_trace_store() {
        let evaluator = evaluator();
        let mut value = trace(evaluator.metric_specs());
        value.steps[0].ingest = Some(IngestTrace {
            request_digest: "sha256:tampered".into(),
            item_ids: vec![Uuid::from_u128(1).to_string()],
            relation_ids: Vec::new(),
            deduplicated: false,
            snapshot_id: None,
            completed_at_ms: Some(200),
            error: None,
            telemetry: None,
        });

        assert!(matches!(
            evaluator.evaluate(&value),
            Err(EvaluatorError::Trace(TraceStoreError::InvalidStep(message)))
                if message.contains("request_digest")
        ));
    }
}
