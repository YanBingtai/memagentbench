//! Aggregation of independent benchmark runs.
//!
//! A LongMemEval corpus is executed as one isolated run per question. This
//! module combines those runs only when their immutable benchmark identity is
//! the same, and weights each metric by the number of samples it represents.

use std::collections::{BTreeMap, BTreeSet};

use pi_agent_benchmark::{
    ArtifactValidationPolicy, BenchmarkLifecycle, DatasetManifest, MethodManifest, MetricSpec,
    MetricValue, RunTrace,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const AGGREGATE_SCHEMA_VERSION: u32 = 2;
pub const COMPARISON_SCHEMA_VERSION: u32 = 1;

/// One metric after combining a set of compatible runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateMetric {
    pub name: String,
    pub value: f64,
    pub sample_count: u64,
    pub run_count: u64,
}

/// A reproducible summary for one method/configuration over several runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateReport {
    pub schema_version: u32,
    pub benchmark_id: String,
    pub namespace: String,
    pub method: MethodManifest,
    pub dataset_revision: String,
    pub dataset_digest: Option<String>,
    pub config_digest: Option<String>,
    pub run_count: u64,
    pub metrics: Vec<AggregateMetric>,
}

/// A fair comparison of several method/configuration aggregates over one
/// immutable benchmark identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub schema_version: u32,
    pub benchmark_id: String,
    pub namespace: String,
    pub dataset_revision: String,
    pub dataset_digest: Option<String>,
    /// Common reference-sidecar digest when every run uses the same sidecar.
    /// A mixed value is valid for run-scoped datasets and is represented as
    /// `None`; each input trace retains its own digest for auditability.
    pub reference_digest: Option<String>,
    pub lifecycle: BenchmarkLifecycle,
    pub artifact_validation_policy: ArtifactValidationPolicy,
    pub visibility_cutoff_ms: Option<u64>,
    pub config: BTreeMap<String, String>,
    pub methods: Vec<AggregateReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SharedIdentity {
    schema_version: u32,
    benchmark_id: String,
    namespace: String,
    dataset_revision: String,
    dataset_digest: Option<String>,
    dataset: Option<DatasetManifest>,
    lifecycle: BenchmarkLifecycle,
    artifact_validation_policy: ArtifactValidationPolicy,
    visibility_cutoff_ms: Option<u64>,
    config: BTreeMap<String, String>,
    metrics: Vec<MetricSpec>,
}

fn shared_identity(trace: &RunTrace) -> SharedIdentity {
    let manifest = &trace.run.manifest;
    SharedIdentity {
        schema_version: manifest.schema_version,
        benchmark_id: manifest.benchmark_id.clone(),
        namespace: manifest.namespace.clone(),
        dataset_revision: manifest.dataset_revision.clone(),
        dataset_digest: manifest.dataset_digest.clone(),
        dataset: manifest.dataset.clone(),
        lifecycle: manifest.lifecycle,
        artifact_validation_policy: manifest.artifact_validation_policy,
        visibility_cutoff_ms: manifest.visibility_cutoff_ms,
        config: manifest.config.clone(),
        metrics: manifest.metrics.clone(),
    }
}

/// Combines complete runs with one immutable benchmark identity.
pub fn aggregate_traces<'a>(
    traces: impl IntoIterator<Item = &'a RunTrace>,
) -> Result<AggregateReport, AggregateError> {
    let mut traces = traces.into_iter();
    let first = traces.next().ok_or(AggregateError::EmptyInput)?;
    validate_trace(first)?;

    let identity = (
        first.run.manifest.benchmark_id.clone(),
        first.run.manifest.namespace.clone(),
        first.run.manifest.method.clone(),
        first.run.manifest.dataset_revision.clone(),
        first.run.manifest.dataset_digest.clone(),
        first.run.manifest.config_digest.clone(),
        first.run.manifest.metrics.clone(),
    );
    let shared = shared_identity(first);
    let declared_names = identity
        .6
        .iter()
        .map(|metric| metric.name.clone())
        .collect::<BTreeSet<_>>();
    if declared_names.len() != identity.6.len() {
        return Err(AggregateError::DuplicateMetricDeclaration);
    }

    let mut run_ids = BTreeSet::new();
    let mut metrics = BTreeMap::<String, MetricAccumulator>::new();
    let mut run_count = 0_u64;

    for trace in std::iter::once(first).chain(traces) {
        validate_trace(trace)?;
        if !run_ids.insert(trace.run.run_id.clone()) {
            return Err(AggregateError::DuplicateRun(trace.run.run_id.clone()));
        }
        let current = (
            trace.run.manifest.benchmark_id.clone(),
            trace.run.manifest.namespace.clone(),
            trace.run.manifest.method.clone(),
            trace.run.manifest.dataset_revision.clone(),
            trace.run.manifest.dataset_digest.clone(),
            trace.run.manifest.config_digest.clone(),
            trace.run.manifest.metrics.clone(),
        );
        if current != identity {
            return Err(AggregateError::IdentityMismatch(trace.run.run_id.clone()));
        }
        if shared_identity(trace) != shared {
            return Err(AggregateError::IdentityMismatch(trace.run.run_id.clone()));
        }
        run_count = run_count
            .checked_add(1)
            .ok_or(AggregateError::CountOverflow)?;

        let mut names = BTreeSet::new();
        for metric in &trace.metrics {
            if !names.insert(metric.name.clone()) {
                return Err(AggregateError::DuplicateMetric {
                    run_id: trace.run.run_id.clone(),
                    name: metric.name.clone(),
                });
            }
            if !declared_names.contains(&metric.name) {
                return Err(AggregateError::UndeclaredMetric {
                    run_id: trace.run.run_id.clone(),
                    name: metric.name.clone(),
                });
            }
            let accumulator = metrics.entry(metric.name.clone()).or_default();
            accumulator.add(metric, &trace.run.run_id)?;
        }
        if names != declared_names {
            return Err(AggregateError::MetricCoverage(trace.run.run_id.clone()));
        }
    }

    Ok(AggregateReport {
        schema_version: AGGREGATE_SCHEMA_VERSION,
        benchmark_id: identity.0,
        namespace: identity.1,
        method: identity.2,
        dataset_revision: identity.3,
        dataset_digest: identity.4,
        config_digest: identity.5,
        run_count,
        metrics: metrics
            .into_iter()
            .map(|(name, accumulator)| AggregateMetric {
                name,
                value: if accumulator.sample_count == 0 {
                    0.0
                } else {
                    accumulator.weighted_sum / accumulator.sample_count as f64
                },
                sample_count: accumulator.sample_count,
                run_count: accumulator.run_count,
            })
            .collect(),
    })
}

/// Compares several memory methods while aggregating repeated runs of the
/// same method independently.
pub fn compare_traces<'a>(
    traces: impl IntoIterator<Item = &'a RunTrace>,
) -> Result<ComparisonReport, AggregateError> {
    let traces = traces.into_iter().collect::<Vec<_>>();
    let first = traces.first().ok_or(AggregateError::EmptyInput)?;
    validate_trace(first)?;
    let identity = shared_identity(first);
    let reference_digests = traces
        .iter()
        .map(|trace| trace.run.manifest.reference_digest.clone())
        .collect::<BTreeSet<_>>();
    let common_reference_digest = (reference_digests.len() == 1)
        .then(|| reference_digests.into_iter().next().flatten())
        .flatten();
    let mut run_ids = BTreeSet::new();
    let mut groups: Vec<Vec<&RunTrace>> = Vec::new();

    for trace in traces {
        validate_trace(trace)?;
        if !run_ids.insert(trace.run.run_id.clone()) {
            return Err(AggregateError::DuplicateRun(trace.run.run_id.clone()));
        }
        if shared_identity(trace) != identity {
            return Err(AggregateError::IdentityMismatch(trace.run.run_id.clone()));
        }
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group[0].run.manifest.method == trace.run.manifest.method)
        {
            group.push(trace);
        } else {
            groups.push(vec![trace]);
        }
    }

    let expected_samples = sample_coverage(&groups[0])?;
    for group in groups.iter().skip(1) {
        let actual_samples = sample_coverage(group)?;
        if actual_samples.keys().ne(expected_samples.keys()) {
            return Err(AggregateError::SampleCoverageMismatch(
                group[0].run.manifest.method.method_id.clone(),
            ));
        }
        for sample_id in expected_samples.keys() {
            let expected = &expected_samples[sample_id];
            let actual = &actual_samples[sample_id];
            if expected.1 != actual.1 {
                return Err(AggregateError::SampleDigestMismatch {
                    run_id: group[0].run.run_id.clone(),
                    sample_id: sample_id.clone(),
                });
            }
            if expected.0 != actual.0 || expected.2 != actual.2 {
                return Err(AggregateError::SampleCoverageMismatch(
                    group[0].run.manifest.method.method_id.clone(),
                ));
            }
        }
    }

    let mut methods = groups
        .iter()
        .map(|group| aggregate_traces(group.iter().copied()))
        .collect::<Result<Vec<_>, _>>()?;
    methods.sort_by(|left, right| {
        left.method
            .method_id
            .cmp(&right.method.method_id)
            .then_with(|| left.method.version.cmp(&right.method.version))
            .then_with(|| left.config_digest.cmp(&right.config_digest))
    });

    Ok(ComparisonReport {
        schema_version: COMPARISON_SCHEMA_VERSION,
        benchmark_id: identity.benchmark_id,
        namespace: identity.namespace,
        dataset_revision: identity.dataset_revision,
        dataset_digest: identity.dataset_digest,
        reference_digest: common_reference_digest,
        lifecycle: identity.lifecycle,
        artifact_validation_policy: identity.artifact_validation_policy,
        visibility_cutoff_ms: identity.visibility_cutoff_ms,
        config: identity.config,
        methods,
    })
}

fn validate_trace(trace: &RunTrace) -> Result<(), AggregateError> {
    if !trace.complete {
        return Err(AggregateError::IncompleteRun(trace.run.run_id.clone()));
    }
    if trace.run.run_id.trim().is_empty() {
        return Err(AggregateError::InvalidRun(
            "run_id must be non-empty".to_owned(),
        ));
    }
    trace
        .run
        .manifest
        .verify_config_digest()
        .map_err(|error| AggregateError::InvalidRun(error.to_string()))?;
    if trace.metrics.is_empty() {
        return Err(AggregateError::MissingMetrics(trace.run.run_id.clone()));
    }
    Ok(())
}

type SampleCoverage = BTreeMap<String, (u64, String, Option<String>)>;

fn sample_coverage(group: &[&RunTrace]) -> Result<SampleCoverage, AggregateError> {
    let mut coverage = BTreeMap::new();
    for trace in group {
        if trace.steps.is_empty() {
            return Err(AggregateError::InvalidRun(format!(
                "run {:?} has no sample steps",
                trace.run.run_id
            )));
        }
        let mut run_samples = BTreeSet::new();
        for step in &trace.steps {
            if step.sample_id.trim().is_empty() {
                return Err(AggregateError::InvalidRun(format!(
                    "run {:?} contains a step with an empty sample_id",
                    trace.run.run_id
                )));
            }
            if !run_samples.insert(step.sample_id.clone()) {
                return Err(AggregateError::DuplicateSample {
                    run_id: trace.run.run_id.clone(),
                    sample_id: step.sample_id.clone(),
                });
            }
            let sample_digest =
                step.sample_digest
                    .clone()
                    .ok_or_else(|| AggregateError::SampleDigestMissing {
                        run_id: trace.run.run_id.clone(),
                        sample_id: step.sample_id.clone(),
                    })?;
            if !is_sha256_digest(&sample_digest) {
                return Err(AggregateError::InvalidRun(format!(
                    "run {:?} contains an invalid sample digest for {:?}",
                    trace.run.run_id, step.sample_id
                )));
            }
            let digest = trace.run.manifest.reference_digest.clone();
            let entry = coverage.entry(step.sample_id.clone()).or_insert((
                0_u64,
                sample_digest.clone(),
                digest.clone(),
            ));
            if entry.1 != sample_digest {
                return Err(AggregateError::SampleDigestMismatch {
                    run_id: trace.run.run_id.clone(),
                    sample_id: step.sample_id.clone(),
                });
            }
            if entry.2 != digest {
                return Err(AggregateError::SampleCoverageMismatch(
                    trace.run.manifest.method.method_id.clone(),
                ));
            }
            entry.0 = entry
                .0
                .checked_add(1)
                .ok_or(AggregateError::CountOverflow)?;
        }
    }
    Ok(coverage)
}

fn is_sha256_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64 && hex.chars().all(|character| character.is_ascii_hexdigit())
}

#[derive(Default)]
struct MetricAccumulator {
    weighted_sum: f64,
    sample_count: u64,
    run_count: u64,
}

impl MetricAccumulator {
    fn add(&mut self, metric: &MetricValue, run_id: &str) -> Result<(), AggregateError> {
        if metric.run_id != run_id || !metric.value.is_finite() {
            return Err(AggregateError::InvalidMetric {
                run_id: run_id.to_owned(),
                name: metric.name.clone(),
            });
        }
        if metric.sample_count == 0 {
            if metric.value != 0.0 {
                return Err(AggregateError::InvalidMetric {
                    run_id: run_id.to_owned(),
                    name: metric.name.clone(),
                });
            }
            self.run_count = self
                .run_count
                .checked_add(1)
                .ok_or(AggregateError::CountOverflow)?;
            return Ok(());
        }
        let contribution = metric.value * metric.sample_count as f64;
        let weighted_sum = self.weighted_sum + contribution;
        if !weighted_sum.is_finite() {
            return Err(AggregateError::InvalidMetric {
                run_id: run_id.to_owned(),
                name: metric.name.clone(),
            });
        }
        self.weighted_sum = weighted_sum;
        self.sample_count = self
            .sample_count
            .checked_add(metric.sample_count)
            .ok_or(AggregateError::CountOverflow)?;
        self.run_count = self
            .run_count
            .checked_add(1)
            .ok_or(AggregateError::CountOverflow)?;
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AggregateError {
    #[error("cannot aggregate an empty run set")]
    EmptyInput,
    #[error("run {0:?} is incomplete")]
    IncompleteRun(String),
    #[error("run {0:?} has no metrics")]
    MissingMetrics(String),
    #[error("run identity differs from the first run: {0:?}")]
    IdentityMismatch(String),
    #[error("run IDs must be unique: {0:?}")]
    DuplicateRun(String),
    #[error("run {run_id:?} contains duplicate metric {name:?}")]
    DuplicateMetric { run_id: String, name: String },
    #[error("run {run_id:?} contains undeclared metric {name:?}")]
    UndeclaredMetric { run_id: String, name: String },
    #[error("run {0:?} does not contain exactly the declared metric set")]
    MetricCoverage(String),
    #[error("the manifest declares a metric more than once")]
    DuplicateMetricDeclaration,
    #[error("run {run_id:?} contains an invalid metric {name:?}")]
    InvalidMetric { run_id: String, name: String },
    #[error("run metadata is invalid: {0}")]
    InvalidRun(String),
    #[error("method sample coverage differs: {0:?}")]
    SampleCoverageMismatch(String),
    #[error("run {run_id:?} contains duplicate sample {sample_id:?}")]
    DuplicateSample { run_id: String, sample_id: String },
    #[error("run {run_id:?} is missing a digest for sample {sample_id:?}")]
    SampleDigestMissing { run_id: String, sample_id: String },
    #[error("sample {sample_id:?} has different content in run {run_id:?}")]
    SampleDigestMismatch { run_id: String, sample_id: String },
    #[error("metric sample counts overflow u64")]
    CountOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ReferenceCatalog, SampleReference};
    use pi_agent_benchmark::{
        AgentObservation, BenchmarkManifest, BenchmarkRun, MetricDirection, MetricSpec, RunTrace,
        StepTelemetry, StepTrace, TerminalReason,
    };
    use sha2::{Digest, Sha256};

    fn trace(run_id: &str, value: f64, sample_count: u64) -> RunTrace {
        let mut manifest = BenchmarkManifest::new(
            "benchmark",
            "namespace",
            MethodManifest {
                method_id: "method".to_owned(),
                version: "1".to_owned(),
                config: BTreeMap::new(),
            },
            "dataset",
            None,
        );
        manifest.metrics = vec![MetricSpec {
            name: "score".to_owned(),
            version: "1".to_owned(),
            unit: "ratio".to_owned(),
            direction: MetricDirection::HigherIsBetter,
        }];
        let run = BenchmarkRun::new(run_id, manifest, 0);
        let mut trace = RunTrace {
            steps: (0..sample_count.max(1))
                .map(|index| StepTrace {
                    run_id: run_id.to_owned(),
                    step_index: index,
                    sample_id: format!("sample-{index}"),
                    sample_digest: Some(format!(
                        "sha256:{:x}",
                        Sha256::digest(format!("sample-{index}").as_bytes())
                    )),
                    namespace: run.isolated_namespace(),
                    visibility_cutoff_ms: None,
                    observation: AgentObservation { parts: Vec::new() },
                    ingest: None,
                    query: None,
                    observed_hits: Vec::new(),
                    agent_answer: None,
                    action: None,
                    reward: None,
                    terminal_reason: (index + 1 == sample_count.max(1))
                        .then_some(TerminalReason::Success),
                    telemetry: StepTelemetry::default(),
                    errors: Vec::new(),
                    runtime_link: None,
                    terminal: index + 1 == sample_count.max(1),
                })
                .collect(),
            run,
            metrics: vec![MetricValue {
                run_id: run_id.to_owned(),
                name: "score".to_owned(),
                value,
                sample_count,
            }],
            finished_at_ms: Some(1),
            complete: true,
        };
        bind_references(&mut trace);
        trace
    }

    fn bind_references(trace: &mut RunTrace) {
        let references = ReferenceCatalog::for_benchmark(
            &trace.run.manifest,
            trace
                .steps
                .iter()
                .map(|step| SampleReference {
                    sample_id: step.sample_id.clone(),
                    relevant_record_ids: BTreeSet::from([format!("source-{}", step.sample_id)]),
                })
                .collect(),
        )
        .unwrap();
        trace.run.manifest.reference_digest = Some(references.digest().unwrap());
    }

    fn question_trace(run_id: &str, method_id: &str, question_id: &str) -> RunTrace {
        let mut value = trace(run_id, 1.0, 1);
        value.run.manifest.method.method_id = method_id.to_owned();
        value.steps[0].sample_id = question_id.to_owned();
        value.steps[0].sample_digest = Some(format!(
            "sha256:{:x}",
            Sha256::digest(question_id.as_bytes())
        ));
        bind_references(&mut value);
        value
    }

    #[test]
    fn aggregates_metrics_by_sample_count() {
        let first = trace("run-1", 1.0, 1);
        let second = trace("run-2", 0.0, 3);
        let report = aggregate_traces([&first, &second]).unwrap();
        assert_eq!(report.namespace, "namespace");
        assert_eq!(report.run_count, 2);
        assert_eq!(report.metrics[0].sample_count, 4);
        assert_eq!(report.metrics[0].run_count, 2);
        assert_eq!(report.metrics[0].value, 0.25);
    }

    #[test]
    fn aggregates_non_applicable_zero_sample_metrics_without_contributing() {
        let first = trace("run-1", 1.0, 1);
        let mut second = trace("run-2", 0.0, 1);
        second.metrics[0].value = 0.0;
        second.metrics[0].sample_count = 0;

        let report = aggregate_traces([&first, &second]).unwrap();
        assert_eq!(report.metrics[0].value, 1.0);
        assert_eq!(report.metrics[0].sample_count, 1);
        assert_eq!(report.metrics[0].run_count, 2);
    }

    #[test]
    fn rejects_different_configurations() {
        let first = trace("run-1", 1.0, 1);
        let mut second = trace("run-2", 0.0, 1);
        second.run.manifest.method.version = "2".to_owned();
        let error = aggregate_traces([&first, &second]).unwrap_err();
        assert_eq!(error, AggregateError::IdentityMismatch("run-2".to_owned()));
    }

    #[test]
    fn rejects_different_namespaces() {
        let first = trace("run-1", 1.0, 1);
        let mut second = trace("run-2", 0.0, 1);
        second.run.manifest.namespace = "another-namespace".to_owned();
        let error = aggregate_traces([&first, &second]).unwrap_err();
        assert_eq!(error, AggregateError::IdentityMismatch("run-2".to_owned()));
    }

    #[test]
    fn compares_method_groups_with_shared_benchmark_identity() {
        let first = trace("run-1", 1.0, 1);
        let mut second = trace("run-2", 0.5, 1);
        second.run.manifest.method.method_id = "method-b".to_owned();

        let report = compare_traces([&first, &second]).unwrap();

        assert_eq!(report.schema_version, COMPARISON_SCHEMA_VERSION);
        assert_eq!(report.namespace, "namespace");
        assert_eq!(report.methods.len(), 2);
        assert_eq!(report.methods[0].method.method_id, "method");
        assert_eq!(report.methods[1].method.method_id, "method-b");
    }

    #[test]
    fn comparison_weights_repeated_runs_inside_each_method_group() {
        let first = trace("run-1", 1.0, 1);
        let mut second = trace("run-2", 0.0, 1);
        second.metrics[0].sample_count = 3;
        let mut third = trace("run-3", 0.5, 1);
        third.run.manifest.method.method_id = "method-b".to_owned();
        let mut fourth = trace("run-4", 0.5, 1);
        fourth.run.manifest.method.method_id = "method-b".to_owned();
        fourth.metrics[0].sample_count = 3;

        let report = compare_traces([&first, &second, &third, &fourth]).unwrap();

        assert_eq!(report.methods[0].metrics[0].value, 0.25);
        assert_eq!(report.methods[0].metrics[0].sample_count, 4);
        assert_eq!(report.methods[1].metrics[0].value, 0.5);
    }

    #[test]
    fn comparison_rejects_a_shared_identity_mismatch() {
        let first = trace("run-1", 1.0, 1);
        let mut second = trace("run-2", 0.5, 1);
        second.run.manifest.dataset_revision = "dataset-v2".to_owned();
        let error = compare_traces([&first, &second]).unwrap_err();
        assert_eq!(error, AggregateError::IdentityMismatch("run-2".to_owned()));
    }

    #[test]
    fn comparison_rejects_a_tampered_configuration_seal() {
        let mut value = trace("run-1", 1.0, 1);
        value.run.manifest.config.insert("seed".into(), "0".into());
        value.run.manifest.config_digest = Some("sha256:tampered".into());

        let error = compare_traces([&value]).unwrap_err();

        assert!(matches!(
            error,
            AggregateError::InvalidRun(message) if message.contains("config_digest")
        ));
    }

    #[test]
    fn comparison_allows_run_scoped_reference_digests() {
        let first = question_trace("run-1", "method-a", "q-1");
        let second = question_trace("run-2", "method-a", "q-2");
        let third = question_trace("run-3", "method-b", "q-1");
        let fourth = question_trace("run-4", "method-b", "q-2");
        assert_ne!(
            first.run.manifest.reference_digest,
            second.run.manifest.reference_digest
        );

        let report = compare_traces([&fourth, &first, &third, &second]).unwrap();

        assert_eq!(report.methods.len(), 2);
        assert!(report.methods.iter().all(|method| method.run_count == 2));
        assert_eq!(report.reference_digest, None);
    }

    #[test]
    fn comparison_rejects_different_question_sets() {
        let first = question_trace("run-1", "method-a", "q-1");
        let second = question_trace("run-2", "method-b", "q-2");

        assert!(compare_traces([&first, &second]).is_err());
    }

    #[test]
    fn comparison_rejects_changed_references_for_the_same_question() {
        let first = question_trace("run-1", "method-a", "q-1");
        let mut second = question_trace("run-2", "method-b", "q-1");
        let references = ReferenceCatalog::for_benchmark(
            &second.run.manifest,
            vec![SampleReference {
                sample_id: "q-1".to_owned(),
                relevant_record_ids: BTreeSet::from(["different-source".to_owned()]),
            }],
        )
        .unwrap();
        second.run.manifest.reference_digest = Some(references.digest().unwrap());

        assert!(compare_traces([&first, &second]).is_err());
    }

    #[test]
    fn comparison_rejects_missing_reference_binding() {
        let first = question_trace("run-1", "method-a", "q-1");
        let mut second = question_trace("run-2", "method-b", "q-1");
        second.run.manifest.reference_digest = None;

        assert!(compare_traces([&first, &second]).is_err());
    }

    #[test]
    fn comparison_rejects_unequal_repetition_counts() {
        let first = question_trace("run-1", "method-a", "q-1");
        let second = question_trace("run-2", "method-a", "q-1");
        let third = question_trace("run-3", "method-b", "q-1");

        assert!(compare_traces([&first, &second, &third]).is_err());
    }

    #[test]
    fn comparison_rejects_a_duplicate_sample_in_a_run() {
        let first = trace("run-1", 1.0, 2);
        let mut second = trace("run-2", 1.0, 2);
        second.run.manifest.method.method_id = "method-b".to_owned();
        second.steps[1].sample_id = "sample-0".to_owned();

        assert_eq!(
            compare_traces([&first, &second]).unwrap_err(),
            AggregateError::DuplicateSample {
                run_id: "run-2".to_owned(),
                sample_id: "sample-0".to_owned(),
            }
        );
    }

    #[test]
    fn comparison_requires_sample_evidence_in_the_trace() {
        let mut value = trace("run-1", 1.0, 1);
        value.steps.clear();

        assert!(compare_traces([&value]).is_err());
    }

    #[test]
    fn comparison_requires_sample_digests() {
        let mut value = trace("run-1", 1.0, 1);
        value.steps[0].sample_digest = None;

        assert!(matches!(
            compare_traces([&value]).unwrap_err(),
            AggregateError::SampleDigestMissing { .. }
        ));
    }

    #[test]
    fn comparison_rejects_changed_sample_content_for_the_same_id() {
        let first = question_trace("run-1", "method-a", "q-1");
        let mut second = question_trace("run-2", "method-b", "q-1");
        second.steps[0].sample_digest = Some(format!("sha256:{}", "f".repeat(64)));

        assert!(matches!(
            compare_traces([&first, &second]).unwrap_err(),
            AggregateError::SampleDigestMismatch { run_id, sample_id }
                if run_id == "run-2" && sample_id == "q-1"
        ));
    }

    #[test]
    fn rejects_missing_metric_coverage() {
        let mut first = trace("run-1", 1.0, 1);
        first.metrics.clear();
        let error = aggregate_traces([&first]).unwrap_err();
        assert_eq!(error, AggregateError::MissingMetrics("run-1".to_owned()));
    }

    #[test]
    fn rejects_metric_from_another_run() {
        let mut first = trace("run-1", 1.0, 1);
        first.metrics[0].run_id = "run-2".to_owned();
        let error = aggregate_traces([&first]).unwrap_err();
        assert_eq!(
            error,
            AggregateError::InvalidMetric {
                run_id: "run-1".to_owned(),
                name: "score".to_owned(),
            }
        );
    }
}
