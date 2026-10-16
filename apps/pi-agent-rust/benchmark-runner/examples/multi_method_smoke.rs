//! Run the same continual dataset through multiple independent memory methods.
//!
//! The example uses two separately configured JSONL projections and a
//! no-memory baseline so the comparison boundary is visible without requiring
//! a model service. A real adapter can replace either method while the dataset,
//! runner, trace schema, and evaluator remain unchanged.
//!
//! Set `PI_AGENT_REMOTE_MEMORY_URLS` to a comma-separated list to add several
//! remote providers to the same run. `PI_AGENT_REMOTE_MEMORY_URL` remains
//! supported for one provider. Set `PI_AGENT_BENCHMARK_OUTPUT_DIR` to retain
//! traces and the comparison report after the process exits.
//! Set `PI_AGENT_BENCHMARK_DATASET` and `PI_AGENT_BENCHMARK_REFERENCE` to run
//! a real continual JSONL dataset with its validated reference sidecar.
//! Set `PI_AGENT_BENCHMARK_METHODS` to a comma-separated list such as
//! `none,projection:bm25,remote:http://127.0.0.1:9000/memory` to select method
//! instances explicitly. The legacy remote URL variables are used only when
//! this list is absent.
//! `PI_AGENT_BENCHMARK_PROFILE` defaults to `strict`; set it to `adapted` to
//! evaluate methods that cannot enforce the benchmark visibility cutoff.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use pi_agent_benchmark::{
    read_run, AgentObservation, ArtifactValidationPolicy, BenchmarkLifecycle, BenchmarkManifest,
    BenchmarkRun, BenchmarkRunContext, DatasetManifest, MethodManifest, MetricValue,
    ObservationPart, TerminalReason,
};
use pi_agent_benchmark_evaluator::{
    ContinualEvaluator, ReferenceCatalog, ReferenceCatalogManifest, ReferenceIdentity,
};
use pi_agent_benchmark_runner::{
    AgentFuture, AgentStepExecutor, AgentStepInput, AgentStepOutput, ArtifactValidationError,
    ArtifactValidationFuture, ArtifactValidationInput, ArtifactValidator, ContinualDatasetSample,
    ContinualEpisodeRunner, JsonlContinualDataset, NoMemoryMethod,
};
use pi_agent_memory::{
    ContentRef, IngestRequest, JsonlProjectionStore, MemoryBudget, MemoryItem, MemoryKind,
    MemoryMethod, MemoryQuery, MemoryRegistry, MemoryScope, Modality, OperationContext,
    ProjectionMemoryMethod,
};
use pi_agent_memory_remote::{ArtifactValidationReference, RemoteMemoryMethod};
use tempfile::{tempdir, TempDir};
use uuid::Uuid;

struct DeterministicAgent;

struct RemoteArtifactValidator {
    provider: Arc<RemoteMemoryMethod>,
}

impl RemoteArtifactValidator {
    fn new(provider: Arc<RemoteMemoryMethod>) -> Self {
        Self { provider }
    }
}

impl ArtifactValidator for RemoteArtifactValidator {
    fn validate<'a>(
        &'a mut self,
        input: ArtifactValidationInput<'a>,
        context: OperationContext,
    ) -> ArtifactValidationFuture<'a> {
        let provider = Arc::clone(&self.provider);
        let references = artifact_references(input);
        Box::pin(async move {
            if references.is_empty() {
                return Ok(());
            }
            provider
                .preflight_artifacts(references, context)
                .await
                .map(|_| ())
                .map_err(|error| ArtifactValidationError::new(error.to_string()))
        })
    }
}

fn artifact_references(input: ArtifactValidationInput<'_>) -> Vec<ArtifactValidationReference> {
    let mut references = Vec::new();
    if let Some(ingest) = input.ingest {
        references.extend(
            ingest
                .items
                .iter()
                .filter_map(|item| blob_reference(&item.content)),
        );
    }
    if let Some(query_input) = input.query.input.as_ref() {
        if let Some(reference) = blob_reference(query_input) {
            references.push(reference);
        }
    }
    references
}

fn blob_reference(content: &ContentRef) -> Option<ArtifactValidationReference> {
    match content {
        ContentRef::Blob {
            uri,
            media_type,
            sha256,
            size_bytes,
        } => Some(ArtifactValidationReference::blob(
            uri.clone(),
            media_type.clone(),
            sha256.clone(),
            *size_bytes,
        )),
        ContentRef::Text { .. } | ContentRef::Json { .. } => None,
    }
}

impl AgentStepExecutor for DeterministicAgent {
    fn execute<'a>(&'a mut self, input: AgentStepInput) -> AgentFuture<'a> {
        let answer = format!(
            "method-independent sample={} retrieved_hits={}",
            input.sample_id,
            input.retrieval.hits.len()
        );
        Box::pin(async move {
            Ok(AgentStepOutput {
                answer: Some(answer),
                ..AgentStepOutput::default()
            })
        })
    }
}

#[derive(Debug, serde::Serialize)]
struct MethodResult {
    instance_id: String,
    method_id: String,
    version: String,
    /// SHA-256 seal over benchmark and method configuration for this trace.
    config_digest: String,
    capabilities_json: String,
    evaluation_profile: String,
    capability_adaptations: Vec<String>,
    trace_path: PathBuf,
    step_count: usize,
    retrieved_hits: usize,
    ingest_success_rate: f64,
    artifact_validation_policy: ArtifactValidationPolicy,
    metrics: Vec<MetricValue>,
}

#[derive(Debug, serde::Serialize)]
struct ComparisonReport {
    schema_version: u32,
    run_id: String,
    benchmark_id: String,
    namespace: String,
    dataset_revision: String,
    dataset_digest: String,
    reference_digest: String,
    modalities: Vec<Modality>,
    methods: Vec<MethodResult>,
}

struct OutputDirectory {
    path: PathBuf,
    run_id: String,
    _temporary: Option<TempDir>,
}

struct DatasetInput {
    path: PathBuf,
    reference_path: Option<PathBuf>,
    benchmark_id: String,
    namespace: String,
    revision: String,
    modalities: Vec<Modality>,
    dataset_digest: String,
    reference_digest: String,
    artifact_validation_policy: ArtifactValidationPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MethodSpec {
    NoMemory,
    Projection { method_id: String },
    Remote { url: String },
}

struct MethodRunConfig {
    method_id: String,
    file_stem: String,
    run_id_prefix: String,
    operation_seed: u128,
    artifact_validator: Option<RemoteArtifactValidator>,
    provider_config: Option<BTreeMap<String, String>>,
    memory_url: Option<String>,
    memory_timeout_secs: Option<u64>,
}

impl DatasetInput {
    fn open(output: &OutputDirectory) -> Result<Self, Box<dyn Error>> {
        let benchmark_id = env::var("PI_AGENT_BENCHMARK_ID")
            .unwrap_or_else(|_| "continual.multi-method".to_owned());
        let namespace = env::var("PI_AGENT_BENCHMARK_NAMESPACE")
            .unwrap_or_else(|_| "benchmark/multi-method".to_owned());
        let revision = env::var("PI_AGENT_BENCHMARK_DATASET_REVISION")
            .unwrap_or_else(|_| "dataset-v1".to_owned());
        if benchmark_id.trim().is_empty()
            || namespace.trim().is_empty()
            || revision.trim().is_empty()
        {
            return Err(
                "PI_AGENT_BENCHMARK_ID, namespace, and dataset revision must be non-empty".into(),
            );
        }
        let modalities = parse_modalities(
            env::var("PI_AGENT_BENCHMARK_MODALITIES")
                .unwrap_or_else(|_| "text".to_owned())
                .as_str(),
        )?;

        match env::var_os("PI_AGENT_BENCHMARK_DATASET") {
            Some(path) => {
                let reference_path = env::var_os("PI_AGENT_BENCHMARK_REFERENCE")
                    .ok_or("PI_AGENT_BENCHMARK_REFERENCE is required with a custom dataset")?;
                let path = PathBuf::from(path);
                let dataset = JsonlContinualDataset::open(&path)?;
                let reference_path = PathBuf::from(reference_path);
                let reference = ReferenceCatalog::read_json(&reference_path)?;
                Ok(Self {
                    path,
                    reference_path: Some(reference_path),
                    benchmark_id,
                    namespace,
                    revision,
                    modalities,
                    dataset_digest: dataset.content_digest()?,
                    reference_digest: reference.digest()?,
                    artifact_validation_policy: reference.manifest.artifact_validation_policy,
                })
            }
            None => {
                if env::var_os("PI_AGENT_BENCHMARK_REFERENCE").is_some() {
                    return Err(
                        "PI_AGENT_BENCHMARK_REFERENCE requires PI_AGENT_BENCHMARK_DATASET".into(),
                    );
                }
                let path = output.path.join("dataset.jsonl");
                let fixture_record_id = Uuid::from_u128(7);
                write_dataset(&path, fixture_record_id)?;
                let dataset = JsonlContinualDataset::open(&path)?;
                let reference = ReferenceCatalog::new(
                    ReferenceCatalogManifest {
                        benchmark_id: benchmark_id.clone(),
                        namespace: namespace.clone(),
                        dataset_revision: revision.clone(),
                        dataset_digest: Some(dataset.content_digest()?),
                        artifact_validation_policy: ArtifactValidationPolicy::Unchecked,
                        visibility_cutoff_ms: Some(100),
                        identity: ReferenceIdentity::ProviderRecordId,
                        allow_empty_relevance: false,
                    },
                    vec![pi_agent_benchmark_evaluator::SampleReference {
                        sample_id: "sample-0".to_owned(),
                        relevant_record_ids: std::collections::BTreeSet::from([
                            fixture_record_id.to_string()
                        ]),
                    }],
                )?;
                let reference_path = output.path.join("reference.json");
                reference.write_json(&reference_path)?;
                Ok(Self {
                    path,
                    reference_path: Some(reference_path),
                    benchmark_id,
                    namespace,
                    revision,
                    modalities,
                    dataset_digest: dataset.content_digest()?,
                    reference_digest: reference.digest()?,
                    artifact_validation_policy: ArtifactValidationPolicy::Unchecked,
                })
            }
        }
    }
}

impl OutputDirectory {
    fn open() -> Result<Self, Box<dyn Error>> {
        if let Some(path) = env::var_os("PI_AGENT_BENCHMARK_OUTPUT_DIR") {
            let root = PathBuf::from(path);
            fs::create_dir_all(&root)?;
            let run_id = Uuid::new_v4().to_string();
            let path = root.join(format!("run-{run_id}"));
            fs::create_dir(&path)?;
            return Ok(Self {
                path,
                run_id,
                _temporary: None,
            });
        }
        let temporary = tempdir()?;
        Ok(Self {
            path: temporary.path().to_path_buf(),
            run_id: Uuid::new_v4().to_string(),
            _temporary: Some(temporary),
        })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let directory = OutputDirectory::open()?;
    let dataset = DatasetInput::open(&directory)?;
    let method_specs = method_specs()?;

    let mut results = Vec::new();
    let timeout_secs = env::var("PI_AGENT_REMOTE_MEMORY_TIMEOUT_SECS")
        .unwrap_or_else(|_| "30".to_owned())
        .parse::<u64>()?;
    for (index, spec) in method_specs.into_iter().enumerate() {
        match spec {
            MethodSpec::NoMemory => {
                let method_id = "no-memory".to_owned();
                let provider = Arc::new(NoMemoryMethod);
                let mut registry = MemoryRegistry::new();
                registry.register(provider)?;
                let method = registry
                    .get(&method_id)
                    .ok_or("registered no-memory baseline disappeared before execution")?;
                results.push(
                    run_method(
                        &directory.path,
                        &dataset,
                        method,
                        MethodRunConfig {
                            method_id,
                            file_stem: "no-memory".to_owned(),
                            run_id_prefix: directory.run_id.clone(),
                            operation_seed: index as u128 + 10,
                            artifact_validator: None,
                            provider_config: None,
                            memory_url: None,
                            memory_timeout_secs: None,
                        },
                    )
                    .await?,
                );
            }
            MethodSpec::Projection { method_id } => {
                let provider = Arc::new(ProjectionMemoryMethod::new(
                    Arc::new(JsonlProjectionStore::open(
                        directory.path.join(format!("projection-{index}.jsonl")),
                    )?),
                    &method_id,
                    "1",
                )?);
                let mut registry = MemoryRegistry::new();
                registry.register(provider)?;
                let method = registry
                    .get(&method_id)
                    .ok_or("registered method disappeared before execution")?;
                results.push(
                    run_method(
                        &directory.path,
                        &dataset,
                        method,
                        MethodRunConfig {
                            method_id,
                            file_stem: format!("projection-{index}"),
                            run_id_prefix: directory.run_id.clone(),
                            operation_seed: index as u128 + 10,
                            artifact_validator: None,
                            provider_config: None,
                            memory_url: None,
                            memory_timeout_secs: None,
                        },
                    )
                    .await?,
                );
            }
            MethodSpec::Remote { url } => {
                let provider = Arc::new(
                    RemoteMemoryMethod::discover(url.clone(), Duration::from_secs(timeout_secs))
                        .await?,
                );
                let method_id = provider.manifest().method_id.clone();
                let provider_config = provider.provider_config().clone();
                let memory_url = provider.base_url().as_str().to_owned();
                let mut registry = MemoryRegistry::new();
                registry.register(Arc::clone(&provider) as Arc<dyn MemoryMethod>)?;
                let method = registry
                    .get(&method_id)
                    .ok_or("discovered remote method disappeared before execution")?;
                results.push(
                    run_method(
                        &directory.path,
                        &dataset,
                        method,
                        MethodRunConfig {
                            method_id,
                            file_stem: format!("remote-{index}"),
                            run_id_prefix: directory.run_id.clone(),
                            operation_seed: 100 + index as u128,
                            artifact_validator: Some(RemoteArtifactValidator::new(Arc::clone(
                                &provider,
                            ))),
                            provider_config: Some(provider_config),
                            memory_url: Some(memory_url),
                            memory_timeout_secs: Some(timeout_secs),
                        },
                    )
                    .await?,
                );
            }
        }
    }

    if results.is_empty() {
        return Err("benchmark has no memory method instances".into());
    }
    for result in &results {
        println!("{}", serde_json::to_string(result)?);
    }
    let method_count = results.len();
    let comparison_path = directory.path.join("comparison.json");
    let report = ComparisonReport {
        schema_version: 1,
        run_id: directory.run_id.clone(),
        benchmark_id: dataset.benchmark_id.clone(),
        namespace: dataset.namespace.clone(),
        dataset_revision: dataset.revision.clone(),
        dataset_digest: dataset.dataset_digest.clone(),
        reference_digest: dataset.reference_digest.clone(),
        modalities: dataset.modalities.clone(),
        methods: results,
    };
    fs::write(&comparison_path, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "multi_method_smoke_ok methods={} comparison={}",
        method_count,
        comparison_path.display()
    );
    Ok(())
}

fn method_specs() -> Result<Vec<MethodSpec>, Box<dyn Error>> {
    let configured = env::var("PI_AGENT_BENCHMARK_METHODS").ok();
    let has_legacy_remote = env::var_os("PI_AGENT_REMOTE_MEMORY_URLS").is_some()
        || env::var_os("PI_AGENT_REMOTE_MEMORY_URL").is_some();
    if configured.is_some() && has_legacy_remote {
        return Err(
            "set only one of PI_AGENT_BENCHMARK_METHODS and the legacy remote URL variables".into(),
        );
    }
    if let Some(raw) = configured {
        return parse_method_specs(&raw).map_err(Into::into);
    }

    let mut specs = vec![
        MethodSpec::NoMemory,
        MethodSpec::Projection {
            method_id: "projection-a".to_owned(),
        },
        MethodSpec::Projection {
            method_id: "projection-b".to_owned(),
        },
    ];
    specs.extend(
        remote_memory_urls()?
            .into_iter()
            .map(|url| MethodSpec::Remote { url }),
    );
    Ok(specs)
}

fn parse_method_specs(raw: &str) -> Result<Vec<MethodSpec>, String> {
    let mut specs = Vec::new();
    let mut seen = BTreeSet::new();
    for item in raw.split(',').map(str::trim) {
        if item.is_empty() {
            return Err("benchmark method list must not contain an empty item".to_owned());
        }
        let (kind, value) = item.split_once(':').unwrap_or((item, ""));
        let value = value.trim();
        if value.is_empty() && kind.trim() != "none" {
            return Err(format!("benchmark method {item:?} has an empty value"));
        }
        let spec = match kind.trim() {
            "none" if value.is_empty() => MethodSpec::NoMemory,
            "projection" => MethodSpec::Projection {
                method_id: value.to_owned(),
            },
            "remote" => MethodSpec::Remote {
                url: value.to_owned(),
            },
            other => {
                let message = format!(
                    "unsupported benchmark method kind {other:?}; expected none, projection, or remote"
                );
                return Err(message);
            }
        };
        if !seen.insert(item.to_owned()) {
            return Err(format!("benchmark method {item:?} is duplicated"));
        }
        specs.push(spec);
    }
    if specs.is_empty() {
        return Err("benchmark method list must contain at least one method".to_owned());
    }
    Ok(specs)
}

fn remote_memory_urls() -> Result<Vec<String>, Box<dyn Error>> {
    parse_remote_memory_urls(
        env::var("PI_AGENT_REMOTE_MEMORY_URLS").ok().as_deref(),
        env::var("PI_AGENT_REMOTE_MEMORY_URL").ok().as_deref(),
    )
    .map_err(Into::into)
}

fn parse_remote_memory_urls(
    plural: Option<&str>,
    singular: Option<&str>,
) -> Result<Vec<String>, String> {
    if plural.is_some() && singular.is_some() {
        return Err(
            "set only one of PI_AGENT_REMOTE_MEMORY_URLS and PI_AGENT_REMOTE_MEMORY_URL".into(),
        );
    }
    let Some(raw) = plural.or(singular) else {
        return Ok(Vec::new());
    };
    let urls = raw
        .split(',')
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if urls.is_empty() {
        return Err("remote memory URL list must contain at least one URL".into());
    }
    Ok(urls)
}

fn parse_modalities(raw: &str) -> Result<Vec<Modality>, String> {
    let mut modalities = Vec::new();
    for value in raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let modality = match value {
            "text" => Modality::Text,
            "image" => Modality::Image,
            "audio" => Modality::Audio,
            "video" => Modality::Video,
            "state" => Modality::State,
            "action" => Modality::Action,
            "trajectory" => Modality::Trajectory,
            "mixed" => Modality::Mixed,
            other => return Err(format!("unsupported benchmark modality: {other}")),
        };
        if !modalities.contains(&modality) {
            modalities.push(modality);
        }
    }
    if modalities.is_empty() {
        return Err("benchmark modalities must contain at least one value".to_owned());
    }
    Ok(modalities)
}

async fn run_method(
    root: &Path,
    input: &DatasetInput,
    method: Arc<dyn MemoryMethod>,
    config: MethodRunConfig,
) -> Result<MethodResult, Box<dyn Error>> {
    let MethodRunConfig {
        method_id,
        file_stem,
        run_id_prefix,
        operation_seed,
        artifact_validator,
        provider_config,
        memory_url,
        memory_timeout_secs,
    } = config;
    let trace_path = root.join(format!("{file_stem}.trace.jsonl"));
    let method_manifest = method.manifest();
    if method_manifest.method_id != method_id {
        return Err(format!(
            "method registry identity mismatch: expected {method_id}, got {}",
            method_manifest.method_id
        )
        .into());
    }
    let dataset = JsonlContinualDataset::open(&input.path)?;
    let dataset_digest = dataset.content_digest()?;
    let evaluation_profile =
        env::var("PI_AGENT_BENCHMARK_PROFILE").unwrap_or_else(|_| "strict".to_owned());
    let (visibility_cutoff_ms, capability_adaptations) = visibility_policy(
        &evaluation_profile,
        method_manifest.capabilities.supports_visibility_cutoff,
    )?;
    let capabilities_json = serde_json::to_string(&method_manifest.capabilities)?;
    let mut method_identity = MethodManifest {
        method_id: method_id.clone(),
        version: method_manifest.version.clone(),
        config: BTreeMap::new(),
    };
    method_identity
        .config
        .insert("capabilities_json".to_owned(), capabilities_json.clone());
    if method_id == "no-memory" {
        method_identity
            .config
            .insert("baseline".to_owned(), "no_memory".to_owned());
    }
    if let Some(provider_config) = provider_config.as_ref() {
        method_identity.bind_provider_config(provider_config);
    }
    if let Some(memory_url) = memory_url {
        method_identity
            .config
            .insert("memory_url".to_owned(), memory_url);
    }
    if let Some(memory_timeout_secs) = memory_timeout_secs {
        method_identity.config.insert(
            "memory_timeout_secs".to_owned(),
            memory_timeout_secs.to_string(),
        );
    }
    let mut manifest = BenchmarkManifest::new(
        &input.benchmark_id,
        &input.namespace,
        method_identity,
        &input.revision,
        visibility_cutoff_ms,
    );
    manifest.lifecycle = BenchmarkLifecycle::Continual;
    manifest
        .config
        .insert("evaluation_profile".to_owned(), evaluation_profile.clone());
    manifest.config.insert(
        "capability_adaptations".to_owned(),
        if capability_adaptations.is_empty() {
            "none".to_owned()
        } else {
            capability_adaptations.join(",")
        },
    );
    manifest.bind_dataset(DatasetManifest::new(
        &input.benchmark_id,
        &input.revision,
        dataset_digest,
        input.modalities.clone(),
    ))?;
    if artifact_validator.is_some()
        && input.modalities.iter().any(|modality| {
            matches!(
                modality,
                Modality::Image | Modality::Audio | Modality::Video
            )
        })
    {
        manifest.artifact_validation_policy = ArtifactValidationPolicy::RemoteService;
    } else {
        manifest.artifact_validation_policy = input.artifact_validation_policy;
    }
    let reference_path = input
        .reference_path
        .as_deref()
        .ok_or("benchmark reference sidecar is missing")?;
    let expected = ReferenceCatalogManifest::from_benchmark(&manifest);
    let references = ReferenceCatalog::read_json_for_manifest(reference_path, &expected)?;
    manifest.reference_digest = Some(references.digest()?);
    let evaluator = ContinualEvaluator::new(1, references)?;
    manifest.metrics = evaluator.metric_specs();
    manifest.seal_config();
    let config_digest = manifest
        .config_digest
        .clone()
        .ok_or("benchmark manifest did not seal its config digest")?;
    let artifact_validation_policy = manifest.artifact_validation_policy;
    let run = BenchmarkRun::new(
        format!("{run_id_prefix}-{file_stem}"),
        manifest,
        1_700_000_000_000 + operation_seed as i64,
    );
    let context = BenchmarkRunContext::open(&trace_path, run, Uuid::from_u128(operation_seed))?;
    let runner = ContinualEpisodeRunner::new(method, dataset, context);
    let summary = if let Some(artifact_validator) = artifact_validator {
        runner
            .with_artifact_validator(artifact_validator)
            .run_with_agent(DeterministicAgent)
            .await?
    } else {
        runner.run_with_agent(DeterministicAgent).await?
    };
    if summary.ingest_error_count != 0 || summary.retrieval_error_count != 0 {
        return Err(format!(
            "method {method_id} failed: ingest_errors={} retrieval_errors={}",
            summary.ingest_error_count, summary.retrieval_error_count
        )
        .into());
    }
    let trace = read_run(&trace_path, &summary.run_id)?;
    let report = evaluator.evaluate(&trace)?;
    report.append_to_trace(&trace_path)?;
    if trace.steps.is_empty() {
        return Err("method run produced no trace step".into());
    }
    let ingest_success_rate = report
        .metrics
        .iter()
        .find(|metric| metric.name == "continual.ingest_success_rate")
        .ok_or("continual evaluator omitted ingest_success_rate")?
        .value;
    Ok(MethodResult {
        instance_id: file_stem,
        method_id,
        version: method_manifest.version,
        config_digest,
        capabilities_json,
        evaluation_profile,
        capability_adaptations,
        trace_path,
        step_count: trace.steps.len(),
        retrieved_hits: trace
            .steps
            .iter()
            .map(|step| step.observed_hits.len())
            .sum(),
        ingest_success_rate,
        artifact_validation_policy,
        metrics: report.metrics,
    })
}

fn visibility_policy(
    evaluation_profile: &str,
    supports_visibility_cutoff: bool,
) -> Result<(Option<u64>, Vec<String>), Box<dyn Error>> {
    match evaluation_profile {
        "strict" => Ok((Some(100), Vec::new())),
        "adapted" if supports_visibility_cutoff => Ok((Some(100), Vec::new())),
        "adapted" => Ok((None, vec!["visibility_cutoff".to_owned()])),
        other => Err(format!(
            "PI_AGENT_BENCHMARK_PROFILE must be strict or adapted, got {other:?}"
        )
        .into()),
    }
}

fn write_dataset(path: &Path, record_id: Uuid) -> Result<(), Box<dyn Error>> {
    let sample = ContinualDatasetSample {
        sample: pi_agent_benchmark_runner::DatasetSample {
            sample_id: "sample-0".to_owned(),
            observation: AgentObservation {
                parts: vec![ObservationPart::Text("multi-method observation".to_owned())],
            },
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
                budget: MemoryBudget {
                    max_hits: 1,
                    ..MemoryBudget::default()
                },
            },
            session_id: None,
            episode_id: None,
            terminal: true,
            terminal_reason: Some(TerminalReason::Success),
        },
        ingest: IngestRequest {
            items: vec![MemoryItem {
                id: record_id,
                version: 1,
                kind: MemoryKind::Semantic,
                scope: MemoryScope::Shared,
                modality: Modality::Text,
                content: ContentRef::Text {
                    text: "same dataset record for every method".to_owned(),
                },
                namespace: String::new(),
                episode_id: None,
                subject: None,
                source: "multi-method-fixture".to_owned(),
                metadata: BTreeMap::new(),
                abstraction_level: 0,
                event_time: None,
                ingest_time_ms: 1,
                schema_version: 1,
            }],
            relations: Vec::new(),
            idempotency_key: "sample-0-ingest".to_owned(),
        },
    };
    fs::write(path, format!("{}\n", serde_json::to_string(&sample)?))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        parse_method_specs, parse_modalities, parse_remote_memory_urls, visibility_policy,
        MethodSpec,
    };

    #[test]
    fn parses_multiple_remote_urls_and_ignores_spacing() {
        assert_eq!(
            parse_remote_memory_urls(Some(" http://one ,http://two "), None).unwrap(),
            vec!["http://one", "http://two"]
        );
    }

    #[test]
    fn rejects_ambiguous_or_empty_remote_url_configuration() {
        assert!(parse_remote_memory_urls(Some("http://one"), Some("http://two")).is_err());
        assert!(parse_remote_memory_urls(Some(" , "), None).is_err());
        assert!(parse_remote_memory_urls(None, None).unwrap().is_empty());
    }

    #[test]
    fn parses_unique_multimodal_dataset_declarations() {
        assert_eq!(
            parse_modalities("text, video, text,trajectory").unwrap(),
            vec![
                pi_agent_memory::Modality::Text,
                pi_agent_memory::Modality::Video,
                pi_agent_memory::Modality::Trajectory,
            ]
        );
        assert!(parse_modalities("unknown").is_err());
        assert!(parse_modalities(" , ").is_err());
    }

    #[test]
    fn parses_explicit_local_and_remote_method_instances() {
        assert_eq!(
            parse_method_specs("projection:bm25, remote:http://memory:9000/api").unwrap(),
            vec![
                MethodSpec::Projection {
                    method_id: "bm25".to_owned()
                },
                MethodSpec::Remote {
                    url: "http://memory:9000/api".to_owned()
                }
            ]
        );
        assert!(parse_method_specs("projection:bm25,projection:bm25").is_err());
        assert!(parse_method_specs("unknown:method").is_err());
        assert!(parse_method_specs(" , ").is_err());
    }

    #[test]
    fn adapted_profile_records_missing_visibility_cutoff() {
        assert_eq!(
            visibility_policy("adapted", false).unwrap(),
            (None, vec!["visibility_cutoff".to_owned()])
        );
        assert_eq!(
            visibility_policy("strict", false).unwrap(),
            (Some(100), Vec::new())
        );
        assert!(visibility_policy("unknown", false).is_err());
    }
}
