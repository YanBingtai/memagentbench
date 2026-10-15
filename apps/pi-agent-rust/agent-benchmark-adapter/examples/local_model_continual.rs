//! Run an online-memory benchmark step through the real Pi agent loop.
//!
//! Unlike the retrieval-only example, this program starts with an empty JSONL
//! projection. The benchmark runner ingests the sample first, retrieves it,
//! and then gives the bounded result to the model adapter. This makes the
//! write/read ordering and their durable trace part of the same experiment.
//!
//! Set `PI_AGENT_MEMORY_URLS` to a comma-separated list to run the same model
//! turn against several remote memory methods. `PI_AGENT_MEMORY_URL` remains
//! supported for one remote method; without either variable, local projection
//! memory is used.
//!
//! `PI_AGENT_BENCHMARK_PROFILE` defaults to `strict`. Set it to `adapted` only
//! when a method lacks a benchmark capability such as visibility cutoffs; the
//! resulting trace and comparison report record that adaptation explicitly.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use pi_agent_benchmark::{
    read_run, AgentObservation, ArtifactValidationPolicy, BenchmarkManifest, BenchmarkRun,
    BenchmarkRunContext, MethodManifest, MetricValue, ObservationPart, TerminalReason,
    TraceStoreError,
};
use pi_agent_benchmark_adapter::{PiAgentExecutor, RemoteArtifactValidator};
use pi_agent_benchmark_evaluator::{
    ContinualEvaluator, ReferenceCatalog, ReferenceCatalogManifest, RuntimeLinkedEvaluator,
    SampleReference,
};
use pi_agent_benchmark_runner::{ContinualDatasetSample, ContinualEpisodeRunner, DatasetSample};
use pi_agent_memory::{
    ArtifactRef, ContentRef, IngestRequest, JsonlProjectionStore, MemoryBudget, MemoryItem,
    MemoryKind, MemoryMethod, MemoryQuery, MemoryScope, Modality, ProjectionMemoryMethod,
};
use pi_agent_memory_remote::RemoteMemoryMethod;
use pi_agent_rust::{AgentLoop, ChatOptions, MemoryHookConfig, ModelClient, ToolRegistry};
use uuid::Uuid;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:6273/v1";
const DEFAULT_MODEL: &str = "qwen3.8";
const VISIBILITY_CUTOFF_MS: u64 = 100;

#[derive(Debug, serde::Serialize)]
struct MethodResult {
    method_index: usize,
    operation_run_id: Uuid,
    run_id: String,
    method_id: String,
    version: String,
    evaluation_profile: String,
    capability_adaptations: Vec<String>,
    trace_path: PathBuf,
    runtime_ledger_path: PathBuf,
    step_count: usize,
    retrieved_hits: usize,
    metrics: Vec<MetricValue>,
}

#[derive(Debug, serde::Serialize)]
struct ComparisonReport {
    schema_version: u32,
    operation_run_id: Uuid,
    benchmark_id: String,
    model: String,
    modality: Modality,
    methods: Vec<MethodResult>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let operation_run_id = parse_operation_run_id()?;
    let methods = memory_methods()?;
    let modality = parse_modality()?;
    let total_methods = methods.len();
    let model_name = env::var("PI_AGENT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
    let mut results = Vec::with_capacity(total_methods);
    for (method_index, memory_url) in methods.into_iter().enumerate() {
        let operation_run_id = operation_run_id_for(operation_run_id, method_index)?;
        results.push(
            run_one(
                method_index,
                total_methods,
                memory_url,
                operation_run_id,
                modality,
            )
            .await?,
        );
    }
    let comparison_path = env_path(
        "PI_AGENT_COMPARISON_FILE",
        format!("./sessions/local-model-continual-{operation_run_id}-comparison.json"),
    );
    if let Some(parent) = comparison_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let report = ComparisonReport {
        schema_version: 1,
        operation_run_id,
        benchmark_id: "agent.local-model.continual".to_owned(),
        model: model_name,
        modality,
        methods: results,
    };
    fs::write(&comparison_path, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "local_model_continual_comparison_ok methods={} comparison={}",
        report.methods.len(),
        comparison_path.display()
    );
    Ok(())
}

async fn run_one(
    method_index: usize,
    total_methods: usize,
    memory_url: Option<String>,
    operation_run_id: Uuid,
    modality: Modality,
) -> Result<MethodResult, Box<dyn Error>> {
    let base_url = env::var("PI_AGENT_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned());
    let model_name = env::var("PI_AGENT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
    // The trace contract hex-encodes run IDs in the isolated namespace. Keep
    // this component short enough that a user-provided namespace still fits
    // the 128-byte storage boundary.
    let run_id = format!("c-{operation_run_id}");
    let default_stem = format!("./sessions/{run_id}");
    let memory_path = method_path(
        env_path(
            "PI_AGENT_MEMORY_FILE",
            format!("{default_stem}-memory.jsonl"),
        ),
        method_index,
        total_methods,
    );
    let trace_path = method_path(
        env_path("PI_AGENT_TRACE_FILE", format!("{default_stem}-trace.jsonl")),
        method_index,
        total_methods,
    );
    let reference_path = method_path(
        env_path(
            "PI_AGENT_REFERENCE_FILE",
            format!("{default_stem}-references.json"),
        ),
        method_index,
        total_methods,
    );
    let runtime_ledger_path = method_path(
        env_path(
            "PI_AGENT_RUNTIME_LEDGER_FILE",
            format!("{default_stem}-runtime.jsonl"),
        ),
        method_index,
        total_methods,
    );
    let namespace = env::var("PI_AGENT_MEMORY_NAMESPACE")
        .unwrap_or_else(|_| "benchmark/local-model-continual".to_owned());
    let (prompt, content, kind, episode_id, observation) = fixture(modality, operation_run_id);
    let record_id = operation_run_id;
    let memory_timeout_secs = env::var("PI_AGENT_MEMORY_TIMEOUT_SECS")
        .unwrap_or_else(|_| "30".to_owned())
        .parse::<u64>()?;

    // The method is intentionally empty here. ContinualEpisodeRunner owns the
    // write, so a successful retrieval proves the method observed this run's
    // ingest rather than a pre-seeded fixture. A remote provider is selected
    // only when explicitly configured; a failed discovery is returned to the
    // caller instead of silently falling back to the local projection.
    let (local_method, remote_method): (Option<Arc<dyn MemoryMethod>>, Option<RemoteMemoryMethod>) =
        if let Some(url) = memory_url {
            let remote =
                RemoteMemoryMethod::discover(url, Duration::from_secs(memory_timeout_secs)).await?;
            (None, Some(remote))
        } else {
            (
                Some(Arc::new(ProjectionMemoryMethod::new(
                    Arc::new(JsonlProjectionStore::open(&memory_path)?),
                    "projection",
                    "1",
                )?) as Arc<dyn MemoryMethod>),
                None,
            )
        };
    let method_manifest = match (&local_method, &remote_method) {
        (Some(method), None) => method.manifest(),
        (None, Some(method)) => method.manifest(),
        _ => unreachable!("exactly one memory method must be selected"),
    };
    let evaluation_profile =
        env::var("PI_AGENT_BENCHMARK_PROFILE").unwrap_or_else(|_| "strict".to_owned());
    let (visibility_cutoff_ms, capability_adaptations) = visibility_policy(
        &evaluation_profile,
        method_manifest.capabilities.supports_visibility_cutoff,
    )?;

    let mut method_identity = MethodManifest {
        method_id: method_manifest.method_id.clone(),
        version: method_manifest.version.clone(),
        config: BTreeMap::new(),
    };
    if let Some(remote) = &remote_method {
        method_identity.bind_provider_config(remote.provider_config());
    }
    if let Some(remote) = &remote_method {
        method_identity.config.insert(
            "memory_url".to_owned(),
            remote.base_url().as_str().to_owned(),
        );
        method_identity.config.insert(
            "memory_timeout_secs".to_owned(),
            memory_timeout_secs.to_string(),
        );
    }
    let mut manifest = BenchmarkManifest::new(
        "agent.local-model.continual",
        &namespace,
        method_identity,
        "local-continual-fixture-v1",
        visibility_cutoff_ms,
    );
    manifest
        .config
        .insert("model".to_owned(), model_name.clone());
    manifest.config.insert("prompt".to_owned(), prompt);
    manifest.config.insert(
        "seed".to_owned(),
        env::var("PI_AGENT_SEED").unwrap_or_else(|_| "0".to_owned()),
    );
    manifest
        .config
        .insert("modality".to_owned(), modality_name(modality).to_owned());
    manifest
        .config
        .insert("base_url".to_owned(), base_url.clone());
    manifest
        .config
        .insert("temperature".to_owned(), "0.0".to_owned());
    manifest
        .config
        .insert("enable_thinking".to_owned(), "true".to_owned());
    manifest
        .config
        .insert("max_steps".to_owned(), "4".to_owned());
    manifest
        .config
        .insert("request_timeout_secs".to_owned(), "120".to_owned());
    manifest.config.insert(
        "memory_injection_mode".to_owned(),
        "runtime_retrieval_proposal".to_owned(),
    );
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
    manifest.artifact_validation_policy = ArtifactValidationPolicy::Unchecked;
    if remote_method.is_some() && matches!(content, ContentRef::Blob { .. }) {
        manifest.artifact_validation_policy = ArtifactValidationPolicy::RemoteService;
    }

    let started_at_ms = unix_time_ms()?;
    let provisional_run = BenchmarkRun::new(run_id.clone(), manifest.clone(), started_at_ms);
    let runtime_namespace = provisional_run.isolated_namespace();
    let artifact_validator = remote_method
        .as_ref()
        .map(|remote| RemoteArtifactValidator::new(remote.clone()));
    let method: Arc<dyn MemoryMethod> = match (local_method, remote_method) {
        (Some(method), None) => method,
        (None, Some(method)) => Arc::new(method),
        _ => unreachable!("exactly one memory method must be selected"),
    };

    let expected_reference_manifest = ReferenceCatalogManifest::from_benchmark(&manifest);
    let references = if reference_path.exists() {
        ReferenceCatalog::read_json_for_manifest(&reference_path, &expected_reference_manifest)?
    } else {
        let references = ReferenceCatalog::for_benchmark(
            &manifest,
            vec![SampleReference {
                sample_id: "local-model-continual-0".to_owned(),
                relevant_record_ids: BTreeSet::from([record_id.to_string()]),
            }],
        )?;
        references.write_json(&reference_path)?;
        references
    };
    manifest.reference_digest = Some(references.digest()?);
    let continual_evaluator = ContinualEvaluator::new(1, references.clone())?;
    let runtime_evaluator = RuntimeLinkedEvaluator::new(1, references)?;
    manifest.metrics = continual_evaluator.metric_specs();
    manifest.seal_config();

    let run = BenchmarkRun::new(run_id, manifest, started_at_ms);
    let context = open_or_resume_context(&trace_path, run, operation_run_id)?;
    let dataset = vec![ContinualDatasetSample {
        sample: DatasetSample {
            sample_id: "local-model-continual-0".to_owned(),
            observation,
            query: MemoryQuery {
                input: None,
                namespace: String::new(),
                scope: None,
                subject: None,
                episode_id: episode_id.clone(),
                kinds: Vec::new(),
                modalities: vec![modality],
                time_range: None,
                visibility_cutoff_ms: None,
                graph: None,
                budget: MemoryBudget {
                    max_hits: 4,
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
                kind,
                scope: MemoryScope::Shared,
                modality,
                content,
                namespace: String::new(),
                episode_id,
                subject: None,
                source: "local-model-continual".to_owned(),
                metadata: BTreeMap::new(),
                abstraction_level: 0,
                event_time: None,
                ingest_time_ms: 1,
                schema_version: 1,
            }],
            relations: Vec::new(),
            idempotency_key: "local-model-continual-0-ingest".to_owned(),
        },
    }];

    let client = ModelClient::new(&base_url, Duration::from_secs(120))?;
    let agent = AgentLoop::new(
        client,
        ToolRegistry::new(),
        model_name,
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(true),
        },
        4,
    )?
    .with_runtime_ledger_file(&runtime_ledger_path);
    let memory_config = MemoryHookConfig {
        namespace: runtime_namespace,
        ..MemoryHookConfig::default()
    };

    let mut runner = ContinualEpisodeRunner::new(
        method,
        pi_agent_benchmark_runner::VecContinualDataset::new(dataset),
        context,
    );
    if let Some(artifact_validator) = artifact_validator {
        runner = runner.with_artifact_validator(artifact_validator);
    }
    let summary = runner
        .run_with_agent(PiAgentExecutor::new_with_runtime_and_retrieval_memory(
            agent,
            operation_run_id,
            memory_config,
        ))
        .await
        .map_err(|error| format!("continual benchmark run failed: {error}"))?;
    let trace = read_run(&trace_path, &summary.run_id)?;
    if summary.ingest_error_count != 0
        || summary.retrieval_error_count != 0
        || summary.agent_error_count != 0
    {
        let errors = trace
            .steps
            .iter()
            .flat_map(|step| step.errors.iter())
            .map(|error| format!("{}: {}", error.code, error.message))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(format!(
            "continual benchmark completed with ingest_errors={} retrieval_errors={} agent_errors={}: {}; inspect {}",
            summary.ingest_error_count,
            summary.retrieval_error_count,
            summary.agent_error_count,
            errors,
            trace_path.display()
        )
        .into());
    }

    let report = continual_evaluator.evaluate(&trace)?;
    report.append_to_trace(&trace_path)?;
    // The continual evaluator owns ingest metrics; the runtime evaluator adds
    // an independent audit that every model turn is linked to the ledger.
    runtime_evaluator.evaluate_with_ledger(&trace, &runtime_ledger_path)?;

    let step = trace
        .steps
        .first()
        .ok_or("continual benchmark produced no trace step")?;
    let ingest = step
        .ingest
        .as_ref()
        .ok_or("continual benchmark trace omitted ingest receipt")?;
    if ingest.item_ids != vec![record_id.to_string()] || step.observed_hits.len() != 1 {
        return Err("continual benchmark did not round-trip the ingested record".into());
    }
    let adaptations_label = if capability_adaptations.is_empty() {
        "none".to_owned()
    } else {
        capability_adaptations.join(",")
    };
    let result = MethodResult {
        method_index,
        operation_run_id,
        run_id: summary.run_id.clone(),
        method_id: method_manifest.method_id.clone(),
        version: method_manifest.version.clone(),
        evaluation_profile: evaluation_profile.clone(),
        capability_adaptations: capability_adaptations.clone(),
        trace_path: trace_path.clone(),
        runtime_ledger_path: runtime_ledger_path.clone(),
        step_count: trace.steps.len(),
        retrieved_hits: trace
            .steps
            .iter()
            .map(|step| step.observed_hits.len())
            .sum(),
        metrics: report.metrics.clone(),
    };
    println!(
        "local_model_continual_ok run_id={} method_index={} method={} profile={} adaptations={} modality={} ingested_items={} retrieved_hits={} metrics={} trace={} runtime_ledger={} answer={}",
        summary.run_id,
        method_index,
        method_manifest.method_id,
        evaluation_profile,
        adaptations_label,
        modality_name(modality),
        ingest.item_ids.len(),
        step.observed_hits.len(),
        report.metrics.len(),
        trace_path.display(),
        runtime_ledger_path.display(),
        step.agent_answer.as_deref().unwrap_or("<empty>")
    );
    Ok(result)
}

fn fixture(
    modality: Modality,
    operation_run_id: Uuid,
) -> (
    String,
    ContentRef,
    MemoryKind,
    Option<String>,
    AgentObservation,
) {
    match modality {
        Modality::Video => {
            let artifact = ArtifactRef {
                artifact_id: operation_run_id,
                uri: "cas://benchmark/local-model-continual/episode-1.mp4".to_owned(),
                media_type: "video/mp4".to_owned(),
                sha256: "sha256:local-model-continual-video".to_owned(),
                bytes: 8_192,
                codec: Some("h264".to_owned()),
                schema: Some("episode-video-v1".to_owned()),
                stream_key: Some("camera/front".to_owned()),
                storage_revision: Some("fixture-v1".to_owned()),
            };
            (
                "请根据刚写入的视频记忆说明抓取物体的关键帧。".to_owned(),
                ContentRef::Blob {
                    uri: artifact.uri,
                    media_type: artifact.media_type,
                    sha256: artifact.sha256,
                    size_bytes: artifact.bytes,
                },
                MemoryKind::Episodic,
                Some("episode-1".to_owned()),
                AgentObservation {
                    parts: vec![ObservationPart::Artifact {
                        artifact_id: artifact.artifact_id.to_string(),
                        modality: "video".to_owned(),
                    }],
                },
            )
        }
        _ => (
            "请解释刚写入的 JSONL 记忆如何避免未来数据泄漏。".to_owned(),
            ContentRef::Text {
                text: "Rust 的 JSONL 记忆投影支持按 namespace 和时间截断检索。".to_owned(),
            },
            MemoryKind::Semantic,
            None,
            AgentObservation {
                parts: vec![ObservationPart::Text(
                    "请解释刚写入的 JSONL 记忆如何避免未来数据泄漏。".to_owned(),
                )],
            },
        ),
    }
}

fn env_path(name: &str, fallback: String) -> PathBuf {
    env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(fallback))
}

fn parse_operation_run_id() -> Result<Uuid, Box<dyn Error>> {
    match env::var("PI_AGENT_OPERATION_RUN_ID") {
        Ok(value) => Ok(Uuid::parse_str(&value)?),
        Err(env::VarError::NotPresent) => Ok(Uuid::new_v4()),
        Err(error) => Err(error.into()),
    }
}

fn operation_run_id_for(base: Uuid, method_index: usize) -> Result<Uuid, Box<dyn Error>> {
    let offset = u128::try_from(method_index)?;
    let value = base
        .as_u128()
        .checked_add(offset)
        .ok_or("PI_AGENT_OPERATION_RUN_ID cannot address every configured method")?;
    Ok(Uuid::from_u128(value))
}

fn memory_methods() -> Result<Vec<Option<String>>, Box<dyn Error>> {
    parse_memory_methods(
        env::var("PI_AGENT_MEMORY_URLS").ok().as_deref(),
        env::var("PI_AGENT_MEMORY_URL").ok().as_deref(),
    )
    .map_err(Into::into)
}

fn parse_memory_methods(
    plural: Option<&str>,
    singular: Option<&str>,
) -> Result<Vec<Option<String>>, String> {
    if plural.is_some() && singular.is_some() {
        return Err("set only one of PI_AGENT_MEMORY_URLS and PI_AGENT_MEMORY_URL".to_owned());
    }
    let Some(raw) = plural.or(singular) else {
        return Ok(vec![None]);
    };
    let mut seen = BTreeSet::new();
    let mut urls = Vec::new();
    for url in raw.split(',').map(str::trim).filter(|url| !url.is_empty()) {
        if !seen.insert(url) {
            return Err(format!("memory URL {url:?} is duplicated"));
        }
        urls.push(Some(url.to_owned()));
    }
    if urls.is_empty() {
        return Err("memory URL list must contain at least one URL".to_owned());
    }
    Ok(urls)
}

fn method_path(path: PathBuf, method_index: usize, total_methods: usize) -> PathBuf {
    if total_methods == 1 {
        path
    } else {
        PathBuf::from(format!("{}.method-{method_index}", path.display()))
    }
}

fn parse_modality() -> Result<Modality, Box<dyn Error>> {
    match env::var("PI_AGENT_BENCHMARK_MODALITY")
        .unwrap_or_else(|_| "text".to_owned())
        .as_str()
    {
        "text" => Ok(Modality::Text),
        "video" => Ok(Modality::Video),
        other => {
            Err(format!("PI_AGENT_BENCHMARK_MODALITY supports text or video, got {other:?}").into())
        }
    }
}

fn modality_name(modality: Modality) -> &'static str {
    match modality {
        Modality::Text => "text",
        Modality::Video => "video",
        _ => "other",
    }
}

fn visibility_policy(
    evaluation_profile: &str,
    supports_visibility_cutoff: bool,
) -> Result<(Option<u64>, Vec<String>), Box<dyn Error>> {
    match evaluation_profile {
        "strict" => Ok((Some(VISIBILITY_CUTOFF_MS), Vec::new())),
        "adapted" if supports_visibility_cutoff => Ok((Some(VISIBILITY_CUTOFF_MS), Vec::new())),
        "adapted" => Ok((None, vec!["visibility_cutoff".to_owned()])),
        other => Err(format!(
            "PI_AGENT_BENCHMARK_PROFILE must be strict or adapted, got {other:?}"
        )
        .into()),
    }
}

fn unix_time_ms() -> Result<i64, Box<dyn Error>> {
    Ok(i64::try_from(
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
    )?)
}

fn open_or_resume_context(
    trace_path: &Path,
    expected_run: BenchmarkRun,
    operation_run_id: Uuid,
) -> Result<BenchmarkRunContext, TraceStoreError> {
    if trace_path.exists() {
        let context = BenchmarkRunContext::resume(trace_path, &expected_run.run_id)?;
        if context.run().manifest != expected_run.manifest {
            return Err(TraceStoreError::InvalidManifest(
                "existing trace does not match the current benchmark configuration".to_owned(),
            ));
        }
        if context.operation_run_id() != operation_run_id {
            return Err(TraceStoreError::InvalidManifest(
                "existing trace operation_run_id does not match the requested id".to_owned(),
            ));
        }
        context.run().manifest.validate_reproducibility_config()?;
        Ok(context)
    } else {
        BenchmarkRunContext::open_reproducible(trace_path, expected_run, operation_run_id)
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_memory_methods, visibility_policy, ComparisonReport, MethodResult};
    use pi_agent_benchmark::MetricValue;
    use pi_agent_memory::Modality;
    use uuid::Uuid;

    #[test]
    fn parses_multiple_remote_memory_urls() {
        assert_eq!(
            parse_memory_methods(Some(" http://one ,http://two "), None).unwrap(),
            vec![Some("http://one".to_owned()), Some("http://two".to_owned())]
        );
    }

    #[test]
    fn rejects_ambiguous_empty_and_duplicate_memory_urls() {
        assert!(parse_memory_methods(Some("http://one"), Some("http://two")).is_err());
        assert!(parse_memory_methods(Some(" , "), None).is_err());
        assert!(parse_memory_methods(Some("http://one,http://one"), None).is_err());
        assert_eq!(parse_memory_methods(None, None).unwrap(), vec![None]);
    }

    #[test]
    fn comparison_report_keeps_method_metrics_and_identity() {
        let report = ComparisonReport {
            schema_version: 1,
            operation_run_id: Uuid::from_u128(7),
            benchmark_id: "agent.local-model.continual".to_owned(),
            model: "qwen3.8".to_owned(),
            modality: Modality::Text,
            methods: vec![MethodResult {
                method_index: 0,
                operation_run_id: Uuid::from_u128(7),
                run_id: "c-7".to_owned(),
                method_id: "projection".to_owned(),
                version: "1".to_owned(),
                evaluation_profile: "strict".to_owned(),
                capability_adaptations: Vec::new(),
                trace_path: "trace.jsonl".into(),
                runtime_ledger_path: "runtime.jsonl".into(),
                step_count: 1,
                retrieved_hits: 1,
                metrics: vec![MetricValue {
                    run_id: "c-7".to_owned(),
                    name: "retrieval.hit_rate@1".to_owned(),
                    value: 1.0,
                    sample_count: 1,
                }],
            }],
        };
        let encoded = serde_json::to_value(report).expect("comparison report should serialize");
        assert_eq!(encoded["methods"][0]["method_id"], "projection");
        assert_eq!(encoded["methods"][0]["metrics"][0]["value"], 1.0);
        assert_eq!(encoded["modality"], "text");
    }

    #[test]
    fn strict_profile_keeps_cutoff_even_for_limited_methods() {
        assert_eq!(
            visibility_policy("strict", false).unwrap(),
            (Some(100), Vec::new())
        );
    }

    #[test]
    fn adapted_profile_records_visibility_cutoff_adaptation() {
        assert_eq!(
            visibility_policy("adapted", false).unwrap(),
            (None, vec!["visibility_cutoff".to_owned()])
        );
        assert_eq!(
            visibility_policy("adapted", true).unwrap(),
            (Some(100), Vec::new())
        );
    }

    #[test]
    fn benchmark_profile_rejects_unknown_values() {
        assert!(visibility_policy("unknown", false).is_err());
    }
}
