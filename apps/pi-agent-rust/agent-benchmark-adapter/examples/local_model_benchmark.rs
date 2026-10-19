//! Run one real model-backed benchmark step through the full agent boundary.
//!
//! This example deliberately keeps the memory method structural: the question
//! is in the observation, while the JSONL projection supplies bounded memory
//! context by namespace. A semantic or multimodal provider can later replace
//! `ProjectionMemoryMethod` without changing the runner or agent executor.

use std::{
    collections::BTreeMap,
    env,
    error::Error,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use pi_agent_benchmark::{
    read_run, AgentObservation, BenchmarkManifest, BenchmarkRun, BenchmarkRunContext,
    MethodManifest, ObservationPart, TerminalReason, TraceStoreError,
};
use pi_agent_benchmark_adapter::PiAgentExecutor;
use pi_agent_benchmark_evaluator::{
    ReferenceCatalog, ReferenceCatalogManifest, RuntimeLinkedEvaluator, SampleReference,
};
use pi_agent_benchmark_runner::{DatasetSample, RetrievalRunner, VecDataset};
use pi_agent_memory::{
    ArtifactRef, ContentRef, EvidenceLocator, JsonlProjectionStore, MemoryBudget, MemoryItem,
    MemoryKind, MemoryProvenance, MemoryQuery, MemoryRecord, MemoryScope, Modality,
    ProjectionMemoryMethod, StorageContext, StoreWriteRequest,
};
use pi_agent_rust::{AgentLoop, ChatOptions, MemoryHookConfig, ModelClient, ToolRegistry};
use uuid::Uuid;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:6273/v1";
const DEFAULT_MODEL: &str = "qwen3.8";

type BenchmarkFixture = (
    String,
    ContentRef,
    Vec<EvidenceLocator>,
    Option<String>,
    AgentObservation,
);

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let base_url = env::var("PI_AGENT_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned());
    let model_name = env::var("PI_AGENT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
    let operation_run_id = match env::var("PI_AGENT_OPERATION_RUN_ID") {
        Ok(value) => Uuid::parse_str(&value)?,
        Err(env::VarError::NotPresent) => Uuid::new_v4(),
        Err(error) => return Err(error.into()),
    };
    let run_id = format!("local-model-{operation_run_id}");
    let project_root = env::var_os("PI_AGENT_PROJECT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."));
    let default_stem = project_root
        .join("results")
        .join(&run_id)
        .display()
        .to_string();
    let memory_path = env::var_os("PI_AGENT_MEMORY_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("{default_stem}-memory.jsonl")));
    let trace_path = env::var_os("PI_AGENT_TRACE_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("{default_stem}-trace.jsonl")));
    let reference_path = env::var_os("PI_AGENT_REFERENCE_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("{default_stem}-references.json")));
    let runtime_ledger_path = env::var_os("PI_AGENT_RUNTIME_LEDGER_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("{default_stem}-runtime.jsonl")));
    let namespace = env::var("PI_AGENT_MEMORY_NAMESPACE")
        .unwrap_or_else(|_| "benchmark/local-model".to_owned());
    let benchmark_modality = parse_benchmark_modality()?;
    let (prompt, content, evidence, episode_id, observation) =
        fixture(benchmark_modality, operation_run_id)?;

    // Derive the fixture identity from the operation so an interrupted run
    // can be resumed without changing the reference catalog or write payload.
    let record_id = operation_run_id;
    let store = Arc::new(JsonlProjectionStore::open(&memory_path)?);
    let mut manifest = BenchmarkManifest::new(
        "agent.local-model",
        &namespace,
        MethodManifest {
            method_id: "projection".to_owned(),
            version: "1".to_owned(),
            config: BTreeMap::new(),
        },
        "local-fixture-v1",
        Some(100),
    );
    manifest
        .config
        .insert("model".to_owned(), model_name.clone());
    manifest.config.insert(
        "seed".to_owned(),
        env::var("PI_AGENT_SEED").unwrap_or_else(|_| "0".to_owned()),
    );
    manifest.config.insert("prompt".to_owned(), prompt.clone());
    manifest.config.insert(
        "modality".to_owned(),
        modality_name(benchmark_modality).to_owned(),
    );
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
    manifest
        .config
        .insert("tool_choice".to_owned(), "auto".to_owned());
    manifest.config.insert(
        "memory_injection_mode".to_owned(),
        "runtime_retrieval_proposal".to_owned(),
    );
    let expected_reference_manifest = ReferenceCatalogManifest::from_benchmark(&manifest);
    let references = if reference_path.exists() {
        ReferenceCatalog::read_json_for_manifest(&reference_path, &expected_reference_manifest)?
    } else {
        let references = ReferenceCatalog::for_benchmark(
            &manifest,
            vec![SampleReference {
                sample_id: "local-model-sample-0".to_owned(),
                relevant_record_ids: std::collections::BTreeSet::from([record_id.to_string()]),
            }],
        )?;
        references.write_json(&reference_path)?;
        references
    };
    manifest.reference_digest = Some(references.digest()?);
    let evaluator = RuntimeLinkedEvaluator::new(1, references)?;
    manifest.metrics = evaluator.metric_specs();
    manifest.seal_config();
    let run = BenchmarkRun::new(run_id.clone(), manifest, unix_time_ms()?);
    let context = open_or_resume_context(&trace_path, run, operation_run_id)?;
    // RetrievalRunner reads the immutable dataset projection; the runtime hook
    // remains scoped to this concrete run for audit isolation.
    let dataset_namespace = context.dataset_namespace().to_owned();
    let runtime_namespace = context.namespace().to_owned();
    store.write_sync(
        StoreWriteRequest {
            records: vec![MemoryRecord {
                item: MemoryItem {
                    id: record_id,
                    version: 1,
                    kind: if matches!(
                        benchmark_modality,
                        Modality::Image | Modality::Audio | Modality::Video
                    ) {
                        MemoryKind::Episodic
                    } else {
                        MemoryKind::Semantic
                    },
                    scope: MemoryScope::Shared,
                    modality: benchmark_modality,
                    content,
                    namespace: dataset_namespace.clone(),
                    episode_id: episode_id.clone(),
                    subject: None,
                    source: "local-model-fixture".to_owned(),
                    metadata: BTreeMap::new(),
                    abstraction_level: 0,
                    event_time: None,
                    ingest_time_ms: 1,
                    schema_version: 1,
                },
                episode: None,
                evidence,
                provenance: MemoryProvenance {
                    producer: "local-model-fixture".to_owned(),
                    source_id: "local-model-sample-0".to_owned(),
                    source_event_id: None,
                    parent_record_ids: Vec::new(),
                    method_version: Some("1".to_owned()),
                },
            }],
            relations: Vec::new(),
            idempotency_key: format!("local-model-fixture-{operation_run_id}"),
            source_revision: Some(1),
        },
        StorageContext {
            namespace: dataset_namespace,
            run_id: operation_run_id,
            snapshot_id: None,
            timeout_ms: None,
        },
    )?;
    let method = ProjectionMemoryMethod::new(store, "projection", "1")?;
    let dataset = VecDataset::new(vec![DatasetSample {
        sample_id: "local-model-sample-0".to_owned(),
        observation: AgentObservation {
            parts: observation.parts,
        },
        query: MemoryQuery {
            input: None,
            namespace: namespace.clone(),
            scope: None,
            subject: None,
            episode_id: episode_id.clone(),
            kinds: Vec::new(),
            modalities: vec![benchmark_modality],
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
    }]);

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
    let summary = RetrievalRunner::new(method, dataset, context)
        .run_with_agent(PiAgentExecutor::new_with_runtime_and_retrieval_memory(
            agent,
            operation_run_id,
            memory_config,
        ))
        .await
        .map_err(|error| format!("benchmark run failed: {error}"))?;
    let trace = read_run(&trace_path, &summary.run_id)?;
    let report = evaluator.evaluate_with_ledger(&trace, &runtime_ledger_path)?;
    report.append_to_trace(&trace_path)?;
    if summary.retrieval_error_count != 0 || summary.agent_error_count != 0 {
        let errors = trace
            .steps
            .iter()
            .flat_map(|step| step.errors.iter())
            .map(|error| format!("{}: {}", error.code, error.message))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(format!(
            "benchmark completed with retrieval_errors={} agent_errors={}: {}; inspect {}",
            summary.retrieval_error_count,
            summary.agent_error_count,
            errors,
            trace_path.display()
        )
        .into());
    }
    let answer = trace
        .steps
        .first()
        .and_then(|step| step.agent_answer.as_deref());
    println!(
        "run_id={} steps={} retrieved_hits={} metrics={} trace={} references={} runtime_ledger={} answer={}",
        summary.run_id,
        summary.step_count,
        trace
            .steps
            .first()
            .map_or(0, |step| step.observed_hits.len()),
        report.metrics.len(),
        trace_path.display(),
        reference_path.display(),
        runtime_ledger_path.display(),
        answer.unwrap_or("<empty>")
    );
    Ok(())
}

fn unix_time_ms() -> Result<i64, Box<dyn Error>> {
    let millis = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    Ok(i64::try_from(millis)?)
}

fn parse_benchmark_modality() -> Result<Modality, Box<dyn Error>> {
    let value = env::var("PI_AGENT_BENCHMARK_MODALITY").unwrap_or_else(|_| "text".to_owned());
    parse_benchmark_modality_value(&value).ok_or_else(|| {
        format!("PI_AGENT_BENCHMARK_MODALITY supports text, image, audio, or video, got {value:?}")
            .into()
    })
}

fn parse_benchmark_modality_value(value: &str) -> Option<Modality> {
    match value {
        "text" => Some(Modality::Text),
        "image" => Some(Modality::Image),
        "audio" => Some(Modality::Audio),
        "video" => Some(Modality::Video),
        _ => None,
    }
}

fn modality_name(modality: Modality) -> &'static str {
    match modality {
        Modality::Text => "text",
        Modality::Image => "image",
        Modality::Audio => "audio",
        Modality::Video => "video",
        _ => "other",
    }
}

fn fixture(modality: Modality, operation_run_id: Uuid) -> Result<BenchmarkFixture, Box<dyn Error>> {
    match modality {
        Modality::Text => Ok((
            "请解释 JSONL 记忆投影如何避免未来数据泄漏。".to_owned(),
            ContentRef::Text {
                text: "Rust 的 JSONL 记忆投影支持按 namespace 和时间截断检索。".to_owned(),
            },
            Vec::new(),
            None,
            AgentObservation {
                parts: vec![ObservationPart::Text(
                    "请解释 JSONL 记忆投影如何避免未来数据泄漏。".to_owned(),
                )],
            },
        )),
        Modality::Image => Ok(binary_fixture(
            operation_run_id,
            Modality::Image,
            "image/png",
            "episode-1.png",
            "请根据检索到的图像证据描述关键内容。",
        )),
        Modality::Audio => Ok(binary_fixture(
            operation_run_id,
            Modality::Audio,
            "audio/wav",
            "episode-1.wav",
            "请根据检索到的音频证据说明关键事件。",
        )),
        Modality::Video => Ok(binary_fixture(
            operation_run_id,
            Modality::Video,
            "video/mp4",
            "episode-1.mp4",
            "请根据检索到的视频证据说明抓取物体的关键帧。",
        )),
        other => Err(format!(
            "PI_AGENT_BENCHMARK_MODALITY supports text, image, audio, or video, got {other:?}"
        )
        .into()),
    }
}

fn binary_fixture(
    operation_run_id: Uuid,
    modality: Modality,
    media_type: &str,
    artifact_name: &str,
    prompt: &str,
) -> BenchmarkFixture {
    let artifact = ArtifactRef {
        artifact_id: operation_run_id,
        uri: format!("cas://benchmark/local-model/{artifact_name}"),
        media_type: media_type.to_owned(),
        sha256: format!("sha256:local-model-{}", modality_name(modality)),
        bytes: 8_192,
        codec: match modality {
            Modality::Video => Some("h264".to_owned()),
            Modality::Audio => Some("pcm_s16le".to_owned()),
            _ => None,
        },
        schema: Some(format!("episode-{}-v1", modality_name(modality))),
        stream_key: match modality {
            Modality::Video => Some("camera/front".to_owned()),
            Modality::Audio => Some("microphone/front".to_owned()),
            _ => None,
        },
        storage_revision: Some("fixture-v1".to_owned()),
    };
    let evidence = EvidenceLocator {
        artifact: artifact.clone(),
        entity_path: Some("episode/object-3".to_owned()),
        time_range: None,
        frame_start: (modality == Modality::Video).then_some(120),
        frame_end: (modality == Modality::Video).then_some(150),
        step_start: (modality == Modality::Video).then_some(30),
        step_end: (modality == Modality::Video).then_some(34),
        byte_start: None,
        byte_end: None,
    };
    (
        prompt.to_owned(),
        ContentRef::Blob {
            uri: artifact.uri,
            media_type: artifact.media_type,
            sha256: artifact.sha256,
            size_bytes: artifact.bytes,
        },
        vec![evidence],
        Some("episode-1".to_owned()),
        AgentObservation {
            parts: vec![ObservationPart::Artifact {
                artifact_id: artifact.artifact_id.to_string(),
                modality: modality_name(modality).to_owned(),
            }],
        },
    )
}

/// Open a new trace or resume the persisted run without changing its start
/// time. A freshly constructed [`BenchmarkRun`] always has a new timestamp,
/// so passing it directly to `open_reproducible` would reject an interrupted
/// trace even when its manifest is unchanged.
///
/// Resuming is intentionally strict: the caller must provide the same run
/// identity and operation UUID, and the newly built manifest must match the
/// persisted manifest. This keeps a resumed trace tied to the exact model,
/// prompt, dataset, and memory method configuration that created it.
fn open_or_resume_context(
    trace_path: &std::path::Path,
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
    use super::*;
    use tempfile::tempdir;

    fn reproducible_run(run_id: &str, started_at_ms: i64) -> BenchmarkRun {
        let mut manifest = BenchmarkManifest::new(
            "test-benchmark",
            "test-namespace",
            MethodManifest {
                method_id: "projection".to_owned(),
                version: "1".to_owned(),
                config: BTreeMap::new(),
            },
            "dataset-v1",
            Some(100),
        );
        manifest
            .config
            .insert("model".to_owned(), "test".to_owned());
        manifest.config.insert("seed".to_owned(), "0".to_owned());
        manifest
            .config
            .insert("prompt".to_owned(), "test prompt".to_owned());
        manifest.seal_config();
        BenchmarkRun::new(run_id, manifest, started_at_ms)
    }

    #[test]
    fn resumes_existing_trace_without_reusing_a_new_start_timestamp() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let operation_run_id = Uuid::new_v4();

        let context = open_or_resume_context(
            &path,
            reproducible_run("local-model-test", 1),
            operation_run_id,
        )
        .unwrap();
        drop(context);

        let resumed = open_or_resume_context(
            &path,
            reproducible_run("local-model-test", 2),
            operation_run_id,
        )
        .unwrap();
        assert_eq!(resumed.run().started_at_ms, 1);
        assert_eq!(resumed.operation_run_id(), operation_run_id);
    }

    #[test]
    fn rejects_resume_when_manifest_or_operation_identity_changes() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let operation_run_id = Uuid::new_v4();
        let context = open_or_resume_context(
            &path,
            reproducible_run("local-model-test", 1),
            operation_run_id,
        )
        .unwrap();
        drop(context);

        let mut changed = reproducible_run("local-model-test", 2);
        changed
            .manifest
            .config
            .insert("model".to_owned(), "changed".to_owned());
        changed.manifest.seal_config();
        assert!(matches!(
            open_or_resume_context(&path, changed, operation_run_id),
            Err(TraceStoreError::InvalidManifest(message))
                if message.contains("current benchmark configuration")
        ));

        assert!(matches!(
            open_or_resume_context(
                &path,
                reproducible_run("local-model-test", 3),
                Uuid::new_v4()
            ),
            Err(TraceStoreError::InvalidManifest(message))
                if message.contains("operation_run_id")
        ));
    }

    #[test]
    fn parses_all_supported_benchmark_modalities() {
        assert_eq!(parse_benchmark_modality_value("text"), Some(Modality::Text));
        assert_eq!(
            parse_benchmark_modality_value("image"),
            Some(Modality::Image)
        );
        assert_eq!(
            parse_benchmark_modality_value("audio"),
            Some(Modality::Audio)
        );
        assert_eq!(
            parse_benchmark_modality_value("video"),
            Some(Modality::Video)
        );
        assert_eq!(parse_benchmark_modality_value("unknown"), None);
    }

    #[test]
    fn binary_fixtures_use_episode_records_and_artifact_evidence() {
        for (modality, media_type, suffix) in [
            (Modality::Image, "image/png", "episode-1.png"),
            (Modality::Audio, "audio/wav", "episode-1.wav"),
            (Modality::Video, "video/mp4", "episode-1.mp4"),
        ] {
            let (_, content, evidence, episode_id, observation) =
                fixture(modality, Uuid::from_u128(7)).unwrap();
            let ContentRef::Blob {
                uri,
                media_type: actual_media_type,
                ..
            } = content
            else {
                panic!("binary fixture must use blob content");
            };
            assert_eq!(actual_media_type, media_type);
            assert!(uri.ends_with(suffix));
            assert_eq!(evidence.len(), 1);
            assert_eq!(episode_id.as_deref(), Some("episode-1"));
            assert!(matches!(
                observation.parts.first(),
                Some(ObservationPart::Artifact { modality: value, .. })
                    if value == modality_name(modality)
            ));
        }
    }
}
