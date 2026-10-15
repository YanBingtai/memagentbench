use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    sync::Arc,
};

use pi_agent_benchmark::{
    read_run, AgentObservation, ArtifactValidationPolicy, BenchmarkManifest, BenchmarkRun,
    MethodManifest, ObservationPart, TerminalReason,
};
use pi_agent_benchmark_evaluator::{ReferenceCatalog, RetrievalEvaluator, SampleReference};
use pi_agent_benchmark_runner::{
    AgentFuture, AgentStepExecutor, AgentStepInput, AgentStepOutput, DatasetSample,
    RetrievalRunner, VecDataset,
};
use pi_agent_memory::{
    ArtifactRef, ContentRef, EvidenceLocator, JsonlProjectionStore, MemoryBudget, MemoryItem,
    MemoryKind, MemoryProvenance, MemoryQuery, MemoryRecord, MemoryScope, Modality,
    ProjectionMemoryMethod, StorageContext, StoreWriteRequest,
};
use tempfile::tempdir;
use uuid::Uuid;

struct DeterministicAgent;

impl AgentStepExecutor for DeterministicAgent {
    fn execute<'a>(&'a mut self, input: AgentStepInput) -> AgentFuture<'a> {
        let answer = format!(
            "sample={} retrieved_hits={}",
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

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let projection_path = directory.path().join("projection.jsonl");
    let trace_path = directory.path().join("trace.jsonl");
    let namespace = "bench/video-smoke".to_owned();
    let operation_run_id = Uuid::from_u128(1);
    let record_id = Uuid::from_u128(2);
    let artifact = ArtifactRef {
        artifact_id: Uuid::from_u128(3),
        uri: "cas://video-smoke/episode-1.mp4".to_owned(),
        media_type: "video/mp4".to_owned(),
        sha256: "sha256:episode-1".to_owned(),
        bytes: 8_192,
        codec: Some("h264".to_owned()),
        schema: Some("episode-video-v1".to_owned()),
        stream_key: Some("camera/front".to_owned()),
        storage_revision: Some("fixture-v1".to_owned()),
    };
    let evidence = EvidenceLocator {
        artifact: artifact.clone(),
        entity_path: Some("robot/gripper/object-3".to_owned()),
        time_range: None,
        frame_start: Some(120),
        frame_end: Some(150),
        step_start: Some(30),
        step_end: Some(34),
        byte_start: None,
        byte_end: None,
    };

    let store = Arc::new(JsonlProjectionStore::open(&projection_path)?);
    let mut manifest = BenchmarkManifest::new(
        "retrieval.video.smoke",
        &namespace,
        MethodManifest {
            method_id: "projection-video".to_owned(),
            version: "1".to_owned(),
            config: BTreeMap::new(),
        },
        "video-fixture-v1",
        Some(100),
    );
    manifest.artifact_validation_policy = ArtifactValidationPolicy::Unchecked;
    let references = ReferenceCatalog::for_benchmark(
        &manifest,
        vec![SampleReference {
            sample_id: "video-sample-0".to_owned(),
            relevant_record_ids: BTreeSet::from([record_id.to_string()]),
        }],
    )?;
    manifest.reference_digest = Some(references.digest()?);
    let evaluator = RetrievalEvaluator::new(1, references)?;
    manifest.metrics = evaluator.metric_specs();

    let query = MemoryQuery {
        input: None,
        namespace: namespace.clone(),
        scope: None,
        subject: None,
        episode_id: Some("episode-1".to_owned()),
        kinds: vec![MemoryKind::Episodic],
        modalities: vec![Modality::Video],
        time_range: None,
        visibility_cutoff_ms: None,
        graph: None,
        budget: MemoryBudget {
            max_hits: 1,
            ..MemoryBudget::default()
        },
    };
    let run = BenchmarkRun::new("video-smoke-run", manifest, 1);
    let context =
        pi_agent_benchmark::BenchmarkRunContext::open(&trace_path, run, operation_run_id)?;
    // This smoke test exercises a prebuilt dataset projection through
    // RetrievalRunner, so the fixture must live in the dataset namespace.
    let dataset_namespace = context.dataset_namespace().to_owned();
    store.write_sync(
        StoreWriteRequest {
            records: vec![MemoryRecord {
                item: MemoryItem {
                    id: record_id,
                    version: 1,
                    kind: MemoryKind::Episodic,
                    scope: MemoryScope::Shared,
                    modality: Modality::Video,
                    content: ContentRef::Blob {
                        uri: artifact.uri.clone(),
                        media_type: artifact.media_type.clone(),
                        sha256: artifact.sha256.clone(),
                        size_bytes: artifact.bytes,
                    },
                    namespace: dataset_namespace.clone(),
                    episode_id: Some("episode-1".to_owned()),
                    subject: None,
                    source: "video-trace-smoke".to_owned(),
                    metadata: BTreeMap::new(),
                    abstraction_level: 0,
                    event_time: None,
                    ingest_time_ms: 90,
                    schema_version: 1,
                },
                episode: None,
                evidence: vec![evidence],
                provenance: MemoryProvenance {
                    producer: "video-trace-smoke".to_owned(),
                    source_id: "episode-1".to_owned(),
                    source_event_id: None,
                    parent_record_ids: Vec::new(),
                    method_version: Some("fixture-1".to_owned()),
                },
            }],
            relations: Vec::new(),
            idempotency_key: "video-fixture-1".to_owned(),
            source_revision: Some(1),
        },
        StorageContext {
            namespace: dataset_namespace,
            run_id: operation_run_id,
            snapshot_id: None,
            timeout_ms: None,
        },
    )?;
    let method = ProjectionMemoryMethod::new(store, "projection-video", "1")?;
    let dataset = VecDataset::new(vec![DatasetSample {
        sample_id: "video-sample-0".to_owned(),
        observation: AgentObservation {
            parts: vec![ObservationPart::Text("找到抓取物体的关键视频帧".to_owned())],
        },
        query,
        session_id: None,
        episode_id: None,
        terminal: true,
        terminal_reason: Some(TerminalReason::Success),
    }]);

    let summary = RetrievalRunner::new(method, dataset, context)
        .run_with_agent(DeterministicAgent)
        .await?;
    let trace = read_run(&trace_path, &summary.run_id)?;
    let report = evaluator.evaluate(&trace)?;
    report.append_to_trace(&trace_path)?;
    let evaluated = read_run(&trace_path, &summary.run_id)?;
    let hit = evaluated
        .steps
        .first()
        .and_then(|step| step.observed_hits.first())
        .ok_or("video benchmark did not record a retrieval hit")?;
    let evidence_id = hit
        .evidence_ids
        .first()
        .ok_or("video benchmark did not record evidence identity")?;

    assert_eq!(summary.step_count, 1);
    assert_eq!(summary.retrieval_error_count, 0);
    assert_eq!(summary.agent_error_count, 0);
    assert_eq!(evaluated.metrics, report.metrics);
    assert!(evidence_id.starts_with(&format!("artifact:{}#sha256:", Uuid::from_u128(3))));
    assert!(evaluated.steps[0].telemetry.evidence_bytes > 0);
    println!(
        "video_trace_smoke_ok steps={} retrieved_hits={} metrics={} evidence_id={} trace={}",
        evaluated.steps.len(),
        evaluated.steps[0].observed_hits.len(),
        evaluated.metrics.len(),
        evidence_id,
        trace_path.display()
    );
    Ok(())
}
