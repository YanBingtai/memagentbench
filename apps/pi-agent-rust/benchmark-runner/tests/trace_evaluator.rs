use std::collections::{BTreeMap, BTreeSet};

use pi_agent_benchmark::{
    read_all, read_run, trace_digest, AgentObservation, BenchmarkManifest, BenchmarkRun,
    MethodManifest, ObservationPart, TerminalReason, TraceErrorStage,
};
use pi_agent_benchmark_evaluator::{ReferenceCatalog, RetrievalEvaluator, SampleReference};
use pi_agent_benchmark_runner::{DatasetSample, RetrievalRunner, VecDataset};
use pi_agent_memory::{
    ArtifactRef, ContentRef, EvidenceLocator, IngestReceipt, MemoryBudget, MemoryCapabilities,
    MemoryError, MemoryFuture, MemoryHit, MemoryItem, MemoryKind, MemoryManifest, MemoryMethod,
    MemoryQuery, Modality, OperationContext, RetrieveResponse,
};
use tempfile::tempdir;
use uuid::Uuid;

struct FixtureMethod {
    artifact_id: Uuid,
    record_id: Uuid,
}

impl MemoryMethod for FixtureMethod {
    fn manifest(&self) -> MemoryManifest {
        MemoryManifest {
            method_id: "fixture".into(),
            version: "1".into(),
            schema_version: 1,
            capabilities: MemoryCapabilities {
                modalities: vec![Modality::Text, Modality::Video],
                memory_kinds: vec![MemoryKind::Semantic],
                supports_content_search: true,
                supports_time_filter: true,
                supports_visibility_cutoff: true,
                supports_graph: false,
                supports_cross_modal: true,
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
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        Box::pin(async { Err(MemoryError::UnsupportedCapability("ingest".into())) })
    }

    fn retrieve<'a>(
        &'a self,
        query: MemoryQuery,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
        let artifact_id = self.artifact_id;
        let record_id = self.record_id;
        Box::pin(async move {
            let is_timeout = matches!(
                query.input,
                Some(ContentRef::Text { ref text }) if text == "timeout"
            );
            if is_timeout {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            if is_timeout {
                return Ok(RetrieveResponse {
                    query_id: Uuid::from_u128(10),
                    hits: Vec::new(),
                    snapshot_id: None,
                    latency_ms: 20,
                    truncated: false,
                    telemetry: None,
                });
            }
            Ok(RetrieveResponse {
                query_id: Uuid::from_u128(11),
                hits: vec![MemoryHit {
                    item_id: record_id,
                    source_record_ids: Vec::new(),
                    version: 1,
                    method_id: "fixture".into(),
                    rank: 1,
                    score: Some(1.0),
                    evidence: Vec::new(),
                    evidence_refs: vec![EvidenceLocator {
                        artifact: ArtifactRef {
                            artifact_id,
                            uri: "cas://fixture/video-1".into(),
                            media_type: "video/mp4".into(),
                            sha256: "sha256:fixture".into(),
                            bytes: 42,
                            codec: Some("h264".into()),
                            schema: None,
                            stream_key: None,
                            storage_revision: Some("r1".into()),
                        },
                        entity_path: Some("robot/gripper/object-1".into()),
                        time_range: None,
                        frame_start: Some(10),
                        frame_end: Some(20),
                        step_start: Some(3),
                        step_end: Some(5),
                        byte_start: None,
                        byte_end: None,
                    }],
                    projection: None,
                    provenance: "fixture".into(),
                    snapshot_id: Some("snapshot-1".into()),
                    latency_ms: 1,
                }],
                snapshot_id: Some("snapshot-1".into()),
                latency_ms: 1,
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
        Box::pin(async { Err(MemoryError::UnsupportedCapability("get".into())) })
    }
}

fn sample(sample_id: &str, input: &str, terminal: bool) -> DatasetSample {
    DatasetSample {
        sample_id: sample_id.into(),
        observation: AgentObservation {
            parts: vec![ObservationPart::Text(format!("observation-{sample_id}"))],
        },
        query: MemoryQuery {
            input: Some(ContentRef::Text { text: input.into() }),
            namespace: String::new(),
            scope: None,
            subject: None,
            episode_id: None,
            kinds: Vec::new(),
            modalities: vec![Modality::Video],
            time_range: None,
            visibility_cutoff_ms: None,
            graph: None,
            budget: MemoryBudget {
                max_hits: 1,
                max_latency_ms: (input == "timeout").then_some(1),
                ..MemoryBudget::default()
            },
        },
        session_id: None,
        episode_id: None,
        terminal,
        terminal_reason: terminal.then_some(TerminalReason::Success),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn runner_trace_evaluator_round_trip_preserves_video_evidence_and_timeout() {
    let directory = tempdir().unwrap();
    let trace_path = directory.path().join("trace.jsonl");
    let record_id = Uuid::from_u128(1);
    let artifact_id = Uuid::from_u128(2);
    let mut manifest = BenchmarkManifest::new(
        "retrieval.integration",
        "bench/integration",
        MethodManifest {
            method_id: "fixture".into(),
            version: "1".into(),
            config: BTreeMap::new(),
        },
        "dataset-v1",
        Some(100),
    );
    let references = ReferenceCatalog::for_benchmark(
        &manifest,
        vec![
            SampleReference {
                sample_id: "video-hit".into(),
                relevant_record_ids: BTreeSet::from([record_id.to_string()]),
            },
            SampleReference {
                sample_id: "timeout".into(),
                relevant_record_ids: BTreeSet::from([Uuid::from_u128(99).to_string()]),
            },
        ],
    )
    .unwrap();
    let reference_digest = references.digest().unwrap();
    manifest.reference_digest = Some(reference_digest.clone());
    let evaluator = RetrievalEvaluator::new(1, references).unwrap();
    manifest.metrics = evaluator.metric_specs();

    let run = BenchmarkRun::new("integration-run", manifest, 1);
    let context =
        pi_agent_benchmark::BenchmarkRunContext::open(&trace_path, run, Uuid::from_u128(3))
            .unwrap();
    let summary = RetrievalRunner::new(
        FixtureMethod {
            artifact_id,
            record_id,
        },
        VecDataset::new(vec![
            sample("video-hit", "video", false),
            sample("timeout", "timeout", true),
        ]),
        context,
    )
    .run()
    .await
    .unwrap();

    let trace = read_run(&trace_path, &summary.run_id).unwrap();
    let report = evaluator.evaluate(&trace).unwrap();
    let trace_digest_before_metrics = trace_digest(&read_all(&trace_path).unwrap()).unwrap();
    report.append_to_trace(&trace_path).unwrap();
    report.append_to_trace(&trace_path).unwrap();
    let evaluated = read_run(&trace_path, &summary.run_id).unwrap();

    assert_eq!(summary.step_count, 2);
    assert_eq!(summary.retrieval_error_count, 1);
    assert_eq!(
        trace.steps[0]
            .query
            .as_ref()
            .unwrap()
            .snapshot_id
            .as_deref(),
        Some("snapshot-1")
    );
    assert_eq!(trace.steps[0].observed_hits[0].evidence_ids.len(), 1);
    assert!(trace.steps[0].observed_hits[0].evidence_ids[0]
        .starts_with(&format!("artifact:{artifact_id}#sha256:")));
    assert!(trace.steps[0].telemetry.evidence_bytes > 0);
    assert_eq!(trace.steps[1].telemetry.evidence_bytes, 0);
    assert_eq!(trace.steps[1].errors[0].stage, TraceErrorStage::Retrieval);
    assert_eq!(trace.steps[1].errors[0].code, "timeout");
    assert_eq!(report.metrics[0].value, 0.5);
    assert_eq!(report.metrics[1].value, 0.5);
    assert_eq!(report.metrics[2].value, 0.5);
    assert_eq!(evaluated.metrics, report.metrics);
    assert_eq!(
        evaluated.run.manifest.reference_digest.as_deref(),
        Some(reference_digest.as_str())
    );

    let records = read_all(&trace_path).unwrap();
    let trace_digest_after_metrics = trace_digest(&records).unwrap();
    assert_ne!(trace_digest_before_metrics, trace_digest_after_metrics);
    assert_eq!(
        trace_digest_after_metrics,
        trace_digest(&read_all(&trace_path).unwrap()).unwrap()
    );
}
