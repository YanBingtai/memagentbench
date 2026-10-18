use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use pi_agent_benchmark::{
    read_run, AgentObservation, BenchmarkManifest, BenchmarkRun, MethodManifest, ObservationPart,
    TerminalReason,
};
use pi_agent_benchmark_evaluator::{ContinualEvaluator, ReferenceCatalog, SampleReference};
use pi_agent_benchmark_runner::{
    AgentFuture, AgentStepExecutor, AgentStepInput, AgentStepOutput, ContinualDatasetSample,
    ContinualEpisodeRunner, DatasetError, DatasetSample, JsonlContinualDataset, JsonlDataset,
    RetrievalRunner, RunnerError,
};
use pi_agent_memory::{
    ContentRef, IngestReceipt, IngestRequest, JsonlProjectionStore, MemoryBudget,
    MemoryCapabilities, MemoryError, MemoryFuture, MemoryItem, MemoryKind, MemoryManifest,
    MemoryMethod, MemoryQuery, MemoryRelation, MemoryScope, Modality, OperationContext,
    ProjectionMemoryMethod, RelationQuery, RelationType, RetrieveResponse, StorageContext,
    StoreGetRequest, TraversalDirection,
};
use tempfile::tempdir;
use uuid::Uuid;

#[derive(Clone)]
struct CountingMethod {
    ingest_calls: Arc<AtomicUsize>,
    retrieve_calls: Arc<AtomicUsize>,
}

impl CountingMethod {
    fn new() -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let ingest_calls = Arc::new(AtomicUsize::new(0));
        let retrieve_calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                ingest_calls: ingest_calls.clone(),
                retrieve_calls: retrieve_calls.clone(),
            },
            ingest_calls,
            retrieve_calls,
        )
    }

    fn manifest() -> MemoryManifest {
        MemoryManifest {
            method_id: "counting".into(),
            version: "1".into(),
            schema_version: 1,
            capabilities: MemoryCapabilities {
                modalities: vec![Modality::Text, Modality::Video, Modality::State],
                memory_kinds: vec![MemoryKind::Episodic, MemoryKind::Semantic],
                supports_content_search: false,
                supports_time_filter: true,
                supports_visibility_cutoff: true,
                supports_graph: false,
                supports_cross_modal: true,
                supports_update: false,
                supports_delete: false,
                supports_batch: true,
                supports_record_ingest: false,
                supports_record_get: false,
                deterministic: true,
                persistent: false,
            },
        }
    }
}

impl MemoryMethod for CountingMethod {
    fn manifest(&self) -> MemoryManifest {
        Self::manifest()
    }

    fn ingest<'a>(
        &'a self,
        request: IngestRequest,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        self.ingest_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            Ok(IngestReceipt {
                item_ids: request.items.into_iter().map(|item| item.id).collect(),
                relation_ids: request
                    .relations
                    .into_iter()
                    .map(|relation| relation.id)
                    .collect(),
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
        self.retrieve_calls.fetch_add(1, Ordering::SeqCst);
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
        Box::pin(async { Err(MemoryError::UnsupportedCapability("get".into())) })
    }
}

#[derive(Clone, Default)]
struct CountingAgent {
    calls: Arc<AtomicUsize>,
}

impl CountingAgent {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                calls: calls.clone(),
            },
            calls,
        )
    }
}

impl AgentStepExecutor for CountingAgent {
    fn execute<'a>(&'a mut self, _input: AgentStepInput) -> AgentFuture<'a> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(AgentStepOutput::default()) })
    }
}

fn sample(sample_id: &str) -> DatasetSample {
    DatasetSample {
        sample_id: sample_id.into(),
        observation: AgentObservation {
            parts: vec![ObservationPart::Text(format!("observation-{sample_id}"))],
        },
        query: MemoryQuery {
            input: None,
            namespace: String::new(),
            scope: None,
            subject: None,
            episode_id: None,
            kinds: Vec::new(),
            modalities: Vec::new(),
            time_range: None,
            visibility_cutoff_ms: None,
            graph: None,
            budget: MemoryBudget {
                max_hits: 8,
                ..MemoryBudget::default()
            },
        },
        session_id: None,
        episode_id: None,
        terminal: true,
        terminal_reason: Some(TerminalReason::Success),
    }
}

fn continual_sample(sample_id: &str) -> ContinualDatasetSample {
    let item_id = Uuid::from_u128(10);
    ContinualDatasetSample {
        sample: sample(sample_id),
        ingest: IngestRequest {
            items: vec![
                MemoryItem {
                    id: item_id,
                    version: 1,
                    kind: MemoryKind::Episodic,
                    scope: MemoryScope::Personalized,
                    modality: Modality::Video,
                    content: ContentRef::Blob {
                        uri: "cas://episode/video.mp4".into(),
                        media_type: "video/mp4".into(),
                        sha256: "sha256:video".into(),
                        size_bytes: 42,
                    },
                    namespace: String::new(),
                    episode_id: Some("episode-1".into()),
                    subject: None,
                    source: "camera.front".into(),
                    metadata: BTreeMap::new(),
                    abstraction_level: 0,
                    event_time: None,
                    ingest_time_ms: 1,
                    schema_version: 1,
                },
                MemoryItem {
                    id: Uuid::from_u128(11),
                    version: 1,
                    kind: MemoryKind::Semantic,
                    scope: MemoryScope::Personalized,
                    modality: Modality::State,
                    content: ContentRef::Json {
                        value: serde_json::json!({"gripper": "open"}),
                    },
                    namespace: String::new(),
                    episode_id: Some("episode-1".into()),
                    subject: None,
                    source: "robot.state".into(),
                    metadata: BTreeMap::new(),
                    abstraction_level: 0,
                    event_time: None,
                    ingest_time_ms: 1,
                    schema_version: 1,
                },
            ],
            relations: vec![MemoryRelation {
                id: Uuid::from_u128(12),
                from: item_id,
                to: Uuid::from_u128(11),
                relation_type: RelationType::Supports,
                weight: Some(1.0),
                confidence: Some(1.0),
                valid_time: None,
                ingest_time_ms: 1,
                source: "fixture".into(),
                metadata: BTreeMap::new(),
                schema_version: 1,
            }],
            idempotency_key: format!("ingest-{sample_id}"),
        },
    }
}

fn manifest(method_id: &str, dataset_digest: Option<String>) -> BenchmarkManifest {
    let mut manifest = BenchmarkManifest::new(
        "jsonl.integration",
        "bench/jsonl",
        MethodManifest {
            method_id: method_id.into(),
            version: "1".into(),
            config: BTreeMap::new(),
        },
        "dataset-v1",
        None,
    );
    manifest.dataset_digest = dataset_digest;
    manifest
}

fn write_jsonl<T: serde::Serialize>(path: &std::path::Path, value: &T) {
    fs::write(path, format!("{}\n", serde_json::to_string(value).unwrap())).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn retrieval_jsonl_digest_is_required_before_provider_calls() {
    for expected in [None, Some("sha256:wrong".to_owned())] {
        let directory = tempdir().unwrap();
        let dataset_path = directory.path().join("dataset.jsonl");
        let trace_path = directory.path().join("trace.jsonl");
        write_jsonl(&dataset_path, &sample("retrieval"));
        let source = JsonlDataset::open(&dataset_path).unwrap();
        let (method, ingest_calls, retrieve_calls) = CountingMethod::new();
        let missing_digest = expected.is_none();
        let run = BenchmarkRun::new("retrieval-run", manifest("counting", expected), 1);
        let context =
            pi_agent_benchmark::BenchmarkRunContext::open(&trace_path, run, Uuid::from_u128(100))
                .unwrap();
        let (agent, agent_calls) = CountingAgent::new();
        let result = RetrievalRunner::new(method, source, context)
            .run_with_agent(agent)
            .await;
        assert!(
            matches!(
                result,
                Err(RunnerError::DatasetDigestMissing)
                    if missing_digest
            ) || matches!(result, Err(RunnerError::DatasetDigestMismatch { .. }))
        );
        assert_eq!(ingest_calls.load(Ordering::SeqCst), 0);
        assert_eq!(retrieve_calls.load(Ordering::SeqCst), 0);
        assert_eq!(agent_calls.load(Ordering::SeqCst), 0);
        assert!(!read_run(&trace_path, "retrieval-run").unwrap().complete);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn continual_jsonl_digest_is_required_before_ingest_retrieve_and_agent() {
    for expected in [None, Some("sha256:wrong".to_owned())] {
        let directory = tempdir().unwrap();
        let dataset_path = directory.path().join("dataset.jsonl");
        let trace_path = directory.path().join("trace.jsonl");
        write_jsonl(&dataset_path, &continual_sample("continual"));
        let source = JsonlContinualDataset::open(&dataset_path).unwrap();
        let (method, ingest_calls, retrieve_calls) = CountingMethod::new();
        let missing_digest = expected.is_none();
        let run = BenchmarkRun::new("continual-run", manifest("counting", expected), 1);
        let context =
            pi_agent_benchmark::BenchmarkRunContext::open(&trace_path, run, Uuid::from_u128(101))
                .unwrap();
        let (agent, agent_calls) = CountingAgent::new();
        let result = ContinualEpisodeRunner::new(method, source, context)
            .run_with_agent(agent)
            .await;
        assert!(
            matches!(
                result,
                Err(RunnerError::DatasetDigestMissing)
                    if missing_digest
            ) || matches!(result, Err(RunnerError::DatasetDigestMismatch { .. }))
        );
        assert_eq!(ingest_calls.load(Ordering::SeqCst), 0);
        assert_eq!(retrieve_calls.load(Ordering::SeqCst), 0);
        assert_eq!(agent_calls.load(Ordering::SeqCst), 0);
        assert!(!read_run(&trace_path, "continual-run").unwrap().complete);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn correct_continual_jsonl_digest_round_trips_multimodal_projection_and_evaluator() {
    let directory = tempdir().unwrap();
    let dataset_path = directory.path().join("dataset.jsonl");
    let trace_path = directory.path().join("trace.jsonl");
    let projection_path = directory.path().join("projection.jsonl");
    let expected = continual_sample("multimodal");
    write_jsonl(&dataset_path, &expected);
    let source = JsonlContinualDataset::open(&dataset_path).unwrap();
    let dataset_digest = source.content_digest().unwrap();
    let store = Arc::new(JsonlProjectionStore::open(&projection_path).unwrap());
    let method = ProjectionMemoryMethod::new(store.clone(), "projection", "1").unwrap();
    let mut run_manifest = manifest("projection", Some(dataset_digest));
    let references = ReferenceCatalog::for_benchmark(
        &run_manifest,
        vec![SampleReference {
            sample_id: "multimodal".into(),
            relevant_record_ids: BTreeSet::from([Uuid::from_u128(10).to_string()]),
        }],
    )
    .unwrap();
    run_manifest.reference_digest = Some(references.digest().unwrap());
    let evaluator = ContinualEvaluator::new(8, references).unwrap();
    run_manifest.metrics = evaluator.metric_specs();
    let operation_run_id = Uuid::from_u128(102);
    let run = BenchmarkRun::new("continual-projection", run_manifest, 1);
    let context =
        pi_agent_benchmark::BenchmarkRunContext::open(&trace_path, run, operation_run_id).unwrap();

    let summary = ContinualEpisodeRunner::new(method, source, context)
        .run()
        .await
        .unwrap();
    let trace = read_run(&trace_path, &summary.run_id).unwrap();
    let report = evaluator.evaluate(&trace).unwrap();

    assert_eq!(summary.step_count, 1);
    assert_eq!(summary.ingest_error_count, 0);
    assert!(trace.complete);
    assert_eq!(
        trace.steps[0].ingest.as_ref().unwrap().item_ids,
        vec![
            Uuid::from_u128(10).to_string(),
            Uuid::from_u128(11).to_string()
        ]
    );
    assert_eq!(
        trace.steps[0].ingest.as_ref().unwrap().relation_ids,
        vec![Uuid::from_u128(12).to_string()]
    );
    assert_eq!(trace.steps[0].observed_hits.len(), 2);
    assert_eq!(report.metrics.len(), 15);
    assert_eq!(report.metrics[10].value, 1.0);

    let storage_context = StorageContext {
        namespace: trace.run.isolated_namespace(),
        run_id: operation_run_id,
        snapshot_id: None,
        timeout_ms: None,
    };
    let video = store
        .get_sync(
            StoreGetRequest {
                record_id: Uuid::from_u128(10),
                version: None,
            },
            storage_context.clone(),
        )
        .unwrap();
    assert!(matches!(
        video.item.content,
        ContentRef::Blob { ref media_type, .. } if media_type == "video/mp4"
    ));
    let relation_page = store
        .relations_sync(
            RelationQuery {
                start_ids: vec![Uuid::from_u128(10)],
                relation_types: vec![RelationType::Supports],
                direction: TraversalDirection::Outgoing,
                limit: 8,
                cursor: None,
                visibility_cutoff_ms: None,
                snapshot_id: None,
            },
            storage_context,
        )
        .unwrap();
    assert_eq!(relation_page.relations[0].id, Uuid::from_u128(12));

    let state = store
        .get_sync(
            StoreGetRequest {
                record_id: Uuid::from_u128(11),
                version: None,
            },
            StorageContext {
                namespace: trace.run.isolated_namespace(),
                run_id: operation_run_id,
                snapshot_id: None,
                timeout_ms: None,
            },
        )
        .unwrap();
    assert!(matches!(state.item.content, ContentRef::Json { .. }));
}

#[tokio::test(flavor = "current_thread")]
async fn opened_jsonl_source_rejects_an_in_place_change_at_eof_without_completing_trace() {
    let directory = tempdir().unwrap();
    let dataset_path = directory.path().join("dataset.jsonl");
    let trace_path = directory.path().join("trace.jsonl");
    write_jsonl(&dataset_path, &sample("before"));
    let source = JsonlDataset::open(&dataset_path).unwrap();
    write_jsonl(&dataset_path, &sample("after"));
    let (method, _ingest_calls, retrieve_calls) = CountingMethod::new();
    let run = BenchmarkRun::new(
        "changed-dataset",
        manifest("counting", Some(source.content_digest().unwrap())),
        1,
    );
    let context =
        pi_agent_benchmark::BenchmarkRunContext::open(&trace_path, run, Uuid::from_u128(103))
            .unwrap();
    let result = RetrievalRunner::new(method, source, context).run().await;
    assert!(matches!(
        result,
        Err(RunnerError::Dataset(DatasetError::Source(message)))
            if message.contains("changed")
    ));
    assert_eq!(retrieve_calls.load(Ordering::SeqCst), 1);
    let trace = read_run(&trace_path, "changed-dataset").unwrap();
    assert!(!trace.complete);
    assert!(trace.steps.is_empty());
}
