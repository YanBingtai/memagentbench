use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    sync::Arc,
};

use pi_agent_benchmark::{
    read_run, AgentObservation, BenchmarkManifest, BenchmarkRun, DatasetManifest, MethodManifest,
    ObservationPart, TerminalReason,
};
use pi_agent_benchmark_evaluator::{ContinualEvaluator, ReferenceCatalog, SampleReference};
use pi_agent_benchmark_runner::{
    AgentFuture, AgentStepExecutor, AgentStepInput, AgentStepOutput, ContinualDatasetSample,
    ContinualEpisodeRunner, JsonlContinualDataset,
};
use pi_agent_memory::{
    ContentRef, IngestRequest, JsonlProjectionStore, MemoryBudget, MemoryItem, MemoryKind,
    MemoryQuery, MemoryScope, Modality, ProjectionMemoryMethod,
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
    let namespace = "bench/continual".to_string();
    let operation_run_id = Uuid::from_u128(1);
    let record_id = Uuid::from_u128(2);

    let method = ProjectionMemoryMethod::new(
        Arc::new(JsonlProjectionStore::open(projection_path)?),
        "projection",
        "1",
    )?;
    let mut manifest = BenchmarkManifest::new(
        "continual.smoke",
        &namespace,
        MethodManifest {
            method_id: "projection".into(),
            version: "1".into(),
            config: BTreeMap::new(),
        },
        "dataset-v1",
        Some(100),
    );
    let query = MemoryQuery {
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
    };
    let sample = ContinualDatasetSample {
        sample: pi_agent_benchmark_runner::DatasetSample {
            sample_id: "sample-0".into(),
            observation: AgentObservation {
                parts: vec![ObservationPart::Text("在线记忆 smoke".into())],
            },
            query,
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
                scope: MemoryScope::Agent,
                modality: Modality::Text,
                content: ContentRef::Text {
                    text: "Rust continual memory".into(),
                },
                namespace: String::new(),
                episode_id: None,
                subject: None,
                source: "smoke".into(),
                metadata: BTreeMap::new(),
                abstraction_level: 0,
                event_time: None,
                ingest_time_ms: 1,
                schema_version: 1,
            }],
            relations: Vec::new(),
            idempotency_key: "sample-0-ingest".into(),
        },
    };

    let dataset_path = directory.path().join("dataset.jsonl");
    fs::write(
        &dataset_path,
        format!("{}\n", serde_json::to_string(&sample)?),
    )?;
    let dataset = JsonlContinualDataset::open(&dataset_path)?;
    manifest.bind_dataset(DatasetManifest::new(
        "continual-smoke",
        "dataset-v1",
        dataset.content_digest()?,
        vec![Modality::Text],
    ))?;
    let references = ReferenceCatalog::for_benchmark(
        &manifest,
        vec![SampleReference {
            sample_id: "sample-0".into(),
            relevant_record_ids: BTreeSet::from([record_id.to_string()]),
        }],
    )?;
    manifest.reference_digest = Some(references.digest()?);
    let evaluator = ContinualEvaluator::new(1, references)?;
    manifest.metrics = evaluator.metric_specs();
    let run = BenchmarkRun::new("continual-smoke", manifest, 1);
    let context =
        pi_agent_benchmark::BenchmarkRunContext::open(&trace_path, run, operation_run_id)?;
    let summary = ContinualEpisodeRunner::new(method, dataset, context)
        .run_with_agent(DeterministicAgent)
        .await?;
    let trace = read_run(&trace_path, &summary.run_id)?;
    let report = evaluator.evaluate(&trace)?;
    report.append_to_trace(&trace_path)?;
    let evaluated = read_run(&trace_path, &summary.run_id)?;

    assert_eq!(summary.step_count, 1);
    assert_eq!(summary.ingest_error_count, 0);
    assert_eq!(evaluated.steps[0].observed_hits.len(), 1);
    assert_eq!(evaluated.metrics.len(), 8);
    assert_eq!(evaluated.metrics, report.metrics);
    println!(
        "trace={} steps={} ingest_success_rate={} ingest_latency_ms={}",
        trace_path.display(),
        evaluated.steps.len(),
        report.metrics[4].value,
        report.metrics[7].value,
    );
    Ok(())
}
