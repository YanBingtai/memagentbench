use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    sync::Arc,
};

use pi_agent_benchmark::{
    read_run, AgentObservation, BenchmarkManifest, BenchmarkRun, MethodManifest, ObservationPart,
    TerminalReason,
};
use pi_agent_benchmark_evaluator::{ReferenceCatalog, RetrievalEvaluator, SampleReference};
use pi_agent_benchmark_runner::{
    AgentFuture, AgentStepExecutor, AgentStepInput, AgentStepOutput, DatasetSample,
    RetrievalRunner, VecDataset,
};
use pi_agent_memory::{
    ContentRef, JsonlProjectionStore, MemoryBudget, MemoryItem, MemoryKind, MemoryProvenance,
    MemoryQuery, MemoryRecord, MemoryScope, Modality, ProjectionMemoryMethod, StorageContext,
    StoreWriteRequest,
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
    let namespace = "bench/smoke".to_string();
    let operation_run_id = Uuid::from_u128(1);
    let record_id = Uuid::from_u128(2);

    let store = Arc::new(JsonlProjectionStore::open(&projection_path)?);
    store.write_sync(
        StoreWriteRequest {
            records: vec![MemoryRecord {
                item: MemoryItem {
                    id: record_id,
                    version: 1,
                    kind: MemoryKind::Semantic,
                    scope: MemoryScope::Shared,
                    modality: Modality::Text,
                    content: ContentRef::Text {
                        text: "Rust memory benchmark".to_string(),
                    },
                    namespace: namespace.clone(),
                    episode_id: None,
                    subject: None,
                    source: "smoke-fixture".to_string(),
                    metadata: BTreeMap::new(),
                    abstraction_level: 0,
                    event_time: None,
                    ingest_time_ms: 90,
                    schema_version: 1,
                },
                episode: None,
                evidence: Vec::new(),
                provenance: MemoryProvenance {
                    producer: "smoke-fixture".to_string(),
                    source_id: "sample-0".to_string(),
                    source_event_id: None,
                    parent_record_ids: Vec::new(),
                    method_version: Some("fixture-1".to_string()),
                },
            }],
            relations: Vec::new(),
            idempotency_key: "fixture-1".to_string(),
            source_revision: Some(1),
        },
        StorageContext {
            namespace: namespace.clone(),
            run_id: operation_run_id,
            snapshot_id: None,
            timeout_ms: None,
        },
    )?;

    let method = ProjectionMemoryMethod::new(store, "projection", "1")?;
    let mut manifest = BenchmarkManifest::new(
        "retrieval.smoke",
        &namespace,
        MethodManifest {
            method_id: "projection".to_string(),
            version: "1".to_string(),
            config: BTreeMap::new(),
        },
        "dataset-v1",
        Some(100),
    );
    let references = ReferenceCatalog::for_benchmark(
        &manifest,
        vec![SampleReference {
            sample_id: "sample-0".to_string(),
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
        episode_id: None,
        kinds: Vec::new(),
        modalities: Vec::new(),
        time_range: None,
        visibility_cutoff_ms: None,
        graph: None,
        budget: MemoryBudget {
            max_hits: 1,
            ..MemoryBudget::default()
        },
    };
    let run = BenchmarkRun::new("smoke-run", manifest, 1);
    let context =
        pi_agent_benchmark::BenchmarkRunContext::open(&trace_path, run, operation_run_id)?;
    let dataset = VecDataset::new(vec![DatasetSample {
        sample_id: "sample-0".to_string(),
        observation: AgentObservation {
            parts: vec![ObservationPart::Text("找到 Rust 记忆".to_string())],
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
    report.append_to_trace(&trace_path)?;
    let evaluated = read_run(&trace_path, &summary.run_id)?;

    assert_eq!(summary.step_count, 1);
    assert_eq!(summary.retrieval_error_count, 0);
    assert_eq!(summary.agent_error_count, 0);
    assert_eq!(evaluated.steps[0].observed_hits.len(), 1);
    assert_eq!(evaluated.metrics.len(), 10);
    assert_eq!(evaluated.metrics, report.metrics);
    println!(
        "trace={} steps={} metrics={} answer={}",
        trace_path.display(),
        evaluated.steps.len(),
        evaluated.metrics.len(),
        evaluated.steps[0]
            .agent_answer
            .as_deref()
            .unwrap_or("<none>")
    );
    Ok(())
}
