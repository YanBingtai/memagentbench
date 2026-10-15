use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    sync::Arc,
};

use pi_agent_benchmark::{
    read_run, AgentObservation, BenchmarkLifecycle, BenchmarkManifest, BenchmarkRun,
    MethodManifest, ObservationPart, TerminalReason,
};
use pi_agent_benchmark_evaluator::{
    OfflineBuildThenQueryEvaluator, ReferenceCatalog, SampleReference,
};
use pi_agent_benchmark_runner::{
    DatasetSample, OfflineBuildThenQueryRunner, OfflineDatasetSample, VecOfflineDataset,
};
use pi_agent_memory::{
    ContentRef, JsonlProjectionStore, MemoryBudget, MemoryItem, MemoryKind, MemoryProvenance,
    MemoryQuery, MemoryRecord, MemoryScope, Modality, ProjectionMemoryMethod, RecordIngestRequest,
};
use tempfile::tempdir;
use uuid::Uuid;

fn sample(sample_id: &str, episode_id: &str, terminal: bool) -> DatasetSample {
    DatasetSample {
        sample_id: sample_id.to_owned(),
        observation: AgentObservation {
            parts: vec![ObservationPart::Text(format!(
                "offline observation: {episode_id}"
            ))],
        },
        query: MemoryQuery {
            input: None,
            namespace: String::new(),
            scope: Some(MemoryScope::Agent),
            subject: None,
            episode_id: Some(episode_id.to_owned()),
            kinds: vec![MemoryKind::Semantic],
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
        terminal,
        terminal_reason: terminal.then_some(TerminalReason::Success),
    }
}

fn offline_sample(
    sample_id: &str,
    episode_id: &str,
    record_id: Uuid,
    content: &str,
    terminal: bool,
) -> OfflineDatasetSample {
    OfflineDatasetSample {
        sample: sample(sample_id, episode_id, terminal),
        build: RecordIngestRequest {
            records: vec![MemoryRecord {
                item: MemoryItem {
                    id: record_id,
                    version: 1,
                    kind: MemoryKind::Semantic,
                    scope: MemoryScope::Agent,
                    modality: Modality::Text,
                    content: ContentRef::Text {
                        text: content.to_owned(),
                    },
                    namespace: String::new(),
                    episode_id: Some(episode_id.to_owned()),
                    subject: None,
                    source: "offline-build-smoke".to_owned(),
                    metadata: BTreeMap::new(),
                    abstraction_level: 0,
                    event_time: None,
                    ingest_time_ms: 1,
                    schema_version: 1,
                },
                episode: None,
                evidence: Vec::new(),
                provenance: MemoryProvenance {
                    producer: "offline-build-smoke".to_owned(),
                    source_id: episode_id.to_owned(),
                    source_event_id: None,
                    parent_record_ids: Vec::new(),
                    method_version: Some("fixture-1".to_owned()),
                },
            }],
            relations: Vec::new(),
            idempotency_key: format!("{sample_id}-build"),
        },
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let trace_path = directory.path().join("offline-trace.jsonl");
    let projection_path = directory.path().join("projection.jsonl");
    let namespace = "bench/offline-smoke";
    let first_id = Uuid::from_u128(101);
    let second_id = Uuid::from_u128(102);

    let method = ProjectionMemoryMethod::new(
        Arc::new(JsonlProjectionStore::open(&projection_path)?),
        "projection-offline",
        "1",
    )?;
    let dataset = VecOfflineDataset::new(vec![
        offline_sample("sample-0", "episode-0", first_id, "first memory", false),
        offline_sample("sample-1", "episode-1", second_id, "second memory", true),
    ]);

    let mut manifest = BenchmarkManifest::new(
        "offline.build_then_query.smoke",
        namespace,
        MethodManifest {
            method_id: "projection-offline".to_owned(),
            version: "1".to_owned(),
            config: BTreeMap::new(),
        },
        "offline-smoke-v1",
        Some(100),
    );
    manifest.lifecycle = BenchmarkLifecycle::OfflineBuildThenQuery;
    let references = ReferenceCatalog::for_benchmark(
        &manifest,
        vec![
            SampleReference {
                sample_id: "sample-0".to_owned(),
                relevant_record_ids: BTreeSet::from([first_id.to_string()]),
            },
            SampleReference {
                sample_id: "sample-1".to_owned(),
                relevant_record_ids: BTreeSet::from([second_id.to_string()]),
            },
        ],
    )?;
    manifest.reference_digest = Some(references.digest()?);
    let evaluator = OfflineBuildThenQueryEvaluator::new(1, references)?;
    manifest.metrics = evaluator.metric_specs();

    let run = BenchmarkRun::new("offline-smoke-run", manifest, 1);
    let context =
        pi_agent_benchmark::BenchmarkRunContext::open(&trace_path, run, Uuid::from_u128(1))?;
    let summary = OfflineBuildThenQueryRunner::new(method, dataset, context)
        .run()
        .await?;
    let trace = read_run(&trace_path, &summary.run_id)?;
    let report = evaluator.evaluate_path(&trace_path, &summary.run_id)?;
    report.append_to_trace(&trace_path)?;

    assert_eq!(summary.step_count, 2);
    assert_eq!(summary.retrieval_error_count, 0);
    assert_eq!(trace.steps.len(), 2);
    assert_eq!(
        trace.steps[0].observed_hits[0].record_id,
        first_id.to_string()
    );
    assert_eq!(
        trace.steps[1].observed_hits[0].record_id,
        second_id.to_string()
    );
    assert_eq!(report.metrics.len(), 8);
    println!(
        "offline_build_smoke_ok steps={} builds=2 metrics={} trace={}",
        trace.steps.len(),
        report.metrics.len(),
        trace_path.display(),
    );
    Ok(())
}
