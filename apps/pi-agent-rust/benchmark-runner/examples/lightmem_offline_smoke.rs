//! Run the offline benchmark lifecycle against a provider-neutral LightMem HTTP service.
//!
//! The service must expose the common memory contract, for example the local
//! `lightmem_protocol_gateway.py`.  This example intentionally has no fallback
//! provider: a connection failure is part of the experiment result.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    time::Duration,
};

use pi_agent_benchmark::{
    read_builds, read_run, AgentObservation, BenchmarkLifecycle, BenchmarkManifest, BenchmarkRun,
    MethodManifest, ObservationPart, TerminalReason,
};
use pi_agent_benchmark_evaluator::{
    OfflineBuildThenQueryEvaluator, ReferenceCatalog, SampleReference,
};
use pi_agent_benchmark_runner::{
    DatasetSample, OfflineBuildThenQueryRunner, OfflineDatasetSample, VecOfflineDataset,
};
use pi_agent_memory::{
    ContentRef, MemoryBudget, MemoryItem, MemoryKind, MemoryMethod, MemoryProvenance, MemoryQuery,
    MemoryRecord, MemoryScope, Modality, RecordIngestRequest,
};
use pi_agent_memory_remote::RemoteMemoryMethod;
use tempfile::tempdir;
use uuid::Uuid;

fn sample(sample_id: &str, episode_id: &str, terminal: bool) -> DatasetSample {
    DatasetSample {
        sample_id: sample_id.to_owned(),
        observation: AgentObservation {
            parts: vec![ObservationPart::Text(format!(
                "LightMem observation: {episode_id}"
            ))],
        },
        query: MemoryQuery {
            input: Some(ContentRef::Text {
                text: "What durable memory belongs to this episode?".to_owned(),
            }),
            namespace: String::new(),
            scope: Some(MemoryScope::Agent),
            subject: None,
            episode_id: Some(episode_id.to_owned()),
            kinds: vec![MemoryKind::Episodic],
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
                    kind: MemoryKind::Episodic,
                    scope: MemoryScope::Agent,
                    modality: Modality::Text,
                    content: ContentRef::Text {
                        text: content.to_owned(),
                    },
                    namespace: String::new(),
                    episode_id: Some(episode_id.to_owned()),
                    subject: None,
                    source: "pi-agent.lightmem-smoke".to_owned(),
                    metadata: BTreeMap::new(),
                    abstraction_level: 0,
                    event_time: None,
                    ingest_time_ms: 1,
                    schema_version: 1,
                },
                episode: None,
                evidence: Vec::new(),
                provenance: MemoryProvenance {
                    producer: "pi-agent.lightmem-smoke".to_owned(),
                    source_id: episode_id.to_owned(),
                    source_event_id: None,
                    parent_record_ids: Vec::new(),
                    method_version: Some("lightmem.text@0.1.0".to_owned()),
                },
            }],
            relations: Vec::new(),
            idempotency_key: format!("{sample_id}-build"),
        },
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let base_url =
        env::var("PI_AGENT_MEMORY_URL").unwrap_or_else(|_| "http://127.0.0.1:8093".to_owned());
    let timeout_secs = env::var("PI_AGENT_MEMORY_TIMEOUT_SECS")
        .unwrap_or_else(|_| "30".to_owned())
        .parse::<u64>()?;
    let provider = RemoteMemoryMethod::discover(&base_url, Duration::from_secs(timeout_secs))
        .await
        .map_err(|error| format!("LightMem provider discovery failed at {base_url}: {error}"))?;
    let provider_manifest = provider.manifest();
    let provider_config = provider.provider_config().clone();
    let namespace = "bench/lightmem-offline-smoke";
    let first_id = Uuid::from_u128(201);
    let second_id = Uuid::from_u128(202);

    let mut method_manifest = MethodManifest {
        method_id: provider_manifest.method_id.clone(),
        version: provider_manifest.version.clone(),
        config: BTreeMap::from([
            ("profile".to_owned(), "text-only".to_owned()),
            (
                "memory_url".to_owned(),
                provider.base_url().as_str().to_owned(),
            ),
            ("memory_timeout_secs".to_owned(), timeout_secs.to_string()),
        ]),
    };
    method_manifest.bind_provider_config(&provider_config);
    let mut manifest = BenchmarkManifest::new(
        "lightmem.offline.smoke",
        namespace,
        method_manifest,
        "lightmem-offline-smoke-v1",
        None,
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
    manifest.seal_config();

    let directory = tempdir()?;
    let trace_path = directory.path().join("lightmem-offline-trace.jsonl");
    let run = BenchmarkRun::new("lightmem-offline-smoke-run", manifest, 1);
    let context = pi_agent_benchmark::BenchmarkRunContext::open(&trace_path, run, Uuid::new_v4())?;
    let dataset = VecOfflineDataset::new(vec![
        offline_sample(
            "sample-0",
            "episode-0",
            first_id,
            "first LightMem memory",
            false,
        ),
        offline_sample(
            "sample-1",
            "episode-1",
            second_id,
            "second LightMem memory",
            true,
        ),
    ]);
    let summary = OfflineBuildThenQueryRunner::new(provider, dataset, context)
        .run()
        .await?;
    let trace = read_run(&trace_path, &summary.run_id)?;
    let builds = read_builds(&trace_path, &summary.run_id)?;
    if summary.ingest_error_count != 0 || summary.retrieval_error_count != 0 {
        return Err(format!(
            "LightMem smoke recorded ingest_errors={} retrieval_errors={}",
            summary.ingest_error_count, summary.retrieval_error_count
        )
        .into());
    }
    if trace.steps.len() != 2 || builds.len() != 2 {
        return Err(format!(
            "LightMem smoke expected two build/query steps, got steps={} builds={}",
            trace.steps.len(),
            builds.len()
        )
        .into());
    }
    for (step, expected_record_id) in trace
        .steps
        .iter()
        .zip([first_id.to_string(), second_id.to_string()])
    {
        if !step
            .observed_hits
            .iter()
            .any(|hit| hit.source_record_ids.contains(&expected_record_id))
        {
            return Err(format!(
                "LightMem smoke did not retrieve source record {expected_record_id} for sample {}",
                step.sample_id
            )
            .into());
        }
    }
    let report = evaluator.evaluate_path(&trace_path, &summary.run_id)?;
    report.append_to_trace(&trace_path)?;

    println!(
        "lightmem_offline_smoke_ok method={} steps={} builds={} metrics={} url={}",
        trace.run.manifest.method.method_id,
        trace.steps.len(),
        builds.len(),
        report.metrics.len(),
        base_url,
    );
    Ok(())
}
