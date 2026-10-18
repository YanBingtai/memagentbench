use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use pi_agent_benchmark::{
    read_run, AgentObservation, BenchmarkManifest, BenchmarkRun, BenchmarkRunContext,
    MethodManifest, ObservationPart, TerminalReason,
};
use pi_agent_benchmark_evaluator::{
    ReferenceCatalog, ReferenceCatalogManifest, RuntimeLinkedEvaluator, SampleReference,
};
use pi_agent_benchmark_runner::{DatasetSample, NoMemoryMethod, RetrievalRunner, VecDataset};
use pi_agent_memory::{ContentRef, MemoryBudget, MemoryQuery};
use pi_agent_runtime::{
    EventSequence, JsonlLedger, MemoryCandidate, MemoryHookContext, MemoryProjection,
    MemoryProposal,
};
use pi_agent_rust::{
    AgentLoop, ChatFuture, ChatModel, ChatOptions, ChatRequest, Message, ToolRegistry,
};
use tempfile::tempdir;
use uuid::Uuid;

#[derive(Clone)]
struct RecordingModel {
    saw_injected_memory: Arc<AtomicBool>,
}

impl ChatModel for RecordingModel {
    fn complete<'a>(&'a self, request: ChatRequest<'a>) -> ChatFuture<'a> {
        let saw_injected_memory = Arc::clone(&self.saw_injected_memory);
        let injected = request.messages.iter().any(|message| {
            message
                .content
                .as_deref()
                .is_some_and(|content| content.contains("remembered memory"))
        });
        Box::pin(async move {
            saw_injected_memory.store(injected, Ordering::Release);
            Ok(Message::assistant(Some("offline answer".to_owned()), None))
        })
    }
}

fn sample() -> DatasetSample {
    DatasetSample {
        sample_id: "sample-1".to_owned(),
        observation: AgentObservation {
            parts: vec![ObservationPart::Text("visible observation".to_owned())],
        },
        query: MemoryQuery {
            input: Some(ContentRef::Text {
                text: "what should I remember?".to_owned(),
            }),
            namespace: "bench/runtime-link".to_owned(),
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
        },
        session_id: None,
        episode_id: None,
        terminal: true,
        terminal_reason: Some(TerminalReason::Success),
    }
}

fn proposal(request: &MemoryHookContext) -> MemoryProposal {
    MemoryProposal {
        proposal_id: Uuid::from_u128(0x200),
        run_id: request.run_id,
        namespace: request.namespace.clone(),
        base_revision: request.current_revision,
        memory_revision: request.memory_revision,
        policy_revision: request.policy_revision,
        candidates: vec![MemoryCandidate {
            candidate_id: Uuid::from_u128(0x201),
            projection: MemoryProjection {
                projection_id: Uuid::from_u128(0x202),
                source_memory_id: Uuid::from_u128(0x203),
                source_version: 1,
                content: "remembered memory".to_owned(),
                evidence_refs: Vec::new(),
                provenance: "offline-test".to_owned(),
                token_cost: 5,
            },
            score: 1.0,
            confidence: 1.0,
            expires_at_ms: None,
        }],
        created_at_ms: request.now_ms,
        expires_at_ms: None,
        retrieval_summary: None,
    }
}

#[tokio::test]
async fn linked_agent_run_is_persisted_and_evaluated_offline() {
    let directory = tempdir().expect("temporary directory");
    let trace_path = directory.path().join("trace.jsonl");
    let ledger_path = directory.path().join("runtime.jsonl");
    let operation_run_id = Uuid::from_u128(0x100);

    let mut manifest = BenchmarkManifest::new(
        "runtime-link",
        "bench/runtime-link",
        MethodManifest {
            method_id: "no-memory".to_owned(),
            version: "1".to_owned(),
            config: BTreeMap::new(),
        },
        "dataset-1",
        None,
    );
    let references = ReferenceCatalog::for_benchmark(
        &manifest,
        vec![SampleReference {
            sample_id: "sample-1".to_owned(),
            relevant_record_ids: BTreeSet::from(["memory-never-returned".to_owned()]),
        }],
    )
    .expect("reference catalog");
    let reference_path = directory.path().join("references.json");
    references
        .write_json(&reference_path)
        .expect("reference sidecar should be durable");
    let reference_digest = references.digest().expect("reference digest");
    let evaluator = RuntimeLinkedEvaluator::new(1, references).expect("runtime evaluator");
    manifest.metrics = evaluator.metric_specs();
    manifest.reference_digest = Some(reference_digest);

    let run = BenchmarkRun::new("run-1", manifest, 1);
    let context =
        BenchmarkRunContext::open(&trace_path, run, operation_run_id).expect("benchmark context");

    let saw_injected_memory = Arc::new(AtomicBool::new(false));
    let model = RecordingModel {
        saw_injected_memory: Arc::clone(&saw_injected_memory),
    };
    let agent = AgentLoop::new(
        model,
        ToolRegistry::new(),
        "offline-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("agent loop")
    .with_runtime_ledger_file(&ledger_path)
    .with_memory_hook(
        pi_agent_rust::MemoryHookConfig {
            namespace: "bench/runtime-link".to_owned(),
            ..Default::default()
        },
        |request| Box::pin(async move { Ok(Some(proposal(&request.context))) }),
    );

    let summary = RetrievalRunner::new(NoMemoryMethod, VecDataset::new(vec![sample()]), context)
        .run_with_agent(
            pi_agent_benchmark_adapter::PiAgentExecutor::new_with_runtime(agent, operation_run_id),
        )
        .await
        .expect("offline benchmark run");
    assert_eq!(summary.step_count, 1);
    assert!(saw_injected_memory.load(Ordering::Acquire));

    let trace = read_run(&trace_path, "run-1").expect("trace should be readable");
    let link = trace.steps[0]
        .runtime_link
        .as_ref()
        .expect("step should contain runtime link");
    assert_eq!(link.runtime_run_id, operation_run_id);
    assert!(!link.turn_id.is_nil());
    let decision = link
        .memory_decision
        .as_ref()
        .expect("memory decision should be linked");
    assert_eq!(decision.decision_kind, "injected");
    assert_ne!(decision.event_sequence, 0);

    drop(evaluator);
    let references = ReferenceCatalog::read_json_for_manifest(
        &reference_path,
        &ReferenceCatalogManifest::from_benchmark(&trace.run.manifest),
    )
    .expect("reference sidecar should reload after a process boundary");
    let evaluator = RuntimeLinkedEvaluator::new(1, references).expect("reloaded evaluator");
    let report = evaluator
        .evaluate_with_ledger(&trace, &ledger_path)
        .expect("reloaded trace, reference, and runtime ledger should evaluate");
    assert_eq!(report.run_id, "run-1");

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events");
    let memory_event = events
        .iter()
        .find(|event| event.draft.kind.as_str() == "memory_injected")
        .expect("memory injection should be durable");
    assert_eq!(memory_event.draft.run_id.as_uuid(), operation_run_id);
    assert_eq!(memory_event.sequence.value(), decision.event_sequence);
}
