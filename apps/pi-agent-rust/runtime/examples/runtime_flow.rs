use pi_agent_runtime::{
    canonical_json_digest, ActionId, ActionProposal, AdmissionPolicy, AdmissionRequest, Budget,
    EventSequence, InjectionDecision, JsonlLedger, MemoryCandidate, MemoryHookContext,
    MemoryProjection, MemoryRevision, PolicyRevision, ResourceScope, Revision, RiskClass, RunId,
    RuntimeCoordinator,
};
use serde_json::json;
use std::{
    error::Error,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

fn main() -> Result<(), Box<dyn Error>> {
    let (ledger_path, remove_ledger) = match std::env::args().nth(1) {
        Some(path) => (PathBuf::from(path), false),
        None => (
            std::env::temp_dir().join(format!("pi-agent-runtime-smoke-{}.jsonl", Uuid::new_v4())),
            true,
        ),
    };
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .min(u64::MAX as u128) as u64;
    let run_id = RunId::new();
    let policy = AdmissionPolicy::deny_all()
        .allow_tool("memory.search")
        .allow_resources(ResourceScope::new(["memory:provider:text"]))
        .with_budget_limit(Budget {
            wall_time_ms: Some(5_000),
            tokens: Some(1_024),
            tool_calls: Some(1),
        });
    let ledger = JsonlLedger::open(&ledger_path)?;
    let mut coordinator = RuntimeCoordinator::new(run_id, policy.clone(), ledger);
    let memory_context = MemoryHookContext {
        run_id,
        turn_id: None,
        namespace: "benchmark/text".to_owned(),
        current_revision: coordinator.current_revision(),
        memory_revision: MemoryRevision::INITIAL,
        policy_revision: PolicyRevision::INITIAL,
        now_ms,
        max_tokens: 128,
        max_candidates: 4,
        max_projection_bytes: 4 * 1024,
    };
    let memory_proposal = pi_agent_runtime::MemoryProposal {
        proposal_id: uuid::Uuid::new_v4(),
        run_id,
        namespace: memory_context.namespace.clone(),
        base_revision: memory_context.current_revision,
        memory_revision: memory_context.memory_revision,
        policy_revision: memory_context.policy_revision,
        candidates: vec![MemoryCandidate {
            candidate_id: uuid::Uuid::new_v4(),
            projection: MemoryProjection {
                projection_id: uuid::Uuid::new_v4(),
                source_memory_id: uuid::Uuid::new_v4(),
                source_version: 1,
                content: "runtime admission is recorded before the provider call".to_owned(),
                evidence_refs: Vec::new(),
                provenance: "runtime-flow:example".to_owned(),
                token_cost: 14,
            },
            score: 0.95,
            confidence: 0.9,
            expires_at_ms: Some(now_ms.saturating_add(5_000)),
        }],
        created_at_ms: now_ms,
        expires_at_ms: Some(now_ms.saturating_add(5_000)),
        retrieval_summary: None,
    };
    let memory_receipt = coordinator.evaluate_memory_proposal_with_receipt(
        &memory_proposal,
        &memory_context,
        now_ms,
    )?;
    assert!(matches!(
        memory_receipt.decision,
        InjectionDecision::Inject { .. }
    ));
    let request = AdmissionRequest {
        proposal: ActionProposal {
            run_id,
            action_id: ActionId::new(),
            base_revision: Revision::INITIAL,
            tool_name: "memory.search".to_owned(),
            arguments: json!({"query": "runtime vertical slice"}),
            arguments_digest: canonical_json_digest(&json!({"query": "runtime vertical slice"})),
            risk: RiskClass::ReadOnly,
            idempotency_key: format!("{}:memory.search:1", run_id.as_uuid()),
            created_at_ms: now_ms,
            expires_at_ms: Some(now_ms.saturating_add(5_000)),
        },
        resource_scope: ResourceScope::new(["memory:provider:text"]),
        budget: Budget {
            wall_time_ms: Some(500),
            tokens: Some(256),
            tool_calls: Some(1),
        },
    };

    let result = coordinator.execute(request, now_ms, |grant, proposal| {
        Ok::<_, std::convert::Infallible>(json!({
            "grant_id": grant.grant_id,
            "tool": grant.tool_name,
            "arguments": proposal.arguments,
            "retrieved": ["runtime event ledger", "admission policy"],
        }))
    })?;
    coordinator.close()?;

    let reopened = JsonlLedger::open(&ledger_path)?;
    let mut recovered = RuntimeCoordinator::recover_latest(policy, reopened)?
        .expect("a completed run should be recoverable from a non-empty ledger");
    assert_eq!(recovered.run_id(), run_id);
    let events = recovered
        .ledger()
        .scan_after(EventSequence::INITIAL)?
        .into_iter()
        .filter(|event| event.draft.run_id == run_id)
        .count();
    assert_eq!(events, 3);
    println!("{}", serde_json::to_string_pretty(&result)?);
    println!(
        "memory_decision_sequence={}",
        memory_receipt.event_sequence.value()
    );
    println!("ledger={}", ledger_path.display());
    println!("current_run_event_count={events}");
    println!(
        "recovered_revision={}",
        recovered.current_revision().value()
    );
    recovered.close()?;
    if remove_ledger {
        let _ = std::fs::remove_file(&ledger_path);
        let _ = std::fs::remove_file(format!("{}.lock", ledger_path.display()));
    }
    Ok(())
}
