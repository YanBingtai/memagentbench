use pi_agent_runtime::{
    canonical_json_digest, ActionId, ActionProposal, AdmissionPolicy, AdmissionRequest, Budget,
    EventSequence, JsonlLedger, ResourceScope, Revision, RiskClass, RunId, RuntimeCoordinator,
};
use serde_json::json;
use std::{
    error::Error,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn main() -> Result<(), Box<dyn Error>> {
    let ledger_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!(
                "pi-agent-runtime-smoke-{}.jsonl",
                std::process::id()
            ))
        });
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
    let mut recovered = RuntimeCoordinator::recover(run_id, policy, reopened)?;
    let events = recovered
        .ledger()
        .scan_after(EventSequence::INITIAL)?
        .into_iter()
        .filter(|event| event.draft.run_id == run_id)
        .count();
    println!("{}", serde_json::to_string_pretty(&result)?);
    println!("ledger={}", ledger_path.display());
    println!("current_run_event_count={events}");
    println!(
        "recovered_revision={}",
        recovered.current_revision().value()
    );
    recovered.close()?;
    Ok(())
}
