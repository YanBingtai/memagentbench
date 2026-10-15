use pi_agent_runtime::{
    canonical_json_digest, ActionId, AdmissionPolicy, Budget, InMemoryLedger,
    PreparationEffectClass, PreparationOutcome, PreparationPlan, PreparationWorkResult,
    ResourceScope, Revision, RunId, RuntimeCoordinator,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let run_id = RunId::new();
    let arguments = json!({"query": "runtime worker"});
    let plan = PreparationPlan::new(
        run_id,
        ActionId::new(),
        Revision::INITIAL,
        "memory.search",
        canonical_json_digest(&arguments),
        PreparationEffectClass::ReadOnly,
        ResourceScope::new(["memory:provider:text"]),
        Budget {
            wall_time_ms: Some(500),
            tokens: Some(256),
            tool_calls: Some(1),
        },
        0.9,
        1_000,
        5_000,
    )?;
    let plan_id = plan.plan_id;
    let mut coordinator = RuntimeCoordinator::new(
        run_id,
        AdmissionPolicy::deny_all(),
        InMemoryLedger::default(),
    );
    coordinator.submit_preparation(plan, 1_000)?;
    let result = coordinator.execute_preparation(plan_id, 1_001, 1_200, |_plan| {
        Ok::<_, &'static str>(PreparationWorkResult::Ready {
            artifacts: Vec::new(),
            evidence: Vec::new(),
        })
    })?;

    let (ledger, policy, _, _) = coordinator.into_parts();
    let recovered = RuntimeCoordinator::recover(run_id, policy, ledger)?;
    let recovered_plan = recovered
        .preparations()
        .get(plan_id)
        .expect("worker plan should survive recovery");
    assert_eq!(
        result,
        PreparationWorkResult::Ready {
            artifacts: Vec::new(),
            evidence: Vec::new(),
        }
    );
    assert_eq!(
        recovered_plan.status,
        pi_agent_runtime::PreparationStatus::Ready
    );
    assert_eq!(recovered_plan.outcome, None::<PreparationOutcome>);
    println!(
        "preparation_worker_flow_ok plan_id={plan_id} status={:?}",
        recovered_plan.status
    );
    Ok(())
}
