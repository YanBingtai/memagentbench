use pi_agent_runtime::{
    canonical_json_digest, ActionId, ActionProposal, Budget, PreparationEffectClass,
    PreparationManager, PreparationPlan, ResourceScope, Revision, RiskClass, RunId,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let run_id = RunId::new();
    let action_id = ActionId::new();
    let revision = Revision::new(7);
    let arguments = json!({"format": "pdf"});
    let digest = canonical_json_digest(&arguments);

    let plan = PreparationPlan::new(
        run_id,
        action_id,
        revision,
        "render_pdf",
        digest.clone(),
        PreparationEffectClass::Draft,
        ResourceScope::new(["workspace:draft"]),
        Budget {
            wall_time_ms: Some(500),
            tokens: Some(128),
            tool_calls: Some(1),
        },
        0.85,
        1_000,
        5_000,
    )?;
    let plan_id = plan.plan_id;

    let mut manager = PreparationManager::new(8)?;
    manager.submit(plan, 1_000)?;
    manager.start(plan_id, run_id, revision, 1_100)?;
    manager.mark_ready(plan_id, run_id, revision, 1_200)?;

    let proposal = ActionProposal {
        run_id,
        action_id,
        base_revision: revision,
        tool_name: "render_pdf".to_owned(),
        arguments_digest: digest,
        arguments,
        risk: RiskClass::ReadOnly,
        idempotency_key: "run-7/action-1".to_owned(),
        created_at_ms: 1_200,
        expires_at_ms: Some(5_000),
    };
    manager.consume_ready(plan_id, &proposal, 1_300)?;

    println!(
        "preparation_flow_ok plan_id={} status={:?}",
        plan_id,
        manager
            .get(plan_id)
            .expect("consumed plan should remain visible")
            .status
    );
    Ok(())
}
