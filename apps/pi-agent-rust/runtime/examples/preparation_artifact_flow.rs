use pi_agent_artifacts::{ArtifactStore, FsArtifactStore, PutArtifactRequest};
use pi_agent_runtime::{
    canonical_json_digest, ActionId, AdmissionPolicy, Budget, InMemoryLedger,
    PreparationEffectClass, PreparationPlan, PreparationWorkResult, ResourceScope, Revision, RunId,
    RuntimeCoordinator,
};
use serde_json::json;
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store_root = env::temp_dir().join(format!(
        "pi-agent-preparation-artifacts-{}",
        uuid::Uuid::new_v4()
    ));
    let store = FsArtifactStore::new(&store_root)?;
    let run_id = RunId::new();
    let arguments = json!({"episode": "episode-7"});
    let plan = PreparationPlan::new(
        run_id,
        ActionId::new(),
        Revision::INITIAL,
        "memory.retrieve_video",
        canonical_json_digest(&arguments),
        PreparationEffectClass::ReadOnly,
        ResourceScope::new(["memory:provider:video"]),
        Budget {
            wall_time_ms: Some(1_000),
            tokens: Some(256),
            tool_calls: Some(1),
        },
        0.95,
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
    let proposal = pi_agent_runtime::ActionProposal {
        run_id,
        action_id: coordinator.preparations().get(plan_id).unwrap().action_id,
        base_revision: Revision::INITIAL,
        tool_name: "memory.retrieve_video".to_owned(),
        arguments: arguments.clone(),
        arguments_digest: canonical_json_digest(&arguments),
        risk: pi_agent_runtime::RiskClass::ReadOnly,
        idempotency_key: "runtime/artifact-flow".to_owned(),
        created_at_ms: 1_000,
        expires_at_ms: Some(5_000),
    };

    coordinator.execute_preparation(plan_id, 1_001, 1_100, |_plan| {
        let artifact = store.put(PutArtifactRequest {
            namespace: "runtime/run-1".to_owned(),
            idempotency_key: Some("episode-7-video".to_owned()),
            media_type: "video/mp4".to_owned(),
            data: b"video bytes".to_vec(),
            codec: Some("h264".to_owned()),
            schema: Some("episode-video-v1".to_owned()),
            stream_key: Some("camera/front".to_owned()),
        })?;
        Ok::<_, pi_agent_artifacts::ArtifactStoreError>(PreparationWorkResult::Ready {
            artifacts: vec![artifact],
            evidence: Vec::new(),
        })
    })?;
    let consumed =
        coordinator.consume_preparation_with_resolver(plan_id, &proposal, 1_200, |artifact| {
            store.stat(artifact)
        })?;

    assert_eq!(
        consumed.status,
        pi_agent_runtime::PreparationStatus::Consumed
    );
    assert_eq!(consumed.artifacts.len(), 1);
    println!(
        "preparation_artifact_flow_ok uri={} status={:?}",
        consumed.artifacts[0].uri, consumed.status
    );
    drop(store);
    let _ = fs::remove_dir_all(store_root);
    Ok(())
}
