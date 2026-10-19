//! Preflight and atomically admit a parallel tool batch.

use std::time::Duration;

use pi_agent_runtime::{
    canonical_json_digest, ActionId, ActionProposal, AdmissionBatchReceipt, AdmissionRequest,
    Budget, EventLedger, ResourceScope, RiskClass, RunId, RuntimeCoordinator,
};
use serde_json::Value;

use crate::{
    events::RunContext,
    message::ToolCall,
    model::ChatModel,
    tools::{PreparedToolCall, ToolConcurrency, ToolReplayPolicy},
};

use super::super::{preflight_model_tool_call, timeout_millis, unix_timestamp_ms, AgentLoop};

pub(super) struct ParallelAdmission {
    pub(super) prepared_calls: Vec<Option<(Value, PreparedToolCall)>>,
    pub(super) receipt: AdmissionBatchReceipt,
    pub(super) tool_timeout: Duration,
}

pub(super) fn admit_parallel_batch<M>(
    agent: &AgentLoop<M>,
    context: RunContext,
    step: usize,
    tool_calls: &[ToolCall],
    runtime_run_id: RunId,
    runtime: &mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
) -> Option<ParallelAdmission>
where
    M: ChatModel,
{
    if tool_calls.len() < 2
        || agent.policy.max_parallel_tool_calls < 2
        || agent.before_tool_hook.is_some()
        || agent.after_tool_hook.is_some()
    {
        return None;
    }

    let mut prepared_calls = Vec::with_capacity(tool_calls.len());
    for tool_call in tool_calls {
        let (arguments, prepared, _validation_error) =
            preflight_model_tool_call(&agent.tools, tool_call);
        let prepared = prepared?;
        if agent.tools.concurrency(&tool_call.function.name) != Some(ToolConcurrency::Parallel)
            || agent.tools.replay_policy(&tool_call.function.name) != Some(ToolReplayPolicy::Safe)
            || agent.tools.risk_class(&tool_call.function.name) != Some(RiskClass::ReadOnly)
        {
            return None;
        }
        prepared_calls.push(Some((arguments, prepared)));
    }

    let tool_timeout = Duration::from_secs(agent.policy.tool_timeout_secs);
    let now_ms = unix_timestamp_ms();
    let requests = prepared_calls
        .iter()
        .enumerate()
        .map(|(tool_index, prepared)| {
            let (arguments, _) = prepared
                .as_ref()
                .expect("parallel preflight entries remain available");
            AdmissionRequest {
                proposal: ActionProposal {
                    run_id: runtime_run_id,
                    action_id: ActionId::new(),
                    base_revision: runtime.current_revision(),
                    tool_name: tool_calls[tool_index].function.name.clone(),
                    arguments: arguments.clone(),
                    arguments_digest: canonical_json_digest(arguments),
                    risk: RiskClass::ReadOnly,
                    idempotency_key: format!(
                        "{}:{}:{}:{}:{}",
                        runtime_run_id.as_uuid(),
                        context.turn_id,
                        step,
                        tool_index,
                        tool_calls[tool_index].id
                    ),
                    created_at_ms: now_ms,
                    expires_at_ms: Some(now_ms.saturating_add(timeout_millis(tool_timeout))),
                },
                resource_scope: ResourceScope::new([format!(
                    "tool:{}",
                    tool_calls[tool_index].function.name
                )]),
                budget: Budget {
                    wall_time_ms: Some(timeout_millis(tool_timeout)),
                    tokens: None,
                    tool_calls: Some(1),
                },
            }
        })
        .collect::<Vec<_>>();
    let receipt = runtime.admit_batch(requests, now_ms).ok()?;

    Some(ParallelAdmission {
        prepared_calls,
        receipt,
        tool_timeout,
    })
}
