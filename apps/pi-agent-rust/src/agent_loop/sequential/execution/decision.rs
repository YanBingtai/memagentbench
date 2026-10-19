use pi_agent_runtime::{
    canonical_json_digest, ActionId, ActionProposal, AdmissionRequest, Budget, CommitGrant,
    ResourceScope, RiskClass,
};
use serde_json::json;

use crate::{model::ChatModel, tools::ToolResult};

use super::super::super::{timeout_millis, AgentLoop};
use super::types::ToolExecutionRequest;

pub(super) fn admit_tool<M>(
    agent: &AgentLoop<M>,
    request: &mut ToolExecutionRequest<'_>,
) -> Result<CommitGrant, ToolResult>
where
    M: ChatModel,
{
    let tool_name = request.tool_call.function.name.clone();
    let now_ms = request.now_ms;
    let admission = AdmissionRequest {
        proposal: ActionProposal {
            run_id: request.runtime_run_id,
            action_id: ActionId::new(),
            base_revision: request.runtime.current_revision(),
            tool_name: tool_name.clone(),
            arguments: request.arguments.clone(),
            arguments_digest: canonical_json_digest(&request.arguments),
            risk: agent
                .tools
                .risk_class(&tool_name)
                .unwrap_or(RiskClass::Unknown),
            idempotency_key: format!(
                "{}:{}:{}:{}:{}",
                request.runtime_run_id.as_uuid(),
                request.context.turn_id,
                request.step,
                request.tool_index,
                request.tool_call.id
            ),
            created_at_ms: now_ms,
            expires_at_ms: Some(now_ms.saturating_add(timeout_millis(request.tool_timeout))),
        },
        resource_scope: ResourceScope::new([format!("tool:{tool_name}")]),
        budget: Budget {
            wall_time_ms: Some(timeout_millis(request.tool_timeout)),
            tokens: None,
            tool_calls: Some(1),
        },
    };
    request.runtime.admit(admission, now_ms).map_err(|error| {
        ToolResult::error(format!("Tool admission failed: {error}")).with_details(json!({
            "status": "rejected",
            "executed": false,
        }))
    })
}
