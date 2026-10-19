//! Admission policy construction for runtime tool actions.

use pi_agent_runtime::{AdmissionPolicy, Budget, ResourceScope};

use crate::message::ToolDefinition;

pub(crate) fn runtime_policy(
    definitions: &[ToolDefinition],
    tool_timeout_secs: u64,
    allow_irreversible_tools: bool,
) -> AdmissionPolicy {
    let mut policy = AdmissionPolicy::deny_all().with_budget_limit(Budget {
        wall_time_ms: Some(tool_timeout_secs.saturating_mul(1_000)),
        tokens: None,
        tool_calls: Some(1),
    });
    if allow_irreversible_tools {
        policy = policy.allow_irreversible();
    }
    for definition in definitions {
        let name = definition.function.name.clone();
        policy = policy
            .allow_tool(name.clone())
            .allow_resources(ResourceScope::new([format!("tool:{name}")]));
    }
    policy
}
