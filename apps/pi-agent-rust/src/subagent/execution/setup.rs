use crate::{
    agent_loop::AgentLoop,
    model::ModelClient,
    tools::{ToolError, ToolExecutionContext},
};

use super::super::{
    arguments::{parse_arguments, validate_arguments},
    SubagentTool,
};

/// Values shared by child construction, execution, and result metadata.
pub(super) struct ChildSpec {
    pub(super) task: String,
    pub(super) role: Option<String>,
    pub(super) model: String,
    pub(super) max_steps: usize,
    pub(super) depth: u16,
}

/// A configured child loop and the metadata needed to report its result.
pub(super) struct PreparedChild {
    pub(super) agent: AgentLoop<ModelClient>,
    pub(super) spec: ChildSpec,
}

pub(super) fn prepare(
    tool: &SubagentTool,
    raw_arguments: &serde_json::Value,
    context: Option<&ToolExecutionContext>,
) -> Result<PreparedChild, ToolError> {
    let arguments = parse_arguments(raw_arguments)?;
    validate_arguments(&arguments)?;

    let task = arguments.task.trim().to_owned();
    let max_steps = arguments
        .max_steps
        .unwrap_or(tool.policy.max_steps)
        .min(tool.policy.max_steps);
    if max_steps == 0 {
        return Err(ToolError::Subagent {
            message: "delegated task must have at least one model step".to_string(),
        });
    }

    let mut policy = tool.policy;
    policy.max_steps = max_steps;
    policy.run_timeout_secs = Some(tool.max_runtime_secs.max(1));

    let depth = context
        .map(ToolExecutionContext::delegation_depth)
        .unwrap_or(0)
        .saturating_add(1);
    let role = arguments
        .role
        .as_deref()
        .map(str::trim)
        .filter(|role| !role.is_empty())
        .map(str::to_owned);

    let agent = AgentLoop::new_with_policy(
        tool.client.clone(),
        tool.tools.clone(),
        tool.model.clone(),
        tool.api_key.clone(),
        tool.options,
        policy,
    )
    .map_err(|error| ToolError::Subagent {
        message: error.to_string(),
    })?;
    let agent = agent.with_delegation_depth(depth);
    let agent = match role.as_deref() {
        Some(role) => agent.with_system_prompt(format!(
            "You are a delegated {role} subagent. Follow the parent task scope and return a concise result."
        )),
        None => agent,
    };
    let agent = match &tool.skills {
        Some(skills) => agent.with_skills(skills.clone()),
        None => agent,
    };

    Ok(PreparedChild {
        agent,
        spec: ChildSpec {
            task,
            role,
            model: tool.model.clone(),
            max_steps,
            depth,
        },
    })
}
