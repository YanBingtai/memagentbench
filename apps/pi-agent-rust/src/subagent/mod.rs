//! A bounded, read-only child-agent tool for delegated, isolated work.
//!
//! A child receives a fresh transcript and an explicitly cloned tool set. The
//! parent gets only a bounded summary, so intermediate child messages do not
//! pollute the parent session. The child is deliberately non-recursive and
//! read-only: delegation and every tool above `ReadOnly` are removed at
//! construction time.

mod arguments;
mod execution;
mod progress;
#[cfg(test)]
mod tests;

use pi_agent_runtime::RiskClass;
use serde_json::json;

use crate::{
    config::RunPolicy,
    message::ToolDefinition,
    model::{ChatOptions, ModelClient},
    skills::SkillCatalog,
    tools::{Tool, ToolError, ToolExecutionContext, ToolFuture, ToolRegistry, ToolResultFuture},
};

use arguments::{parse_arguments, validate_arguments};

const DEFAULT_MAX_OUTPUT_BYTES: usize = 32 * 1024;

/// Executes one delegated task with a fresh AgentLoop.
pub struct SubagentTool {
    client: ModelClient,
    model: String,
    api_key: Option<String>,
    options: ChatOptions,
    tools: ToolRegistry,
    policy: RunPolicy,
    max_runtime_secs: u64,
    max_output_bytes: usize,
    skills: Option<SkillCatalog>,
}

impl SubagentTool {
    // Keep the public constructor explicit: each dependency is an intentional
    // capability boundary for the child agent. A settings struct can be added
    // later without silently changing this compatibility surface.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: ModelClient,
        model: impl Into<String>,
        api_key: Option<String>,
        options: ChatOptions,
        tools: ToolRegistry,
        policy: RunPolicy,
        max_runtime_secs: u64,
        max_output_bytes: usize,
    ) -> Self {
        let max_runtime_secs = max_runtime_secs.max(1).min(policy.tool_timeout_secs.max(1));
        let max_output_bytes = if max_output_bytes == 0 {
            DEFAULT_MAX_OUTPUT_BYTES
        } else {
            max_output_bytes.min(DEFAULT_MAX_OUTPUT_BYTES)
        };
        let mut child_tools = tools.without("spawn_subagent");
        for definition in child_tools.definitions() {
            let name = definition.function.name;
            if child_tools.risk_class(&name) != Some(RiskClass::ReadOnly) {
                child_tools = child_tools.without(&name);
            }
        }

        Self {
            client,
            model: model.into(),
            api_key,
            options,
            tools: child_tools,
            policy,
            max_runtime_secs,
            max_output_bytes,
            skills: None,
        }
    }

    /// Give child agents the same model-visible skill catalog as their parent.
    pub fn with_skills(mut self, skills: SkillCatalog) -> Self {
        if !skills.is_empty() {
            if self.tools.exposure("read_skill").is_none() {
                self.tools
                    .register_skill_reader(skills.clone(), self.max_output_bytes)
                    .expect("the built-in read_skill tool must remain valid");
            }
            self.skills = Some(skills);
        } else {
            self.skills = None;
        }
        self
    }
}

impl Tool for SubagentTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "spawn_subagent",
            "Delegate one well-scoped task to an isolated child agent and return its bounded summary.",
            json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Complete task instructions for the child agent",
                        "minLength": 1
                    },
                    "role": {
                        "type": "string",
                        "description": "Optional role such as researcher, reviewer, or planner"
                    },
                    "max_steps": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional child step limit; the parent policy remains the upper bound"
                    }
                },
                "required": ["task"],
                "additionalProperties": false
            }),
        )
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn validate_arguments(&self, arguments: &serde_json::Value) -> Result<(), ToolError> {
        let arguments = parse_arguments(arguments)?;
        validate_arguments(&arguments)
    }

    fn execute<'a>(&'a self, arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move { Ok(self.execute_result(arguments).await?.content) })
    }

    fn execute_result<'a>(&'a self, arguments: serde_json::Value) -> ToolResultFuture<'a> {
        Box::pin(self.run_child(arguments, None))
    }

    fn execute_result_with_context<'a>(
        &'a self,
        arguments: serde_json::Value,
        context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(self.run_child(arguments, Some(context)))
    }
}
