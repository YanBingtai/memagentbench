//! A bounded child-agent tool for delegated, isolated work.
//!
//! A child receives a fresh transcript and an explicitly cloned tool set. The
//! parent gets only a bounded summary, so intermediate child messages do not
//! pollute the parent session. The initial implementation deliberately keeps
//! the child non-recursive: the registry is cloned before this tool is
//! registered.

use serde::Deserialize;
use serde_json::json;

use pi_agent_runtime::RiskClass;

use crate::{
    agent_loop::AgentLoop,
    config::RunPolicy,
    message::ToolDefinition,
    model::{ChatOptions, ModelClient},
    tools::{Tool, ToolError, ToolFuture, ToolRegistry},
};

const DEFAULT_MAX_OUTPUT_BYTES: usize = 32 * 1024;
const MAX_TASK_BYTES: usize = 16 * 1024;
const MAX_ROLE_BYTES: usize = 256;

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
}

impl SubagentTool {
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
        Self {
            client,
            model: model.into(),
            api_key,
            options,
            // A child must never inherit its own delegation capability. This
            // keeps the first version bounded even if a caller accidentally
            // passes the parent registry instead of a pre-filtered clone.
            tools: tools.without("spawn_subagent"),
            policy,
            max_runtime_secs,
            max_output_bytes,
        }
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

    fn execute<'a>(&'a self, arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: SubagentArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "spawn_subagent".to_string(),
                        source,
                    }
                })?;
            let task = arguments.task.trim();
            if task.is_empty() {
                return Err(ToolError::Subagent {
                    message: "delegated task must not be empty".to_string(),
                });
            }
            if task.len() > MAX_TASK_BYTES {
                return Err(ToolError::Subagent {
                    message: format!(
                        "delegated task is too large ({} bytes, limit {} bytes)",
                        task.len(),
                        MAX_TASK_BYTES
                    ),
                });
            }
            if let Some(role) = arguments.role.as_deref() {
                if role.trim().len() > MAX_ROLE_BYTES {
                    return Err(ToolError::Subagent {
                        message: format!(
                            "delegated role is too large ({} bytes, limit {} bytes)",
                            role.trim().len(),
                            MAX_ROLE_BYTES
                        ),
                    });
                }
            }

            let max_steps = arguments
                .max_steps
                .unwrap_or(self.policy.max_steps)
                .min(self.policy.max_steps);
            if max_steps == 0 {
                return Err(ToolError::Subagent {
                    message: "delegated task must have at least one model step".to_string(),
                });
            }

            let mut policy = self.policy;
            policy.max_steps = max_steps;
            policy.run_timeout_secs = Some(self.max_runtime_secs.max(1));
            let prompt = match arguments.role.as_deref().map(str::trim) {
                Some(role) if !role.is_empty() => {
                    format!("You are a delegated {role} subagent.\n\nTask:\n{task}")
                }
                _ => task.to_string(),
            };
            let child = AgentLoop::new_with_policy(
                self.client.clone(),
                self.tools.clone(),
                self.model.clone(),
                self.api_key.clone(),
                self.options,
                policy,
            )
            .map_err(|error| ToolError::Subagent {
                message: error.to_string(),
            })?;
            let result = child
                .run(prompt)
                .await
                .map_err(|error| ToolError::Subagent {
                    message: error.to_string(),
                })?;
            let summary = match result.final_message.content {
                Some(content) => content,
                None => serde_json::to_string(&result.final_message).unwrap_or_default(),
            };

            Ok(serde_json::to_string(&json!({
                "status": "completed",
                "steps": result.steps,
                "summary": truncate_output(summary, self.max_output_bytes),
            }))
            .unwrap_or_else(|_| "{\"status\":\"completed\"}".to_string()))
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SubagentArguments {
    task: String,
    role: Option<String>,
    max_steps: Option<usize>,
}

fn truncate_output(mut text: String, max_bytes: usize) -> String {
    if max_bytes == 0 {
        return String::new();
    }
    if text.len() <= max_bytes {
        return text;
    }
    let marker = "…";
    if max_bytes < marker.len() {
        let mut end = max_bytes.min(text.len());
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        return text;
    }
    let limit = max_bytes.saturating_sub(marker.len());
    let mut end = limit.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text.push_str(marker);
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, tools::ToolContext};

    #[test]
    fn advertises_a_bounded_delegation_tool() {
        let config = Config::default();
        let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
            .expect("read_file should register");
        let client = ModelClient::new("http://localhost:1", std::time::Duration::from_secs(1))
            .expect("endpoint should be valid");
        let tool = SubagentTool::new(
            client,
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            tools,
            config.run_policy(),
            30,
            1024,
        );
        let definition = tool.definition();

        assert_eq!(definition.function.name, "spawn_subagent");
        assert_eq!(definition.function.parameters["required"], json!(["task"]));
        assert_eq!(
            definition.function.parameters["additionalProperties"],
            false
        );
    }
}
