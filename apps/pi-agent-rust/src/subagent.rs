//! A bounded, read-only child-agent tool for delegated, isolated work.
//!
//! A child receives a fresh transcript and an explicitly cloned tool set. The
//! parent gets only a bounded summary, so intermediate child messages do not
//! pollute the parent session. The child is deliberately non-recursive and
//! read-only: delegation and every tool above `ReadOnly` are removed at
//! construction time.

use std::sync::{Arc, Mutex};

use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use pi_agent_runtime::RiskClass;

use crate::{
    agent_loop::{AgentError, AgentLoop},
    config::RunPolicy,
    events::{AgentEvent, RunOutcome},
    message::ToolDefinition,
    model::{ChatOptions, ModelClient},
    skills::SkillCatalog,
    tools::{
        Tool, ToolError, ToolExecutionContext, ToolFuture, ToolRegistry, ToolResult,
        ToolResultFuture, ToolUpdate,
    },
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

impl SubagentTool {
    async fn run_child(
        &self,
        arguments: serde_json::Value,
        context: Option<ToolExecutionContext>,
    ) -> Result<ToolResult, ToolError> {
        let arguments = parse_arguments(&arguments)?;
        validate_arguments(&arguments)?;
        let task = arguments.task.trim();

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
        let child_depth = context
            .as_ref()
            .map(ToolExecutionContext::delegation_depth)
            .unwrap_or(0)
            .saturating_add(1);
        let role = arguments
            .role
            .as_deref()
            .map(str::trim)
            .filter(|role| !role.is_empty())
            .map(str::to_owned);
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
        let child = child.with_delegation_depth(child_depth);
        let child = match role.as_deref() {
            Some(role) if !role.is_empty() => child.with_system_prompt(format!(
                "You are a delegated {role} subagent. Follow the parent task scope and return a concise result."
            )),
            _ => child,
        };
        let child = match &self.skills {
            Some(skills) => child.with_skills(skills.clone()),
            None => child,
        };
        let observed_outcome = Arc::new(Mutex::new(None));
        let observed_run_id = Arc::new(Mutex::new(None::<Uuid>));
        let progress = context.clone();
        let observed_for_sink = Arc::clone(&observed_outcome);
        let run_id_for_sink = Arc::clone(&observed_run_id);
        let child = child.with_event_sink(move |event| {
            if let AgentEvent::RunStarted { context, .. } = &event {
                *run_id_for_sink
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(context.run_id);
            }
            if let AgentEvent::RunFinished { outcome, .. } = &event {
                *observed_for_sink
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(*outcome);
            }
            if let (Some(progress), Some(update)) = (progress.as_ref(), subagent_progress(&event)) {
                progress.update(update);
            }
        });
        let cancellation = context
            .as_ref()
            .map(ToolExecutionContext::cancellation_token);
        let result = match match cancellation {
            Some(cancellation) => child.run_with_cancellation(task, cancellation).await,
            None => child.run(task).await,
        } {
            Ok(result) => result,
            Err(error) => {
                let observed = *observed_outcome
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                let child_run_id = *observed_run_id
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                let status = child_error_status_with_outcome(&error, observed);
                let message = error.to_string();
                let mut details = child_details(
                    context.as_ref(),
                    child_run_id,
                    role.as_deref(),
                    &self.model,
                    max_steps,
                    child_depth,
                    status,
                );
                if let Value::Object(fields) = &mut details {
                    fields.insert("error".to_string(), Value::String(message.clone()));
                }
                return Ok(ToolResult::error(format!("subagent {status}: {message}"))
                    .with_details(details));
            }
        };
        let child_run_id = *observed_run_id
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let summary = match result.final_message.content {
            Some(content) => content,
            None => serde_json::to_string(&result.final_message).unwrap_or_default(),
        };

        let mut details = child_details(
            context.as_ref(),
            child_run_id,
            role.as_deref(),
            &self.model,
            max_steps,
            child_depth,
            "completed",
        );
        if let Value::Object(fields) = &mut details {
            fields.insert("steps".to_string(), json!(result.steps));
        }
        Ok(ToolResult::text(truncate_output(summary, self.max_output_bytes)).with_details(details))
    }
}

fn child_details(
    context: Option<&ToolExecutionContext>,
    child_run_id: Option<Uuid>,
    role: Option<&str>,
    model: &str,
    max_steps: usize,
    depth: u16,
    status: &str,
) -> Value {
    let mut details = json!({
        "status": status,
        "child_run_id": child_run_id,
        "role": role,
        "model": model,
        "max_steps": max_steps,
        "depth": depth,
    });
    if let (Some(context), Value::Object(fields)) = (context, &mut details) {
        if let Some(parent_run_id) = context.parent_run_id() {
            fields.insert("parent_run_id".to_string(), json!(parent_run_id));
        }
        if let Some(parent_turn_id) = context.parent_turn_id() {
            fields.insert("parent_turn_id".to_string(), json!(parent_turn_id));
        }
        if let Some(parent_step) = context.parent_step() {
            fields.insert("parent_step".to_string(), json!(parent_step));
        }
        if let Some(parent_tool_call_id) = context.parent_tool_call_id() {
            fields.insert(
                "parent_tool_call_id".to_string(),
                Value::String(parent_tool_call_id.to_string()),
            );
        }
    }
    details
}

fn child_error_status(error: &AgentError) -> &'static str {
    match error {
        AgentError::RunCancelled => "cancelled",
        AgentError::RunTimedOut { .. } => "timed_out",
        AgentError::RuntimeActionUnknown { .. } => "unknown",
        _ => "failed",
    }
}

fn child_error_status_with_outcome(
    error: &AgentError,
    outcome: Option<RunOutcome>,
) -> &'static str {
    if outcome == Some(RunOutcome::Unknown) {
        "unknown"
    } else {
        child_error_status(error)
    }
}

fn subagent_progress(event: &AgentEvent) -> Option<ToolUpdate> {
    match event {
        AgentEvent::ModelStarted { step, model, .. } => Some(
            ToolUpdate::text(format!("subagent model step {step}: {model}")).with_details(json!({
                "status": "model_started",
                "step": step,
            })),
        ),
        AgentEvent::ToolStarted { call, .. } => Some(
            ToolUpdate::text(format!("subagent tool started: {}", call.function.name))
                .with_details(json!({
                    "status": "tool_started",
                    "tool": call.function.name,
                    "call_id": call.id,
                })),
        ),
        AgentEvent::ToolFinished {
            tool_name,
            is_error,
            ..
        } => Some(
            ToolUpdate::text(format!(
                "subagent tool {}: {}",
                tool_name,
                if *is_error { "failed" } else { "completed" }
            ))
            .with_details(json!({
                "status": "tool_finished",
                "tool": tool_name,
                "is_error": is_error,
            })),
        ),
        AgentEvent::RunFinished {
            steps,
            outcome,
            error,
            ..
        } => {
            let status = match outcome {
                RunOutcome::Completed => "completed",
                RunOutcome::Cancelled => "cancelled",
                RunOutcome::Failed => "failed",
                RunOutcome::Unknown => "unknown",
            };
            Some(
                ToolUpdate::text(format!("subagent {status} after {steps} step(s)")).with_details(
                    json!({
                        "status": status,
                        "steps": steps,
                        "error": error,
                    }),
                ),
            )
        }
        _ => None,
    }
}

fn parse_arguments(arguments: &serde_json::Value) -> Result<SubagentArguments, ToolError> {
    serde_json::from_value(arguments.clone()).map_err(|source| ToolError::InvalidArguments {
        name: "spawn_subagent".to_string(),
        source,
    })
}

fn validate_arguments(arguments: &SubagentArguments) -> Result<(), ToolError> {
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
    if arguments.max_steps == Some(0) {
        return Err(ToolError::Subagent {
            message: "delegated task must have at least one model step".to_string(),
        });
    }
    Ok(())
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
    use wiremock::{matchers::method, Mock, MockServer, ResponseTemplate};

    #[test]
    fn advertises_a_bounded_delegation_tool() {
        let config = Config::default();
        let mut tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
            .expect("read_file should register");
        tools
            .register_write_file(ToolContext::from_config(&config))
            .expect("write_file should register");
        tools
            .register_edit_file(ToolContext::from_config(&config))
            .expect("edit_file should register");
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
        assert_eq!(
            tool.tools.risk_class("read_file"),
            Some(RiskClass::ReadOnly)
        );
        assert_eq!(tool.tools.risk_class("write_file"), None);
        assert_eq!(tool.tools.risk_class("edit_file"), None);
    }

    #[test]
    fn truncation_respects_small_utf8_limits() {
        assert_eq!(truncate_output("你好".to_string(), 0), "");
        assert_eq!(truncate_output("你好".to_string(), 1), "");
        assert_eq!(truncate_output("你好".to_string(), 3), "…");
        assert!(truncate_output("a long result".to_string(), 5).len() <= 5);
    }

    #[test]
    fn child_details_preserve_parent_lineage() {
        let parent_run_id = Uuid::from_u128(1);
        let parent_turn_id = Uuid::from_u128(2);
        let child_run_id = Uuid::from_u128(3);
        let context = ToolExecutionContext::new(crate::CancellationToken::new())
            .with_parent_identity(parent_run_id, parent_turn_id, 4, "call-child")
            .with_delegation_depth(2);

        let details = child_details(
            Some(&context),
            Some(child_run_id),
            Some("researcher"),
            "test-model",
            3,
            3,
            "completed",
        );

        assert_eq!(details["parent_run_id"], json!(parent_run_id));
        assert_eq!(details["parent_turn_id"], json!(parent_turn_id));
        assert_eq!(details["parent_step"], json!(4));
        assert_eq!(details["parent_tool_call_id"], json!("call-child"));
        assert_eq!(details["child_run_id"], json!(child_run_id));
        assert_eq!(details["depth"], json!(3));
    }

    #[test]
    fn rejects_invalid_arguments_before_execution() {
        let config = Config::default();
        let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
            .expect("read_file should register");
        let tool = SubagentTool::new(
            ModelClient::new("http://localhost:1", std::time::Duration::from_secs(1))
                .expect("endpoint should be valid"),
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

        let empty_task = tool
            .validate_arguments(&json!({"task": "   "}))
            .expect_err("empty task should fail preflight");
        assert!(matches!(
            empty_task,
            ToolError::Subagent { message } if message.contains("must not be empty")
        ));

        let zero_steps = tool
            .validate_arguments(&json!({"task": "inspect", "max_steps": 0}))
            .expect_err("zero max_steps should fail preflight");
        assert!(matches!(
            zero_steps,
            ToolError::Subagent { message } if message.contains("at least one model step")
        ));
    }

    #[tokio::test]
    async fn role_and_task_are_sent_in_separate_messages() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"done\"}}]}\n\n"
                            .to_owned()
                            + "data: [DONE]\n\n",
                    ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let config = Config::default();
        let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
            .expect("read_file should register");
        let tool = SubagentTool::new(
            ModelClient::new(&server.uri(), std::time::Duration::from_secs(5))
                .expect("endpoint should be valid"),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            tools,
            config.run_policy(),
            5,
            1024,
        );

        tool.execute_result(json!({
            "task": "inspect the repository",
            "role": "researcher"
        }))
        .await
        .expect("child should complete");

        let requests = server
            .received_requests()
            .await
            .expect("wiremock should return received requests");
        assert_eq!(requests.len(), 1);
        let body: Value =
            serde_json::from_slice(&requests[0].body).expect("model request should be valid JSON");
        assert_eq!(body["messages"][0]["role"], "system");
        assert!(body["messages"][0]["content"]
            .as_str()
            .is_some_and(|content| content.contains("researcher")));
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "inspect the repository");
    }

    #[test]
    fn summarizes_child_lifecycle_without_forwarding_transcript() {
        let context = crate::events::RunContext::new(None);
        let update = subagent_progress(&AgentEvent::ModelStarted {
            context,
            step: 2,
            model: "qwen3.8".to_string(),
        })
        .expect("model start should produce progress");

        assert_eq!(update.details["status"], "model_started");
        assert_eq!(update.details["step"], 2);
        assert!(update.content.contains("qwen3.8"));
        assert!(subagent_progress(&AgentEvent::AssistantMessage {
            context,
            step: 2,
            message: crate::message::Message::assistant(Some("private answer".to_string()), None),
        })
        .is_none());
    }

    #[test]
    fn classifies_child_terminal_errors_for_structured_results() {
        assert_eq!(child_error_status(&AgentError::RunCancelled), "cancelled");
        assert_eq!(
            child_error_status(&AgentError::RunTimedOut { timeout_ms: 10 }),
            "timed_out"
        );
        assert_eq!(
            child_error_status(&AgentError::RuntimeActionUnknown {
                action_id: uuid::Uuid::new_v4(),
                tool_name: "write_file".to_owned(),
                reason: "executor disconnected".to_owned(),
            }),
            "unknown"
        );
        assert_eq!(child_error_status(&AgentError::EmptyPrompt), "failed");
        assert_eq!(
            child_error_status_with_outcome(&AgentError::RunCancelled, Some(RunOutcome::Unknown)),
            "unknown"
        );
        assert_eq!(
            child_error_status_with_outcome(&AgentError::RunCancelled, Some(RunOutcome::Cancelled)),
            "cancelled"
        );
    }
}
