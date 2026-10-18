use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    agent_loop::AgentLoop,
    events::AgentEvent,
    tools::{ToolError, ToolExecutionContext, ToolResult},
};

use super::{
    arguments::{parse_arguments, validate_arguments},
    progress::{
        child_details, child_error_status_with_outcome, subagent_progress, truncate_output,
    },
    SubagentTool,
};

impl SubagentTool {
    pub(super) async fn run_child(
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
