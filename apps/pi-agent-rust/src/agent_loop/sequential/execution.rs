use std::time::Duration;

use pi_agent_runtime::{
    canonical_json_digest, ActionId, ActionProposal, AdmissionRequest, Budget, CommitGrant,
    EventLedger, ResourceScope, RiskClass, RunId, RuntimeCoordinator,
};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    events::{AgentEvent, RunContext},
    message::{Message, ToolCall},
    model::ChatModel,
    tools::{PreparedToolCall, ToolExecutionContext, ToolResult},
};

use super::super::tool_batch::{await_tool_hook, ToolHookWait};
use super::super::{
    timeout_error, timeout_millis, tool_result_has_unknown_outcome, unix_timestamp_ms, AgentError,
    AgentLoop, BeforeToolDecision, CancellationToken, RunStats, ToolHookContext,
};

pub(super) struct ToolExecutionOutcome {
    pub(super) tool_result: ToolResult,
    pub(super) runtime_action: Option<(Uuid, Uuid)>,
    pub(super) pending_runtime_terminal: Option<CommitGrant>,
    pub(super) stop_after_tool: Option<(AgentError, String)>,
}

enum SequentialExecutionOutcome {
    Completed(Result<ToolResult, crate::tools::ToolError>),
    ToolTimedOut,
    RunTimedOut(Duration),
    Cancelled,
}

pub(super) struct ToolExecutionRequest<'a> {
    pub(super) context: RunContext,
    pub(super) step: usize,
    pub(super) tool_calls: &'a [ToolCall],
    pub(super) tool_index: usize,
    pub(super) tool_call: &'a ToolCall,
    pub(super) arguments: Value,
    pub(super) prepared_call: Option<crate::tools::PreparedToolCall>,
    pub(super) validation_error: Option<String>,
    pub(super) tool_timeout: Duration,
    pub(super) now_ms: u64,
    pub(super) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(super) cancellation: &'a CancellationToken,
    pub(super) runtime_run_id: RunId,
    pub(super) messages: &'a mut Vec<Message>,
    pub(super) transcript: &'a mut Vec<Message>,
    pub(super) stats: &'a mut RunStats,
    pub(super) runtime: &'a mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
}

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn admit_and_execute_tool(
        &self,
        request: ToolExecutionRequest<'_>,
    ) -> Result<ToolExecutionOutcome, AgentError> {
        let ToolExecutionRequest {
            context,
            step,
            tool_calls,
            tool_index,
            tool_call,
            arguments,
            mut prepared_call,
            validation_error,
            tool_timeout,
            now_ms,
            deadline,
            cancellation,
            runtime_run_id,
            messages,
            transcript,
            stats,
            runtime,
        } = request;

        let mut stop_after_tool: Option<(AgentError, String)> = None;
        let mut runtime_action: Option<(Uuid, Uuid)> = None;
        // Keep a known terminal action open until the final result
        // has passed the after-tool hook and output normalization.
        // Unknown outcomes are committed immediately above because
        // they must remain pending for external reconciliation.
        let mut pending_runtime_terminal: Option<CommitGrant> = None;
        let tool_result = if let Some(reason) = validation_error {
            ToolResult::error(reason)
        } else {
            let before_decision = match &self.before_tool_hook {
                Some(hook) => {
                    let hook_result = await_tool_hook(
                        hook(ToolHookContext {
                            run: context,
                            step,
                            call: tool_call.clone(),
                            arguments: arguments.clone(),
                        }),
                        deadline,
                        cancellation,
                    )
                    .await;
                    match hook_result {
                        ToolHookWait::Completed(result) => {
                            result.unwrap_or_else(|error| BeforeToolDecision::Block {
                                reason: format!("before_tool_call hook failed: {error}"),
                            })
                        }
                        ToolHookWait::TimedOut(run_timeout) => {
                            let reason = format!(
                                "before_tool_call hook timed out after {} milliseconds",
                                timeout_millis(run_timeout)
                            );
                            self.append_cancelled_tool_results(
                                context, step, tool_calls, tool_index, messages, transcript, None,
                                &reason,
                            )
                            .await?;
                            return Err(timeout_error(run_timeout));
                        }
                        ToolHookWait::Cancelled => {
                            let reason = "before_tool_call hook cancelled by the caller";
                            self.append_cancelled_tool_results(
                                context, step, tool_calls, tool_index, messages, transcript, None,
                                reason,
                            )
                            .await?;
                            return Err(AgentError::RunCancelled);
                        }
                    }
                }
                None => BeforeToolDecision::Allow,
            };
            if cancellation.is_cancelled() {
                let reason = "Tool execution cancelled by the caller";
                self.append_cancelled_tool_results(
                    context, step, tool_calls, tool_index, messages, transcript, None, reason,
                )
                .await?;
                return Err(AgentError::RunCancelled);
            }
            match before_decision {
                BeforeToolDecision::Block { reason } => ToolResult::error(format!(
                    "Tool blocked: {reason}"
                ))
                .with_details(serde_json::json!({
                    "status": "rejected",
                    "executed": false,
                })),
                BeforeToolDecision::Terminate { reason } => {
                    let reason = format!("Tool execution terminated by policy: {reason}");
                    stop_after_tool = Some((
                        AgentError::ToolTerminated {
                            reason: reason.clone(),
                        },
                        reason.clone(),
                    ));
                    ToolResult::error(reason)
                        .with_details(serde_json::json!({
                            "status": "rejected",
                            "executed": false,
                        }))
                        .terminating()
                }
                BeforeToolDecision::Allow => {
                    let request = AdmissionRequest {
                        proposal: ActionProposal {
                            run_id: runtime_run_id,
                            action_id: ActionId::new(),
                            base_revision: runtime.current_revision(),
                            tool_name: tool_call.function.name.clone(),
                            arguments: arguments.clone(),
                            arguments_digest: canonical_json_digest(&arguments),
                            risk: self
                                .tools
                                .risk_class(&tool_call.function.name)
                                .unwrap_or(RiskClass::Unknown),
                            idempotency_key: format!(
                                "{}:{}:{}:{}:{}",
                                runtime_run_id.as_uuid(),
                                context.turn_id,
                                step,
                                tool_index,
                                tool_call.id
                            ),
                            created_at_ms: now_ms,
                            expires_at_ms: Some(
                                now_ms.saturating_add(timeout_millis(tool_timeout)),
                            ),
                        },
                        resource_scope: ResourceScope::new([format!(
                            "tool:{}",
                            tool_call.function.name
                        )]),
                        budget: Budget {
                            wall_time_ms: Some(timeout_millis(tool_timeout)),
                            tokens: None,
                            tool_calls: Some(1),
                        },
                    };
                    match runtime.admit(request, now_ms) {
                        Err(error) => ToolResult::error(format!("Tool admission failed: {error}"))
                            .with_details(json!({
                                "status": "rejected",
                                "executed": false,
                            })),
                        Ok(grant) => {
                            runtime_action = Some((grant.action_id.as_uuid(), grant.grant_id));
                            let prepared = prepared_call
                                .take()
                                .expect("a call reaching execution must be prepared");
                            let tool_context = ToolExecutionContext::new(cancellation.clone())
                                .with_output_limit(self.tools.output_limit())
                                .with_parent_identity(
                                    context.run_id,
                                    context.turn_id,
                                    step,
                                    tool_call.id.clone(),
                                )
                                .with_delegation_depth(self.delegation_depth)
                                .with_update_sink({
                                    let event_sink = self.event_sink.clone();
                                    let call_id = tool_call.id.clone();
                                    let tool_name = tool_call.function.name.clone();
                                    move |update| {
                                        if let Some(sink) = &event_sink {
                                            sink.emit(AgentEvent::ToolUpdate {
                                                context,
                                                step,
                                                call_id: call_id.clone(),
                                                tool_name: tool_name.clone(),
                                                content: update.content,
                                                details: update.details,
                                            });
                                        }
                                    }
                                });
                            let execution = self
                                .execute_prepared_tool(
                                    prepared,
                                    tool_context,
                                    tool_timeout,
                                    deadline,
                                    cancellation.clone(),
                                )
                                .await;
                            match execution {
                                SequentialExecutionOutcome::Completed(Ok(output)) => {
                                    if tool_result_has_unknown_outcome(&output) {
                                        let reason = format!(
                                            "Tool execution reported an uncertain outcome: {}",
                                            output.content
                                        );
                                        let action_id = grant.action_id.as_uuid();
                                        let tool_name = grant.tool_name.clone();
                                        let mut details = output.details.clone();
                                        if let Value::Object(fields) = &mut details {
                                            fields.insert(
                                                "status".to_string(),
                                                Value::String("unknown".to_string()),
                                            );
                                            fields
                                                .insert("executed".to_string(), Value::Bool(true));
                                        } else {
                                            details = serde_json::json!({
                                                "status": "unknown",
                                                "executed": true,
                                            });
                                        }
                                        runtime
                                            .mark_unknown(
                                                grant,
                                                reason.clone(),
                                                unix_timestamp_ms(),
                                            )
                                            .map_err(|error| {
                                                AgentError::Runtime(error.to_string())
                                            })?;
                                        stats.has_unknown_action = true;
                                        stop_after_tool = Some((
                                            AgentError::RuntimeActionUnknown {
                                                action_id,
                                                tool_name,
                                                reason: reason.clone(),
                                            },
                                            reason.clone(),
                                        ));
                                        ToolResult {
                                            content: reason,
                                            details,
                                            is_error: true,
                                            terminate: false,
                                        }
                                    } else {
                                        pending_runtime_terminal = Some(grant);
                                        output
                                    }
                                }
                                SequentialExecutionOutcome::Completed(Err(error)) => {
                                    pending_runtime_terminal = Some(grant);
                                    ToolResult::error(format!("Tool execution failed: {error}"))
                                }
                                SequentialExecutionOutcome::RunTimedOut(run_timeout) => {
                                    let reason = format!(
                                        "Tool execution cancelled because the agent run timed out after {} milliseconds",
                                        timeout_millis(run_timeout)
                                    );
                                    let runtime_identity =
                                        (grant.action_id.as_uuid(), grant.grant_id);
                                    runtime
                                        .mark_unknown(grant, reason.clone(), unix_timestamp_ms())
                                        .map_err(|error| AgentError::Runtime(error.to_string()))?;
                                    stats.has_unknown_action = true;
                                    self.append_cancelled_tool_results(
                                        context,
                                        step,
                                        tool_calls,
                                        tool_index,
                                        messages,
                                        transcript,
                                        Some(runtime_identity),
                                        &reason,
                                    )
                                    .await?;
                                    return Err(timeout_error(run_timeout));
                                }
                                SequentialExecutionOutcome::Cancelled => {
                                    let reason = "Tool execution cancelled by the caller";
                                    let runtime_identity =
                                        (grant.action_id.as_uuid(), grant.grant_id);
                                    runtime
                                        .mark_unknown(grant, reason, unix_timestamp_ms())
                                        .map_err(|error| AgentError::Runtime(error.to_string()))?;
                                    stats.has_unknown_action = true;
                                    self.append_cancelled_tool_results(
                                        context,
                                        step,
                                        tool_calls,
                                        tool_index,
                                        messages,
                                        transcript,
                                        Some(runtime_identity),
                                        reason,
                                    )
                                    .await?;
                                    return Err(AgentError::RunCancelled);
                                }
                                SequentialExecutionOutcome::ToolTimedOut => {
                                    let reason = format!(
                                        "Tool execution timed out after {} seconds; outcome is unknown",
                                        tool_timeout.as_secs()
                                    );
                                    let action_id = grant.action_id.as_uuid();
                                    let tool_name = grant.tool_name.clone();
                                    runtime
                                        .mark_unknown(grant, reason.clone(), unix_timestamp_ms())
                                        .map_err(|error| AgentError::Runtime(error.to_string()))?;
                                    stats.has_unknown_action = true;
                                    stop_after_tool = Some((
                                        AgentError::RuntimeActionUnknown {
                                            action_id,
                                            tool_name,
                                            reason: reason.clone(),
                                        },
                                        reason.clone(),
                                    ));
                                    ToolResult {
                                        content: reason,
                                        details: serde_json::json!({
                                            "status": "unknown",
                                            "executed": true,
                                        }),
                                        is_error: true,
                                        terminate: false,
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        Ok(ToolExecutionOutcome {
            tool_result,
            runtime_action,
            pending_runtime_terminal,
            stop_after_tool,
        })
    }

    async fn execute_prepared_tool(
        &self,
        prepared: PreparedToolCall,
        tool_context: ToolExecutionContext,
        tool_timeout: Duration,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: CancellationToken,
    ) -> SequentialExecutionOutcome {
        match deadline {
            Some((deadline, run_timeout)) => tokio::select! {
                result = tokio::time::timeout_at(
                    deadline,
                    tokio::time::timeout(
                        tool_timeout,
                        self.tools.execute_prepared_result_with_context(
                            prepared,
                            tool_context.clone(),
                        ),
                    ),
                ) => match result {
                    Ok(Ok(result)) => SequentialExecutionOutcome::Completed(result),
                    Ok(Err(_)) => SequentialExecutionOutcome::ToolTimedOut,
                    Err(_) => SequentialExecutionOutcome::RunTimedOut(run_timeout),
                },
                _ = cancellation.cancelled() => SequentialExecutionOutcome::Cancelled,
            },
            None => tokio::select! {
                result = tokio::time::timeout(
                    tool_timeout,
                    self.tools.execute_prepared_result_with_context(prepared, tool_context),
                ) => match result {
                    Ok(result) => SequentialExecutionOutcome::Completed(result),
                    Err(_) => SequentialExecutionOutcome::ToolTimedOut,
                },
                _ = cancellation.cancelled() => SequentialExecutionOutcome::Cancelled,
            },
        }
    }
}
