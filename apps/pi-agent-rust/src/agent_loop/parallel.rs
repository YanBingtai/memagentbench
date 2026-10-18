//! Parallel execution of independent read-only tool calls.
//!
//! Admission, execution, terminal ledger commits, and transcript writes stay
//! in the same order as the original AgentLoop implementation. This module
//! only owns the parallel scheduling boundary.

use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

use pi_agent_runtime::{
    canonical_json_digest, ActionId, ActionProposal, AdmissionRequest, Budget, EventLedger,
    ResourceScope, RiskClass, RunId, RuntimeCoordinator,
};
use serde_json::{json, Value};

use crate::{
    events::{AgentEvent, RunContext},
    message::{Message, ToolCall},
    model::ChatModel,
    tools::{ToolConcurrency, ToolExecutionContext, ToolReplayPolicy, ToolResult},
};

use super::results::PersistToolResultRequest;
use super::runtime::commit_tool_result;
use super::tool_batch::{execute_parallel_tool, ParallelExecutionOutcome};
use super::{
    preflight_model_tool_call, timeout_error, timeout_millis, tool_result_has_unknown_outcome,
    unix_timestamp_ms, AgentError, AgentLoop, CancellationToken, RunStats,
};

/// Result of one parallel tool batch after all terminal writes are complete.
///
/// The parent loop uses this to preserve the existing turn lifecycle: it
/// emits `TurnFinished`, then either returns a terminal result or requests the
/// next model turn.
pub(super) struct ParallelBatchOutcome {
    pub(super) all_tools_terminate: bool,
    pub(super) last_tool_message: Option<Message>,
    pub(super) terminal_error: Option<AgentError>,
}

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn try_run_parallel_tool_batch(
        &self,
        context: RunContext,
        step: usize,
        tool_calls: &[ToolCall],
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
        turn_tool_results: &mut Vec<Message>,
        stats: &mut RunStats,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
        runtime_run_id: RunId,
        runtime: &mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    ) -> Result<Option<ParallelBatchOutcome>, AgentError> {
        if tool_calls.len() < 2
            || self.policy.max_parallel_tool_calls < 2
            || self.before_tool_hook.is_some()
            || self.after_tool_hook.is_some()
        {
            return Ok(None);
        }

        let mut prepared_calls = Vec::with_capacity(tool_calls.len());
        for tool_call in tool_calls {
            let (arguments, prepared, _validation_error) =
                preflight_model_tool_call(&self.tools, tool_call);
            let Some(prepared) = prepared else {
                return Ok(None);
            };
            if self.tools.concurrency(&tool_call.function.name) != Some(ToolConcurrency::Parallel)
                || self.tools.replay_policy(&tool_call.function.name)
                    != Some(ToolReplayPolicy::Safe)
                || self.tools.risk_class(&tool_call.function.name) != Some(RiskClass::ReadOnly)
            {
                return Ok(None);
            }
            prepared_calls.push(Some((arguments, prepared)));
        }

        let tool_timeout = Duration::from_secs(self.policy.tool_timeout_secs);
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
        let receipt = match runtime.admit_batch(requests, now_ms) {
            Ok(receipt) => receipt,
            Err(_) => return Ok(None),
        };

        let mut all_tools_terminate = true;
        let mut last_tool_message = None;
        let mut terminal_error = None;
        let max_parallel = self.policy.max_parallel_tool_calls;
        let mut chunk_start = 0;

        while chunk_start < tool_calls.len() {
            let chunk_end = (chunk_start + max_parallel).min(tool_calls.len());
            let mut tasks = tokio::task::JoinSet::new();
            for index in chunk_start..chunk_end {
                let call = &tool_calls[index];
                self.emit(AgentEvent::ToolStarted {
                    context,
                    step,
                    call: call.clone(),
                });
                let (_, prepared) = prepared_calls[index]
                    .take()
                    .expect("parallel tool preparation should be consumed once");
                let event_sink = self.event_sink.clone();
                let call_id = call.id.clone();
                let tool_name = call.function.name.clone();
                let tool_context = ToolExecutionContext::new(cancellation.clone())
                    .with_output_limit(self.tools.output_limit())
                    .with_parent_identity(context.run_id, context.turn_id, step, call.id.clone())
                    .with_delegation_depth(self.delegation_depth)
                    .with_update_sink(move |update| {
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
                    });
                let tools = self.tools.clone();
                let grant = receipt.grants[index].clone();
                let cancellation = cancellation.clone();
                let started_at = Instant::now();
                tasks.spawn(async move {
                    let execution = execute_parallel_tool(
                        tools,
                        prepared,
                        tool_context,
                        tool_timeout,
                        deadline,
                        cancellation,
                    )
                    .await;
                    (index, grant, started_at, execution)
                });
            }

            let mut completed = Vec::with_capacity(chunk_end - chunk_start);
            let mut worker_error = None;
            while let Some(joined) = tasks.join_next().await {
                match joined {
                    Ok(result) => completed.push(result),
                    Err(error) => {
                        worker_error
                            .get_or_insert_with(|| format!("parallel tool worker failed: {error}"));
                    }
                }
            }
            completed.sort_by_key(|(index, _, _, _)| *index);
            let completed_indices = completed
                .iter()
                .map(|(index, _, _, _)| *index)
                .collect::<HashSet<_>>();

            for (index, grant, started_at, execution) in completed {
                let runtime_action = Some((grant.action_id.as_uuid(), grant.grant_id));
                let mut terminal_runtime = true;
                let mut unknown_reason = None;
                let mut stop_error = None;
                let mut tool_result = match execution {
                    ParallelExecutionOutcome::Completed(Ok(output)) => {
                        if tool_result_has_unknown_outcome(&output) {
                            let reason = format!(
                                "Tool execution reported an uncertain outcome: {}",
                                output.content
                            );
                            unknown_reason = Some(reason.clone());
                            stop_error = Some(AgentError::RuntimeActionUnknown {
                                action_id: grant.action_id.as_uuid(),
                                tool_name: grant.tool_name.clone(),
                                reason: reason.clone(),
                            });
                            let mut details = output.details.clone();
                            if let Value::Object(fields) = &mut details {
                                fields.insert(
                                    "status".to_string(),
                                    Value::String("unknown".to_string()),
                                );
                                fields.insert("executed".to_string(), Value::Bool(true));
                            } else {
                                details = json!({"status": "unknown", "executed": true});
                            }
                            ToolResult {
                                content: reason,
                                details,
                                is_error: true,
                                terminate: false,
                            }
                        } else {
                            output
                        }
                    }
                    ParallelExecutionOutcome::Completed(Err(error)) => {
                        ToolResult::error(format!("Tool execution failed: {error}"))
                    }
                    ParallelExecutionOutcome::ToolTimedOut => {
                        let reason = format!(
                            "Tool execution timed out after {} seconds; outcome is unknown",
                            tool_timeout.as_secs()
                        );
                        unknown_reason = Some(reason.clone());
                        stop_error = Some(AgentError::RuntimeActionUnknown {
                            action_id: grant.action_id.as_uuid(),
                            tool_name: grant.tool_name.clone(),
                            reason: reason.clone(),
                        });
                        ToolResult::error(reason).with_details(json!({
                            "status": "unknown",
                            "executed": true,
                        }))
                    }
                    ParallelExecutionOutcome::RunTimedOut(run_timeout) => {
                        let reason = format!(
                            "Tool execution cancelled because the agent run timed out after {} milliseconds",
                            timeout_millis(run_timeout)
                        );
                        unknown_reason = Some(reason.clone());
                        stop_error = Some(timeout_error(run_timeout));
                        ToolResult::error(reason).with_details(json!({
                            "status": "unknown",
                            "executed": true,
                        }))
                    }
                    ParallelExecutionOutcome::Cancelled => {
                        let reason = "Tool execution cancelled by the caller".to_string();
                        unknown_reason = Some(reason.clone());
                        stop_error = Some(AgentError::RunCancelled);
                        ToolResult::error(reason).with_details(json!({
                            "status": "unknown",
                            "executed": true,
                        }))
                    }
                };

                if let Some(reason) = unknown_reason {
                    runtime
                        .mark_unknown(grant.clone(), reason, unix_timestamp_ms())
                        .map_err(|error| AgentError::Runtime(error.to_string()))?;
                    stats.has_unknown_action = true;
                    terminal_runtime = false;
                }
                tool_result = self.tools.normalize_result(tool_result);
                if terminal_runtime {
                    commit_tool_result(runtime, grant.clone(), &tool_result)?;
                }

                let tool_message = self
                    .persist_tool_result(PersistToolResultRequest {
                        context,
                        step,
                        tool_call: &tool_calls[index],
                        result: &tool_result,
                        runtime_identity: runtime_action,
                        started_at,
                        messages,
                        transcript,
                        turn_tool_results,
                    })
                    .await?;
                last_tool_message = Some(tool_message);
                if !tool_result.terminate {
                    all_tools_terminate = false;
                }
                if terminal_error.is_none() {
                    terminal_error = stop_error;
                }
            }

            if let Some(reason) = worker_error {
                for index in chunk_start..chunk_end {
                    if completed_indices.contains(&index) {
                        continue;
                    }
                    let grant = receipt.grants[index].clone();
                    runtime
                        .mark_unknown(grant.clone(), reason.clone(), unix_timestamp_ms())
                        .map_err(|error| AgentError::Runtime(error.to_string()))?;
                    stats.has_unknown_action = true;

                    let message_start = messages.len();
                    self.append_cancelled_tool_results(
                        context,
                        step,
                        &tool_calls[index..index + 1],
                        0,
                        messages,
                        transcript,
                        Some((grant.action_id.as_uuid(), grant.grant_id)),
                        &reason,
                    )
                    .await?;
                    if let Some(message) = messages.get(message_start).cloned() {
                        turn_tool_results.push(message.clone());
                        last_tool_message = Some(message);
                    }
                    all_tools_terminate = false;
                }
                if terminal_error.is_none() {
                    terminal_error = Some(AgentError::Runtime(reason));
                }
            }

            chunk_start = chunk_end;
            if terminal_error.is_some() {
                if chunk_start < tool_calls.len() {
                    let reason = "parallel tool batch stopped before remaining calls executed";
                    for grant in receipt.grants.iter().skip(chunk_start) {
                        runtime
                            .fail(grant.clone(), reason, unix_timestamp_ms())
                            .map_err(|error| AgentError::Runtime(error.to_string()))?;
                    }
                    let pending = self
                        .append_unexecuted_tool_results(
                            context,
                            step,
                            &tool_calls[chunk_start..],
                            messages,
                            transcript,
                            reason,
                            "cancelled",
                        )
                        .await?;
                    turn_tool_results.extend(pending);
                }
                break;
            }
        }

        Ok(Some(ParallelBatchOutcome {
            all_tools_terminate,
            last_tool_message,
            terminal_error,
        }))
    }
}
