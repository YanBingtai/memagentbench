//! Sequential execution of a model-provided tool batch.
//!
//! This module owns batch order, terminal decisions, hook/result normalization,
//! and transcript persistence. Admission and timeout mechanics live in sibling
//! modules so this control flow remains easy to audit.

mod execution;
mod hooks;

use std::time::{Duration, Instant};

use crate::{
    events::{AgentEvent, RunContext},
    message::{Message, ToolCall},
    model::ChatModel,
};
use pi_agent_runtime::{EventLedger, RunId, RuntimeCoordinator};

use super::results::PersistToolResultRequest;
use super::runtime::commit_tool_result;
use super::{
    preflight_model_tool_call, unix_timestamp_ms, AgentError, AgentLoop, CancellationToken,
    RunResult, RunStats,
};

use execution::{ToolExecutionOutcome, ToolExecutionRequest};
use hooks::{AfterToolHookOutcome, AfterToolHookRequest};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn try_run_sequential_tool_batch(
        &self,
        context: RunContext,
        step: usize,
        tool_calls: &[ToolCall],
        assistant_message: Message,
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
        mut turn_tool_results: Vec<Message>,
        stats: &mut RunStats,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
        runtime_run_id: RunId,
        runtime: &mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    ) -> Result<Option<RunResult>, AgentError> {
        let mut preflight = tool_calls
            .iter()
            .map(|tool_call| preflight_model_tool_call(&self.tools, tool_call))
            .collect::<Vec<_>>()
            .into_iter();
        let mut all_tools_terminate = true;
        let mut last_tool_message = None;
        for (tool_index, tool_call) in tool_calls.iter().enumerate() {
            let (arguments, prepared_call, validation_error) = preflight
                .next()
                .expect("one preflight result should exist for every tool call");
            let tool_started_at = Instant::now();
            self.emit(AgentEvent::ToolStarted {
                context,
                step,
                call: tool_call.clone(),
            });
            let tool_timeout = Duration::from_secs(self.policy.tool_timeout_secs);
            let now_ms = unix_timestamp_ms();
            let ToolExecutionOutcome {
                mut tool_result,
                runtime_action,
                mut pending_runtime_terminal,
                stop_after_tool,
            } = self
                .admit_and_execute_tool(ToolExecutionRequest {
                    context,
                    step,
                    tool_calls,
                    tool_index,
                    tool_call,
                    arguments: arguments.clone(),
                    prepared_call,
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
                })
                .await?;
            let AfterToolHookOutcome {
                content,
                is_error,
                stop: stop_after_hook,
            } = self
                .apply_after_tool_hook(AfterToolHookRequest {
                    context,
                    step,
                    tool_call,
                    arguments: &arguments,
                    tool_result: &mut tool_result,
                    deadline,
                    cancellation,
                    skip: stop_after_tool.is_some(),
                })
                .await;
            tool_result.content = content;
            tool_result.is_error = is_error;
            tool_result = self.tools.normalize_result(tool_result);
            if let Some(grant) = pending_runtime_terminal.take() {
                commit_tool_result(runtime, grant, &tool_result)?;
            }
            let tool_message = self
                .persist_tool_result(PersistToolResultRequest {
                    context,
                    step,
                    tool_call,
                    result: &tool_result,
                    runtime_identity: runtime_action,
                    started_at: tool_started_at,
                    messages,
                    transcript,
                    turn_tool_results: &mut turn_tool_results,
                })
                .await?;
            if !tool_result.terminate {
                all_tools_terminate = false;
            }
            last_tool_message = Some(tool_message);
            if let Some((error, reason)) = stop_after_tool.or(stop_after_hook) {
                self.append_cancelled_tool_results(
                    context,
                    step,
                    tool_calls,
                    tool_index + 1,
                    messages,
                    transcript,
                    None,
                    &reason,
                )
                .await?;
                self.emit(AgentEvent::TurnFinished {
                    context,
                    step,
                    message: assistant_message,
                    tool_results: turn_tool_results,
                });
                return Err(error);
            }
        }
        if all_tools_terminate {
            self.emit(AgentEvent::TurnFinished {
                context,
                step,
                message: assistant_message,
                tool_results: turn_tool_results,
            });
            let (input_tokens, output_tokens, total_tokens) = stats.model_usage();
            return Ok(Some(RunResult {
                messages: std::mem::take(messages),
                final_message: last_tool_message
                    .expect("a non-empty tool batch produces a tool message"),
                steps: step,
                input_tokens,
                output_tokens,
                total_tokens,
            }));
        }
        self.emit(AgentEvent::TurnFinished {
            context,
            step,
            message: assistant_message,
            tool_results: turn_tool_results,
        });
        Ok(None)
    }
}
