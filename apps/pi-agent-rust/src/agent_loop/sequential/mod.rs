//! Ordered execution of a model-provided tool batch.
//!
//! This facade owns only batch order. The per-call lifecycle and terminal
//! paths live in sibling modules so admission, persistence, and cancellation
//! remain easy to audit independently.

mod batch;
mod execution;
mod hooks;
mod terminal;

use std::time::Duration;

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    model::ChatModel,
};
use pi_agent_runtime::{EventLedger, RunId, RuntimeCoordinator};

use super::{AgentError, AgentLoop, CancellationToken, RunResult, RunStats};
use batch::{preflight_calls, BatchState, ToolStepRequest};

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
        let preflight = preflight_calls(&self.tools, tool_calls);
        let mut state = BatchState::new();
        for (tool_index, (tool_call, preflight)) in tool_calls.iter().zip(preflight).enumerate() {
            let mut outcome = batch::execute_tool(
                self,
                ToolStepRequest {
                    context,
                    step,
                    tool_calls,
                    tool_index,
                    tool_call,
                    preflight,
                    deadline,
                    cancellation,
                    runtime_run_id,
                    messages,
                    transcript,
                    turn_tool_results: &mut turn_tool_results,
                    stats,
                    runtime,
                },
            )
            .await?;
            let stop = outcome.stop.take();
            state.record(outcome);
            if let Some((error, reason)) = stop {
                return terminal::stop_after_tool(
                    self,
                    terminal::StopRequest {
                        context,
                        step,
                        tool_calls,
                        first_pending: tool_index + 1,
                        assistant_message,
                        messages,
                        transcript,
                        turn_tool_results,
                        cancellation_reason: reason,
                        error,
                    },
                )
                .await;
            }
        }

        terminal::finish(
            self,
            terminal::FinishRequest {
                context,
                step,
                assistant_message,
                messages,
                turn_tool_results,
                stats,
                state,
            },
        )
    }
}
