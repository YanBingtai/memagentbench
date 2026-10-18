//! One model/tool turn and its terminal outcome.

use std::time::Duration;

use pi_agent_runtime::{EventLedger, RunId, RuntimeCoordinator};

use crate::{
    events::{AgentEvent, RunContext},
    message::{Message, ToolDefinition},
    model::ChatModel,
};

use super::{
    model_turn::{request_assistant, ModelTurnRequest},
    results::UnexecutedToolTurnRequest,
    tool_validation::validate_tool_call_batch,
    AgentError, AgentLoop, CancellationToken, RunResult, RunStats,
};

pub(super) struct TurnRequest<'a> {
    pub(super) context: RunContext,
    pub(super) step: usize,
    pub(super) definitions: &'a [ToolDefinition],
    pub(super) messages: &'a mut Vec<Message>,
    pub(super) transcript: &'a mut Vec<Message>,
    pub(super) stats: &'a mut RunStats,
    pub(super) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(super) cancellation: &'a CancellationToken,
    pub(super) runtime_run_id: RunId,
    pub(super) runtime: &'a mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
}

pub(super) enum TurnOutcome {
    Continue,
    Completed(Box<RunResult>),
}

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn run_turn(
        &self,
        request: TurnRequest<'_>,
    ) -> Result<TurnOutcome, AgentError> {
        let TurnRequest {
            context,
            step,
            definitions,
            messages,
            transcript,
            stats,
            deadline,
            cancellation,
            runtime_run_id,
            runtime,
        } = request;

        self.inject_steering_messages(messages, transcript).await?;
        stats.steps = step;
        self.emit(AgentEvent::TurnStarted { context, step });
        self.emit(AgentEvent::ModelStarted {
            context,
            step,
            model: self.model.clone(),
        });
        let mut request_messages = self
            .fit_context(context, step, messages.clone(), deadline, cancellation)
            .await?;
        self.prepare_memory_context(
            context,
            step,
            runtime_run_id,
            &mut request_messages,
            runtime,
            deadline,
            cancellation,
        )
        .await?;
        request_messages = self
            .fit_context(context, step, request_messages, deadline, cancellation)
            .await?;
        let mut assistant = request_assistant(
            ModelTurnRequest {
                client: &self.client,
                model: &self.model,
                api_key: self.api_key.as_deref(),
                options: self.options,
                definitions,
                context,
                step,
                messages: &request_messages,
                deadline,
                cancellation,
                max_retries: self.policy.max_model_retries,
                retry_backoff_ms: self.policy.model_retry_backoff_ms,
            },
            stats,
            &|event| self.emit(event),
        )
        .await?;
        let has_tool_calls = assistant
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty());
        let assistant_message = assistant.clone();
        if has_tool_calls {
            let tool_calls = assistant_message
                .tool_calls
                .as_deref()
                .expect("non-empty tool calls must be present");
            let validation_error = if tool_calls.len() > self.policy.max_tool_calls_per_step {
                Some(AgentError::TooManyToolCalls {
                    step,
                    count: tool_calls.len(),
                    limit: self.policy.max_tool_calls_per_step,
                })
            } else {
                validate_tool_call_batch(step, tool_calls).err()
            };
            if let Some(error) = validation_error {
                // A malformed assistant batch is an ephemeral protocol error.
                // Do not append it to the durable transcript: session loading
                // requires every assistant tool call to have a valid, ordered
                // result, and this batch has not reached execution yet.
                self.emit(AgentEvent::AssistantMessage {
                    context,
                    step,
                    message: assistant_message.clone(),
                });
                self.emit(AgentEvent::TurnFinished {
                    context,
                    step,
                    message: assistant_message,
                    tool_results: Vec::new(),
                });
                return Err(error);
            }
        }
        messages.push(assistant_message.clone());
        transcript.push(assistant_message.clone());
        if let Some(session) = &self.session {
            session.append(&assistant_message).await?;
        }
        self.emit(AgentEvent::AssistantMessage {
            context,
            step,
            message: assistant_message.clone(),
        });
        let mut turn_tool_results = Vec::new();

        if assistant_message.stop_reason.as_deref() == Some("content_filter") {
            let tool_calls = assistant_message.tool_calls.clone().unwrap_or_default();
            self.append_unexecuted_tool_turn(UnexecutedToolTurnRequest {
                context,
                step,
                assistant_message,
                tool_calls: &tool_calls,
                messages,
                transcript,
                reason:
                    "model output was blocked by content filter before the tool call could execute"
                        .to_string(),
                status: "rejected",
            })
            .await?;
            return Err(AgentError::ModelTerminated {
                reason: "content_filter".to_string(),
            });
        }

        if matches!(
            assistant_message.stop_reason.as_deref(),
            Some("error" | "aborted")
        ) {
            let reason = assistant_message
                .stop_reason
                .clone()
                .expect("matched terminal model reason must be present");
            let tool_calls = assistant_message.tool_calls.clone().unwrap_or_default();
            self.append_unexecuted_tool_turn(UnexecutedToolTurnRequest {
                context,
                step,
                assistant_message,
                tool_calls: &tool_calls,
                messages,
                transcript,
                reason: format!("model terminated before tool execution: {reason}"),
                status: &reason,
            })
            .await?;
            return Err(AgentError::ModelTerminated { reason });
        }

        if !has_tool_calls {
            if assistant_message.stop_reason.as_deref() == Some("length") {
                self.emit(AgentEvent::TurnFinished {
                    context,
                    step,
                    message: assistant_message,
                    tool_results: turn_tool_results,
                });
                return Err(AgentError::ModelTerminated {
                    reason: "length".to_string(),
                });
            }
            if step < self.policy.max_steps {
                if self.inject_steering_messages(messages, transcript).await? {
                    self.emit(AgentEvent::TurnFinished {
                        context,
                        step,
                        message: assistant_message.clone(),
                        tool_results: turn_tool_results,
                    });
                    return Ok(TurnOutcome::Continue);
                }
                if self.peek_steering().is_some() {
                    self.emit(AgentEvent::TurnFinished {
                        context,
                        step,
                        message: assistant_message.clone(),
                        tool_results: turn_tool_results,
                    });
                    return Ok(TurnOutcome::Continue);
                }
                if self.inject_follow_up_message(messages, transcript).await? {
                    self.emit(AgentEvent::TurnFinished {
                        context,
                        step,
                        message: assistant_message.clone(),
                        tool_results: turn_tool_results,
                    });
                    return Ok(TurnOutcome::Continue);
                }
            }
            self.emit(AgentEvent::TurnFinished {
                context,
                step,
                message: assistant_message.clone(),
                tool_results: turn_tool_results,
            });
            let (input_tokens, output_tokens, total_tokens) = stats.model_usage();
            return Ok(TurnOutcome::Completed(Box::new(RunResult {
                messages: std::mem::take(messages),
                final_message: assistant,
                steps: step,
                input_tokens,
                output_tokens,
                total_tokens,
            })));
        }

        let tool_calls = assistant.tool_calls.take().unwrap_or_default();

        if assistant_message.stop_reason.as_deref() == Some("length") {
            self.append_unexecuted_tool_turn(UnexecutedToolTurnRequest {
                context,
                step,
                assistant_message,
                tool_calls: &tool_calls,
                messages,
                transcript,
                reason: "model output was truncated before the tool call could execute".to_string(),
                status: "truncated",
            })
            .await?;
            if step == self.policy.max_steps {
                return Err(AgentError::MaxStepsExceeded {
                    max_steps: self.policy.max_steps,
                });
            }
            return Ok(TurnOutcome::Continue);
        }

        if step == self.policy.max_steps {
            self.append_unexecuted_tool_turn(UnexecutedToolTurnRequest {
                context,
                step,
                assistant_message,
                tool_calls: &tool_calls,
                messages,
                transcript,
                reason: "agent reached its maximum step count before executing the tool call"
                    .to_string(),
                status: "max_steps",
            })
            .await?;
            return Err(AgentError::MaxStepsExceeded {
                max_steps: self.policy.max_steps,
            });
        }

        if let Some(outcome) = self
            .try_run_parallel_tool_batch(
                context,
                step,
                &tool_calls,
                messages,
                transcript,
                &mut turn_tool_results,
                stats,
                deadline,
                cancellation,
                runtime_run_id,
                runtime,
            )
            .await?
        {
            self.emit(AgentEvent::TurnFinished {
                context,
                step,
                message: assistant_message,
                tool_results: turn_tool_results,
            });
            if let Some(error) = outcome.terminal_error {
                return Err(error);
            }
            if outcome.all_tools_terminate {
                let (input_tokens, output_tokens, total_tokens) = stats.model_usage();
                return Ok(TurnOutcome::Completed(Box::new(RunResult {
                    messages: std::mem::take(messages),
                    final_message: outcome
                        .last_tool_message
                        .expect("a non-empty parallel batch produces a tool message"),
                    steps: step,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                })));
            }
            return Ok(TurnOutcome::Continue);
        }

        match self
            .try_run_sequential_tool_batch(
                context,
                step,
                &tool_calls,
                assistant_message,
                messages,
                transcript,
                turn_tool_results,
                stats,
                deadline,
                cancellation,
                runtime_run_id,
                runtime,
            )
            .await?
        {
            Some(result) => Ok(TurnOutcome::Completed(Box::new(result))),
            None => Ok(TurnOutcome::Continue),
        }
    }
}
