//! One model request with streaming callbacks and bounded retries.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use crate::{
    events::AgentEvent,
    message::{Message, ToolDefinition},
    model::{ChatModel, ChatOptions, ChatRequest, ToolCallDelta},
};

use super::{
    model_retry_delay_ms, timeout_error, AgentError, CancellationToken, RunContext, RunStats,
};

/// Inputs for one provider request and its retry policy.
///
/// Keeping this state in one value avoids a long argument list and makes the
/// boundary explicit for callers in the main loop.
pub(crate) struct ModelTurnRequest<'a, M> {
    pub client: &'a M,
    pub model: &'a str,
    pub api_key: Option<&'a str>,
    pub options: ChatOptions,
    pub definitions: &'a [ToolDefinition],
    pub context: RunContext,
    pub step: usize,
    pub messages: &'a [Message],
    pub deadline: Option<(tokio::time::Instant, Duration)>,
    pub cancellation: &'a CancellationToken,
    pub max_retries: usize,
    pub retry_backoff_ms: u64,
}

/// Request one assistant message while preserving per-attempt stream events.
///
/// This function owns only provider retry state. Tool validation, runtime
/// admission, transcript persistence, and turn lifecycle events remain in the
/// parent loop, so moving it cannot change ledger ordering.
pub(crate) async fn request_assistant<M, F>(
    request: ModelTurnRequest<'_, M>,
    stats: &mut RunStats,
    emit: &F,
) -> Result<Message, AgentError>
where
    M: ChatModel,
    F: Fn(AgentEvent) + Sync,
{
    let ModelTurnRequest {
        client,
        model,
        api_key,
        options,
        definitions,
        context,
        step,
        messages: request_messages,
        deadline,
        cancellation,
        max_retries,
        retry_backoff_ms,
    } = request;
    let mut retry_count = 0usize;
    loop {
        let attempt_emitted = Arc::new(AtomicBool::new(false));
        let delta_emitted = Arc::clone(&attempt_emitted);
        let mut emit_delta = |content: &str| {
            delta_emitted.store(true, Ordering::Release);
            emit(AgentEvent::AssistantDelta {
                context,
                step,
                content: content.to_string(),
            });
        };
        let reasoning_emitted = Arc::clone(&attempt_emitted);
        let mut emit_reasoning = |content: &str| {
            reasoning_emitted.store(true, Ordering::Release);
            emit(AgentEvent::ReasoningDelta {
                context,
                step,
                content: content.to_string(),
            });
        };
        let tool_call_emitted = Arc::clone(&attempt_emitted);
        let mut emit_tool_call = |delta: &ToolCallDelta| {
            tool_call_emitted.store(true, Ordering::Release);
            emit(AgentEvent::ToolCallDelta {
                context,
                step,
                index: delta.index,
                id: delta.id.clone(),
                call_type: delta.call_type.clone(),
                name: delta.name.clone(),
                arguments: delta.arguments.clone(),
            });
        };

        let model_request = client.stream_with_reasoning_and_tools(
            ChatRequest {
                model,
                api_key,
                messages: request_messages,
                tools: definitions,
                options,
            },
            &mut emit_delta,
            &mut emit_reasoning,
            &mut emit_tool_call,
        );
        let result = match deadline {
            Some((deadline, timeout)) => tokio::select! {
                result = tokio::time::timeout_at(deadline, model_request) => {
                    result
                        .map_err(|_| timeout_error(timeout))
                        .and_then(|result| result.map_err(AgentError::Model))
                }
                _ = cancellation.cancelled() => Err(AgentError::RunCancelled),
            },
            None => tokio::select! {
                result = model_request => result.map_err(AgentError::Model),
                _ = cancellation.cancelled() => Err(AgentError::RunCancelled),
            },
        };

        let attempt_number = retry_count.saturating_add(1);
        let attempt_had_output = attempt_emitted.load(Ordering::Acquire);
        match result {
            Ok(message) => {
                stats.record_model_usage(client.take_last_usage());
                return Ok(message);
            }
            Err(AgentError::Model(error)) if error.is_retryable() && retry_count < max_retries => {
                if attempt_had_output {
                    emit(AgentEvent::AssistantStreamReset {
                        context,
                        step,
                        attempt: attempt_number,
                        reason: error.to_string(),
                    });
                }
                retry_count = retry_count.saturating_add(1);
                stats.record_retry();
                let delay_ms = model_retry_delay_ms(retry_backoff_ms, retry_count);
                emit(AgentEvent::ModelRetry {
                    context,
                    step,
                    attempt: retry_count,
                    delay_ms,
                    error: error.to_string(),
                });
                let delay = Duration::from_millis(delay_ms);
                match deadline {
                    Some((deadline, timeout)) => tokio::select! {
                        result = tokio::time::timeout_at(deadline, tokio::time::sleep(delay)) => {
                            result.map_err(|_| timeout_error(timeout))?;
                        }
                        _ = cancellation.cancelled() => return Err(AgentError::RunCancelled),
                    },
                    None => tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        _ = cancellation.cancelled() => return Err(AgentError::RunCancelled),
                    },
                }
            }
            Err(error) => {
                if attempt_had_output {
                    emit(AgentEvent::AssistantStreamReset {
                        context,
                        step,
                        attempt: attempt_number,
                        reason: error.to_string(),
                    });
                }
                return Err(error);
            }
        }
    }
}
