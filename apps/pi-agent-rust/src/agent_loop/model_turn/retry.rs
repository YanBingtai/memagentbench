//! Retry policy, stream reset events, and cancellable backoff.

use std::{fmt::Display, time::Duration};

use crate::{
    events::{AgentEvent, RunContext},
    model::ModelError,
};

use crate::agent_loop::helpers::timeout_error;

use super::{AgentError, CancellationToken};

pub(super) fn is_eligible(error: &ModelError, retry_count: usize, max_retries: usize) -> bool {
    error.is_retryable() && retry_count < max_retries
}

pub(super) fn emit_stream_reset<F, D>(
    emit: &F,
    context: RunContext,
    step: usize,
    attempt: usize,
    had_output: bool,
    reason: D,
) where
    F: Fn(AgentEvent) + Sync,
    D: Display,
{
    if had_output {
        emit(AgentEvent::AssistantStreamReset {
            context,
            step,
            attempt,
            reason: reason.to_string(),
        });
    }
}

pub(super) fn emit_retry<F>(
    emit: &F,
    context: RunContext,
    step: usize,
    attempt: usize,
    delay_ms: u64,
    error: &ModelError,
) where
    F: Fn(AgentEvent) + Sync,
{
    emit(AgentEvent::ModelRetry {
        context,
        step,
        attempt,
        delay_ms,
        error: error.to_string(),
    });
}

pub(super) async fn wait(
    delay: Duration,
    deadline: Option<(tokio::time::Instant, Duration)>,
    cancellation: &CancellationToken,
) -> Result<(), AgentError> {
    match deadline {
        Some((deadline, timeout)) => tokio::select! {
            result = tokio::time::timeout_at(deadline, tokio::time::sleep(delay)) => {
                result.map_err(|_| timeout_error(timeout))
            }
            _ = cancellation.cancelled() => Err(AgentError::RunCancelled),
        },
        None => tokio::select! {
            _ = tokio::time::sleep(delay) => Ok(()),
            _ = cancellation.cancelled() => Err(AgentError::RunCancelled),
        },
    }
}
