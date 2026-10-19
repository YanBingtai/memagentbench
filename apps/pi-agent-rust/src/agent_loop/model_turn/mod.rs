//! One model request with streaming callbacks and bounded retries.
//!
//! A model turn is deliberately narrower than the surrounding agent turn: it
//! owns provider attempts, retry timing, and ephemeral stream events. The
//! caller remains responsible for validating and persisting the final message.

mod attempt;
mod retry;

use std::time::Duration;

use crate::{
    events::AgentEvent,
    message::{Message, ToolDefinition},
    model::{ChatModel, ChatOptions},
};

use super::{model_retry_delay_ms, AgentError, CancellationToken, RunContext, RunStats};

/// Inputs for one provider request and its retry policy.
///
/// Keeping this state in one value avoids a long argument list and makes the
/// boundary explicit for callers in the main loop.
pub(crate) struct ModelTurnRequest<'a, M> {
    pub(crate) client: &'a M,
    pub(crate) model: &'a str,
    pub(crate) api_key: Option<&'a str>,
    pub(crate) options: ChatOptions,
    pub(crate) definitions: &'a [ToolDefinition],
    pub(crate) context: RunContext,
    pub(crate) step: usize,
    pub(crate) messages: &'a [Message],
    pub(crate) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(crate) cancellation: &'a CancellationToken,
    pub(crate) max_retries: usize,
    pub(crate) retry_backoff_ms: u64,
}

/// Request one assistant message while preserving per-attempt stream events.
///
/// Tool validation, runtime admission, transcript persistence, and turn
/// lifecycle events stay in the parent loop. Keeping them outside this module
/// makes retries unable to reorder durable turn state.
pub(crate) async fn request_assistant<M, F>(
    request: ModelTurnRequest<'_, M>,
    stats: &mut RunStats,
    emit: &F,
) -> Result<Message, AgentError>
where
    M: ChatModel,
    F: Fn(AgentEvent) + Sync,
{
    let mut retry_count = 0usize;

    loop {
        let attempt = attempt::run(&request, emit).await;
        let attempt_number = retry_count.saturating_add(1);

        match attempt.result {
            Ok(message) => {
                stats.record_model_usage(request.client.take_last_usage());
                return Ok(message);
            }
            Err(AgentError::Model(error))
                if retry::is_eligible(&error, retry_count, request.max_retries) =>
            {
                retry::emit_stream_reset(
                    emit,
                    request.context,
                    request.step,
                    attempt_number,
                    attempt.had_output,
                    &error,
                );
                retry_count = retry_count.saturating_add(1);
                stats.record_retry();
                let delay_ms = model_retry_delay_ms(request.retry_backoff_ms, retry_count);
                retry::emit_retry(
                    emit,
                    request.context,
                    request.step,
                    retry_count,
                    delay_ms,
                    &error,
                );
                retry::wait(
                    Duration::from_millis(delay_ms),
                    request.deadline,
                    request.cancellation,
                )
                .await?;
            }
            Err(error) => {
                retry::emit_stream_reset(
                    emit,
                    request.context,
                    request.step,
                    attempt_number,
                    attempt.had_output,
                    &error,
                );
                return Err(error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::retry;

    #[test]
    fn retry_eligibility_stops_at_policy_limit() {
        let error = crate::model::ModelError::StreamIncomplete;
        assert!(retry::is_eligible(&error, 0, 1));
        assert!(!retry::is_eligible(&error, 1, 1));
    }
}
