//! Build the model request without mutating durable turn state.

use crate::{message::Message, model::ChatModel};

use super::super::super::{
    model_turn::{request_assistant, ModelTurnRequest},
    AgentError, AgentLoop,
};
use super::super::TurnRequest;

/// Apply the two context-budget passes around ephemeral memory injection.
pub(super) async fn context<M>(
    agent: &AgentLoop<M>,
    request: &mut TurnRequest<'_>,
) -> Result<Vec<Message>, AgentError>
where
    M: ChatModel,
{
    let mut messages = agent
        .fit_context(
            request.context,
            request.step,
            request.messages.clone(),
            request.deadline,
            request.cancellation,
        )
        .await?;

    agent
        .prepare_memory_context(
            request.context,
            request.step,
            request.runtime_run_id,
            &mut messages,
            request.runtime,
            request.deadline,
            request.cancellation,
        )
        .await?;

    agent
        .fit_context(
            request.context,
            request.step,
            messages,
            request.deadline,
            request.cancellation,
        )
        .await
}

/// Send one model request after context preparation has completed.
pub(super) async fn model<M>(
    agent: &AgentLoop<M>,
    request: &mut TurnRequest<'_>,
    messages: &[Message],
) -> Result<Message, AgentError>
where
    M: ChatModel,
{
    request_assistant(
        ModelTurnRequest {
            client: &agent.client,
            model: &agent.model,
            api_key: agent.api_key.as_deref(),
            options: agent.options,
            definitions: request.definitions,
            context: request.context,
            step: request.step,
            messages,
            deadline: request.deadline,
            cancellation: request.cancellation,
            max_retries: agent.policy.max_model_retries,
            retry_backoff_ms: agent.policy.model_retry_backoff_ms,
        },
        request.stats,
        &|event| agent.emit(event),
    )
    .await
}
