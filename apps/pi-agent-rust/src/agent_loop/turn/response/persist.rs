//! Validate, persist, and emit a completed assistant response.

use crate::{
    events::{AgentEvent, RunContext},
    message::Message,
    model::ChatModel,
};

use super::super::super::{AgentError, AgentLoop};
use super::super::{validation::validate_assistant_tool_calls, AssistantResponse, TurnRequest};

pub(super) async fn assistant<M>(
    agent: &AgentLoop<M>,
    request: &mut TurnRequest<'_>,
    assistant: Message,
) -> Result<AssistantResponse, AgentError>
where
    M: ChatModel,
{
    let has_tool_calls = assistant
        .tool_calls
        .as_ref()
        .is_some_and(|calls| !calls.is_empty());

    if let Err(error) = validate_assistant_tool_calls(
        request.step,
        assistant.tool_calls.as_deref(),
        agent.policy.max_tool_calls_per_step,
    ) {
        emit_invalid(agent, request.context, request.step, &assistant);
        return Err(error);
    }

    record_valid(agent, request, &assistant).await?;
    Ok(AssistantResponse {
        assistant,
        has_tool_calls,
    })
}

fn emit_invalid<M>(agent: &AgentLoop<M>, context: RunContext, step: usize, message: &Message)
where
    M: ChatModel,
{
    agent.emit(AgentEvent::AssistantMessage {
        context,
        step,
        message: message.clone(),
    });
    agent.emit(AgentEvent::TurnFinished {
        context,
        step,
        message: message.clone(),
        tool_results: Vec::new(),
    });
}

async fn record_valid<M>(
    agent: &AgentLoop<M>,
    request: &mut TurnRequest<'_>,
    message: &Message,
) -> Result<(), AgentError>
where
    M: ChatModel,
{
    request.messages.push(message.clone());
    request.transcript.push(message.clone());
    if let Some(session) = &agent.session {
        session.append(message).await?;
    }
    agent.emit(AgentEvent::AssistantMessage {
        context: request.context,
        step: request.step,
        message: message.clone(),
    });
    Ok(())
}
