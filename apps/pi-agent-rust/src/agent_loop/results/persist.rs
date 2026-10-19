use std::time::Instant;

use uuid::Uuid;

use crate::{
    events::{AgentEvent, RunContext},
    message::{Message, ToolCall, ToolResultMetadata},
    model::ChatModel,
    tools::ToolResult,
};

use super::super::{elapsed_millis, tool_result_status, AgentError, AgentLoop};

pub(in crate::agent_loop) struct PersistToolResultRequest<'a> {
    pub(in crate::agent_loop) context: RunContext,
    pub(in crate::agent_loop) step: usize,
    pub(in crate::agent_loop) tool_call: &'a ToolCall,
    pub(in crate::agent_loop) result: &'a ToolResult,
    pub(in crate::agent_loop) runtime_identity: Option<(Uuid, Uuid)>,
    pub(in crate::agent_loop) started_at: Instant,
    pub(in crate::agent_loop) messages: &'a mut Vec<Message>,
    pub(in crate::agent_loop) transcript: &'a mut Vec<Message>,
    pub(in crate::agent_loop) turn_tool_results: &'a mut Vec<Message>,
}

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(in crate::agent_loop) async fn persist_tool_result(
        &self,
        request: PersistToolResultRequest<'_>,
    ) -> Result<Message, AgentError> {
        let PersistToolResultRequest {
            context,
            step,
            tool_call,
            result,
            runtime_identity,
            started_at,
            messages,
            transcript,
            turn_tool_results,
        } = request;
        let content = result.content.clone();
        self.emit(AgentEvent::ToolFinished {
            context,
            step,
            call_id: tool_call.id.clone(),
            tool_name: tool_call.function.name.clone(),
            content: content.clone(),
            is_error: result.is_error,
            details: result.details.clone(),
            terminate: result.terminate,
            duration_ms: elapsed_millis(started_at),
            status: tool_result_status(result),
            action_id: runtime_identity.map(|(action_id, _)| action_id),
            grant_id: runtime_identity.map(|(_, grant_id)| grant_id),
        });
        let metadata = runtime_identity
            .map(|(action_id, grant_id)| {
                ToolResultMetadata::new(result.details.clone(), result.is_error, result.terminate)
                    .with_runtime_identity(action_id, grant_id)
            })
            .unwrap_or_else(|| {
                ToolResultMetadata::new(result.details.clone(), result.is_error, result.terminate)
            });
        let tool_message = Message::tool_with_result(tool_call.id.clone(), content, metadata);
        messages.push(tool_message.clone());
        transcript.push(tool_message.clone());
        turn_tool_results.push(tool_message.clone());
        if let Some(session) = &self.session {
            session.append(&tool_message).await?;
        }
        Ok(tool_message)
    }
}
