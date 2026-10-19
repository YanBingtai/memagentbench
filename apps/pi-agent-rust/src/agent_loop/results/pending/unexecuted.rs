use crate::{
    events::{AgentEvent, RunContext},
    message::{Message, ToolCall, ToolResultMetadata},
    model::ChatModel,
};

use crate::agent_loop::{AgentError, AgentLoop};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    #[allow(clippy::too_many_arguments)]
    pub(in crate::agent_loop) async fn append_unexecuted_tool_results(
        &self,
        context: RunContext,
        step: usize,
        tool_calls: &[ToolCall],
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
        reason: &str,
        status: &str,
    ) -> Result<Vec<Message>, AgentError> {
        let mut results = Vec::with_capacity(tool_calls.len());
        for tool_call in tool_calls {
            self.emit(AgentEvent::ToolStarted {
                context,
                step,
                call: tool_call.clone(),
            });
            let details = serde_json::json!({
                "status": status,
                "executed": false,
            });
            self.emit(AgentEvent::ToolFinished {
                context,
                step,
                call_id: tool_call.id.clone(),
                tool_name: tool_call.function.name.clone(),
                content: reason.to_string(),
                is_error: true,
                details: details.clone(),
                terminate: false,
                duration_ms: 0,
                status: status.to_string(),
                action_id: None,
                grant_id: None,
            });
            let tool_message = Message::tool_with_result(
                tool_call.id.clone(),
                reason,
                ToolResultMetadata::new(details, true, false),
            );
            messages.push(tool_message.clone());
            transcript.push(tool_message.clone());
            if let Some(session) = &self.session {
                session.append(&tool_message).await?;
            }
            results.push(tool_message);
        }
        Ok(results)
    }
}
