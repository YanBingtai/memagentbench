use uuid::Uuid;

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
    pub(in crate::agent_loop) async fn append_cancelled_tool_results(
        &self,
        context: RunContext,
        step: usize,
        tool_calls: &[ToolCall],
        first_pending: usize,
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
        runtime_identity: Option<(Uuid, Uuid)>,
        reason: &str,
    ) -> Result<(), AgentError> {
        for (index, tool_call) in tool_calls.iter().enumerate().skip(first_pending) {
            if index > first_pending {
                self.emit(AgentEvent::ToolStarted {
                    context,
                    step,
                    call: tool_call.clone(),
                });
            }
            let admitted_unknown = index == first_pending && runtime_identity.is_some();
            let details = serde_json::json!({
                "status": if admitted_unknown { "unknown" } else { "cancelled" },
                "executed": admitted_unknown,
            });
            let (action_id, grant_id) = if index == first_pending {
                runtime_identity
                    .map(|(action_id, grant_id)| (Some(action_id), Some(grant_id)))
                    .unwrap_or((None, None))
            } else {
                (None, None)
            };
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
                status: if admitted_unknown {
                    "unknown".to_string()
                } else {
                    "cancelled".to_string()
                },
                action_id,
                grant_id,
            });
            let tool_message = Message::tool_with_result(
                tool_call.id.clone(),
                reason,
                match (action_id, grant_id) {
                    (Some(action_id), Some(grant_id)) => {
                        ToolResultMetadata::new(details, true, false)
                            .with_runtime_identity(action_id, grant_id)
                    }
                    _ => ToolResultMetadata::new(details, true, false),
                },
            );
            messages.push(tool_message.clone());
            transcript.push(tool_message.clone());
            if let Some(session) = &self.session {
                session.append(&tool_message).await?;
            }
        }
        Ok(())
    }
}
