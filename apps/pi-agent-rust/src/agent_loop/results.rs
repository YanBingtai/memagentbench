//! Durable transcript and event records for tools that never complete.
//!
//! The parent loop owns scheduling and runtime admission. This module keeps
//! the cancellation and pre-execution result persistence paths together so
//! those paths share exactly the same event and session ordering.

use std::time::Instant;

use uuid::Uuid;

use crate::{
    events::{AgentEvent, RunContext},
    message::{Message, ToolCall, ToolResultMetadata},
    model::ChatModel,
    tools::ToolResult,
};

use super::{elapsed_millis, tool_result_status, AgentError, AgentLoop};

pub(super) struct PersistToolResultRequest<'a> {
    pub(super) context: RunContext,
    pub(super) step: usize,
    pub(super) tool_call: &'a ToolCall,
    pub(super) result: &'a ToolResult,
    pub(super) runtime_identity: Option<(Uuid, Uuid)>,
    pub(super) started_at: Instant,
    pub(super) messages: &'a mut Vec<Message>,
    pub(super) transcript: &'a mut Vec<Message>,
    pub(super) turn_tool_results: &'a mut Vec<Message>,
}

pub(super) struct UnexecutedToolTurnRequest<'a> {
    pub(super) context: RunContext,
    pub(super) step: usize,
    pub(super) assistant_message: Message,
    pub(super) tool_calls: &'a [ToolCall],
    pub(super) messages: &'a mut Vec<Message>,
    pub(super) transcript: &'a mut Vec<Message>,
    pub(super) reason: String,
    pub(super) status: &'a str,
}

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(super) async fn append_unexecuted_tool_turn(
        &self,
        request: UnexecutedToolTurnRequest<'_>,
    ) -> Result<Vec<Message>, AgentError> {
        let UnexecutedToolTurnRequest {
            context,
            step,
            assistant_message,
            tool_calls,
            messages,
            transcript,
            reason,
            status,
        } = request;
        let tool_results = self
            .append_unexecuted_tool_results(
                context, step, tool_calls, messages, transcript, &reason, status,
            )
            .await?;
        self.emit(AgentEvent::TurnFinished {
            context,
            step,
            message: assistant_message,
            tool_results: tool_results.clone(),
        });
        Ok(tool_results)
    }

    pub(super) async fn persist_tool_result(
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

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn append_cancelled_tool_results(
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

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn append_unexecuted_tool_results(
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
