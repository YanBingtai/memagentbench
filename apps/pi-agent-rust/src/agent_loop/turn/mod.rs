//! One model/tool turn and its terminal outcome.

mod response;
mod terminal;
mod tool_batch;
pub(crate) mod validation;

use std::time::Duration;

use pi_agent_runtime::{EventLedger, RunId, RuntimeCoordinator};

use crate::{
    events::{AgentEvent, RunContext},
    message::{Message, ToolDefinition},
    model::ChatModel,
};

use super::{AgentError, AgentLoop, CancellationToken, RunResult, RunStats};
use tool_batch::ToolBatchRequest;

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

pub(super) struct AssistantResponse {
    pub(super) assistant: Message,
    pub(super) has_tool_calls: bool,
}

pub(super) enum TurnOutcome {
    Continue,
    Completed(Box<RunResult>),
}

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(super) async fn run_turn(
        &self,
        mut request: TurnRequest<'_>,
    ) -> Result<TurnOutcome, AgentError> {
        self.inject_steering_messages(request.messages, request.transcript)
            .await?;
        request.stats.steps = request.step;
        self.emit(AgentEvent::TurnStarted {
            context: request.context,
            step: request.step,
        });
        self.emit(AgentEvent::ModelStarted {
            context: request.context,
            step: request.step,
            model: self.model.clone(),
        });

        let response = self.request_assistant_response(&mut request).await?;
        let AssistantResponse {
            assistant,
            has_tool_calls,
        } = response;
        let context = request.context;
        let step = request.step;

        if let Some(error) = self
            .handle_terminal_model_response(
                context,
                step,
                &assistant,
                request.messages,
                request.transcript,
            )
            .await?
        {
            return Err(error);
        }

        if !has_tool_calls {
            return self
                .finish_without_tools(
                    context,
                    step,
                    assistant,
                    request.messages,
                    request.transcript,
                    request.stats,
                )
                .await;
        }

        let assistant_message = assistant.clone();
        let tool_calls = assistant.tool_calls.unwrap_or_default();
        self.run_tool_batch(ToolBatchRequest {
            context,
            step,
            assistant_message,
            tool_calls,
            messages: request.messages,
            transcript: request.transcript,
            stats: request.stats,
            deadline: request.deadline,
            cancellation: request.cancellation,
            runtime_run_id: request.runtime_run_id,
            runtime: request.runtime,
        })
        .await
    }
}
