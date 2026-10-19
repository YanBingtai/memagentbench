//! Dispatch and terminal handling for model-provided tool calls.

mod dispatch;
mod result;
mod terminal;

use std::time::Duration;

use pi_agent_runtime::{EventLedger, RunId, RuntimeCoordinator};

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    model::ChatModel,
};

use super::{
    super::{AgentError, AgentLoop, CancellationToken, RunStats},
    TurnOutcome,
};

pub(super) struct ToolBatchRequest<'a> {
    pub(super) context: RunContext,
    pub(super) step: usize,
    pub(super) assistant_message: Message,
    pub(super) tool_calls: Vec<ToolCall>,
    pub(super) messages: &'a mut Vec<Message>,
    pub(super) transcript: &'a mut Vec<Message>,
    pub(super) stats: &'a mut RunStats,
    pub(super) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(super) cancellation: &'a CancellationToken,
    pub(super) runtime_run_id: RunId,
    pub(super) runtime: &'a mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
}

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(super) async fn run_tool_batch(
        &self,
        mut request: ToolBatchRequest<'_>,
    ) -> Result<TurnOutcome, AgentError> {
        if let Some(outcome) = self.handle_unexecutable_tool_batch(&mut request).await? {
            return Ok(outcome);
        }

        self.dispatch_tool_batch(request).await
    }
}
