//! Inputs shared by one parallel tool chunk.

use std::time::Duration;

use pi_agent_runtime::{CommitGrant, EventLedger, RuntimeCoordinator};
use serde_json::Value;

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    model::ChatModel,
    tools::PreparedToolCall,
};

use super::super::completion::ParallelBatchState;
use super::super::{AgentLoop, CancellationToken, RunStats};

/// All state needed to dispatch and persist one bounded parallel chunk.
pub(crate) struct ParallelChunkRequest<'a, M: ChatModel> {
    pub(crate) agent: &'a AgentLoop<M>,
    pub(crate) context: RunContext,
    pub(crate) step: usize,
    pub(crate) tool_calls: &'a [ToolCall],
    pub(crate) prepared_calls: &'a mut [Option<(Value, PreparedToolCall)>],
    pub(crate) grants: &'a [CommitGrant],
    pub(crate) chunk_start: usize,
    pub(crate) chunk_end: usize,
    pub(crate) tool_timeout: Duration,
    pub(crate) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(crate) cancellation: &'a CancellationToken,
    pub(crate) messages: &'a mut Vec<Message>,
    pub(crate) transcript: &'a mut Vec<Message>,
    pub(crate) turn_tool_results: &'a mut Vec<Message>,
    pub(crate) stats: &'a mut RunStats,
    pub(crate) runtime: &'a mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    pub(crate) state: &'a mut ParallelBatchState,
}
