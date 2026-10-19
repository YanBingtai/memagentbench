use std::time::Duration;

use pi_agent_runtime::{CommitGrant, EventLedger, RunId, RuntimeCoordinator};
use serde_json::Value;
use uuid::Uuid;

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    tools::{PreparedToolCall, ToolResult},
};

use super::super::super::{AgentError, CancellationToken, RunStats};

pub(in crate::agent_loop::sequential) struct ToolExecutionOutcome {
    pub(in crate::agent_loop::sequential) tool_result: ToolResult,
    pub(in crate::agent_loop::sequential) runtime_action: Option<(Uuid, Uuid)>,
    pub(in crate::agent_loop::sequential) pending_runtime_terminal: Option<CommitGrant>,
    pub(in crate::agent_loop::sequential) stop_after_tool: Option<(AgentError, String)>,
}

pub(in crate::agent_loop::sequential) type SequentialExecutionOutcome =
    crate::agent_loop::tool_execution::ToolExecutionOutcome;

pub(in crate::agent_loop::sequential) struct ToolExecutionRequest<'a> {
    pub(in crate::agent_loop::sequential) context: RunContext,
    pub(in crate::agent_loop::sequential) step: usize,
    pub(in crate::agent_loop::sequential) tool_calls: &'a [ToolCall],
    pub(in crate::agent_loop::sequential) tool_index: usize,
    pub(in crate::agent_loop::sequential) tool_call: &'a ToolCall,
    pub(in crate::agent_loop::sequential) arguments: Value,
    pub(in crate::agent_loop::sequential) prepared_call: Option<PreparedToolCall>,
    pub(in crate::agent_loop::sequential) validation_error: Option<String>,
    pub(in crate::agent_loop::sequential) tool_timeout: Duration,
    pub(in crate::agent_loop::sequential) now_ms: u64,
    pub(in crate::agent_loop::sequential) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(in crate::agent_loop::sequential) cancellation: &'a CancellationToken,
    pub(in crate::agent_loop::sequential) runtime_run_id: RunId,
    pub(in crate::agent_loop::sequential) messages: &'a mut Vec<Message>,
    pub(in crate::agent_loop::sequential) transcript: &'a mut Vec<Message>,
    pub(in crate::agent_loop::sequential) stats: &'a mut RunStats,
    pub(in crate::agent_loop::sequential) runtime:
        &'a mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
}
