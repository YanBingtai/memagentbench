//! Persist terminal records and cancelled messages for failed parallel workers.

use std::collections::HashSet;

use pi_agent_runtime::{CommitGrant, EventLedger, RuntimeCoordinator};

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    model::ChatModel,
};

use super::super::super::{AgentError, AgentLoop, RunStats};
use super::super::completion::ParallelBatchState;

#[allow(clippy::too_many_arguments)]
pub async fn persist_worker_error_results<M>(
    agent: &AgentLoop<M>,
    context: RunContext,
    step: usize,
    tool_calls: &[ToolCall],
    chunk_start: usize,
    chunk_end: usize,
    completed_indices: &HashSet<usize>,
    grants: &[CommitGrant],
    reason: &str,
    runtime: &mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    stats: &mut RunStats,
    messages: &mut Vec<Message>,
    transcript: &mut Vec<Message>,
    turn_tool_results: &mut Vec<Message>,
    state: &mut ParallelBatchState,
) -> Result<(), AgentError>
where
    M: ChatModel,
{
    for index in chunk_start..chunk_end {
        if completed_indices.contains(&index) {
            continue;
        }
        let grant = grants[index].clone();
        runtime
            .mark_unknown(
                grant.clone(),
                reason.to_string(),
                super::super::super::unix_timestamp_ms(),
            )
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
        stats.has_unknown_action = true;

        let message_start = messages.len();
        agent
            .append_cancelled_tool_results(
                context,
                step,
                &tool_calls[index..index + 1],
                0,
                messages,
                transcript,
                Some((grant.action_id.as_uuid(), grant.grant_id)),
                reason,
            )
            .await?;
        let message = messages.get(message_start).cloned();
        if let Some(message) = &message {
            turn_tool_results.push(message.clone());
        }
        state.record_cancelled_message(message);
    }
    Ok(())
}
