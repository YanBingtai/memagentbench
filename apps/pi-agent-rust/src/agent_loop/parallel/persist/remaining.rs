//! Persist tool results for calls that were never dispatched.

use pi_agent_runtime::{CommitGrant, EventLedger, RuntimeCoordinator};

use crate::{
    events::RunContext,
    message::{Message, ToolCall},
    model::ChatModel,
};

use super::super::super::{AgentError, AgentLoop};

#[allow(clippy::too_many_arguments)]
pub async fn persist_remaining_tool_results<M>(
    agent: &AgentLoop<M>,
    context: RunContext,
    step: usize,
    tool_calls: &[ToolCall],
    first_pending: usize,
    grants: &[CommitGrant],
    runtime: &mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    messages: &mut Vec<Message>,
    transcript: &mut Vec<Message>,
    turn_tool_results: &mut Vec<Message>,
    reason: &str,
) -> Result<(), AgentError>
where
    M: ChatModel,
{
    for grant in grants.iter().skip(first_pending) {
        runtime
            .fail(
                grant.clone(),
                reason,
                super::super::super::unix_timestamp_ms(),
            )
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
    }
    let pending = agent
        .append_unexecuted_tool_results(
            context,
            step,
            &tool_calls[first_pending..],
            messages,
            transcript,
            reason,
            "cancelled",
        )
        .await?;
    turn_tool_results.extend(pending);
    Ok(())
}
