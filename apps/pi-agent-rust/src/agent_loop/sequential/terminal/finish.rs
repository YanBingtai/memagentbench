//! Completion path after an ordered tool batch.

use crate::events::RunContext;
use crate::message::Message;
use crate::model::ChatModel;

use super::super::super::{AgentError, AgentLoop, RunResult, RunStats};
use super::super::batch::BatchState;
use super::emit_turn_finished;

pub(in crate::agent_loop::sequential) struct FinishRequest<'a> {
    pub(in crate::agent_loop::sequential) context: RunContext,
    pub(in crate::agent_loop::sequential) step: usize,
    pub(in crate::agent_loop::sequential) assistant_message: Message,
    pub(in crate::agent_loop::sequential) messages: &'a mut Vec<Message>,
    pub(in crate::agent_loop::sequential) turn_tool_results: Vec<Message>,
    pub(in crate::agent_loop::sequential) stats: &'a RunStats,
    pub(in crate::agent_loop::sequential) state: BatchState,
}

pub(in crate::agent_loop::sequential) fn finish<M>(
    agent: &AgentLoop<M>,
    request: FinishRequest<'_>,
) -> Result<Option<RunResult>, AgentError>
where
    M: ChatModel,
{
    let FinishRequest {
        context,
        step,
        assistant_message,
        messages,
        turn_tool_results,
        stats,
        state,
    } = request;
    emit_turn_finished(agent, context, step, assistant_message, turn_tool_results);
    if !state.all_tools_terminate() {
        return Ok(None);
    }

    let (input_tokens, output_tokens, total_tokens) = stats.model_usage();
    Ok(Some(RunResult {
        messages: std::mem::take(messages),
        final_message: state.take_last_tool_message(),
        steps: step,
        input_tokens,
        output_tokens,
        total_tokens,
    }))
}
