//! Terminal paths for an ordered model-provided tool batch.

mod finish;
mod stop;

use crate::{
    events::{AgentEvent, RunContext},
    message::Message,
    model::ChatModel,
};

use super::super::AgentLoop;

pub(super) use finish::{finish, FinishRequest};
pub(super) use stop::{stop_after_tool, StopRequest};

pub(super) fn emit_turn_finished<M>(
    agent: &AgentLoop<M>,
    context: RunContext,
    step: usize,
    message: Message,
    tool_results: Vec<Message>,
) where
    M: ChatModel,
{
    agent.emit(AgentEvent::TurnFinished {
        context,
        step,
        message,
        tool_results,
    });
}
