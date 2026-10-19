//! Construction of completed turn results after tool execution.

use crate::message::Message;

use super::super::{RunResult, RunStats, TurnOutcome};

pub(super) fn completed_tool_turn(
    messages: &mut Vec<Message>,
    final_message: Message,
    step: usize,
    stats: &RunStats,
) -> TurnOutcome {
    let (input_tokens, output_tokens, total_tokens) = stats.model_usage();
    TurnOutcome::Completed(Box::new(RunResult {
        messages: std::mem::take(messages),
        final_message,
        steps: step,
        input_tokens,
        output_tokens,
        total_tokens,
    }))
}
