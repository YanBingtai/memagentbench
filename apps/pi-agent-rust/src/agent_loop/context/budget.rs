//! Context-size accounting and complete-turn truncation.

use crate::{agent_loop::AgentError, message::Message};

/// Keep the newest complete user turns within the context budget.
///
/// System messages are retained when they fit the byte budget. A turn starts at a user message and
/// includes every following assistant and tool message until the next user message. If the newest
/// turn is larger than the budget, it is kept whole so an assistant tool call is never separated
/// from its tool results; if it cannot fit, the run fails with a typed context-limit error instead
/// of sending an oversized request. The byte budget is an approximate serialized-message limit;
/// providers should still apply their own token-aware request limit when one is available.
pub(crate) fn truncate_context(
    messages: &[Message],
    max_context_messages: usize,
    max_context_bytes: usize,
) -> Result<Vec<Message>, AgentError> {
    let mut system_messages = Vec::new();
    let mut turns = Vec::new();
    let mut current_turn = Vec::new();

    for message in messages {
        if message.role == "system" {
            system_messages.push(message.clone());
            continue;
        }
        if message.role == "user" && !current_turn.is_empty() {
            turns.push(current_turn);
            current_turn = Vec::new();
        }
        current_turn.push(message.clone());
    }
    if !current_turn.is_empty() {
        turns.push(current_turn);
    }

    let mut selected_turns = Vec::new();
    let mut used_messages = 0;
    let system_bytes = system_messages.iter().map(message_size).sum::<usize>();
    if system_bytes > max_context_bytes {
        return Err(AgentError::ContextTooLarge {
            required_bytes: system_bytes,
            limit_bytes: max_context_bytes,
        });
    }
    let turn_budget = max_context_bytes.saturating_sub(system_bytes);
    let mut used_bytes: usize = 0;
    for turn in turns.into_iter().rev() {
        let turn_bytes = turn.iter().map(message_size).sum::<usize>();
        if selected_turns.is_empty() && turn_bytes > turn_budget {
            return Err(AgentError::ContextTooLarge {
                required_bytes: system_bytes.saturating_add(turn_bytes),
                limit_bytes: max_context_bytes,
            });
        }
        if selected_turns.is_empty()
            || (used_messages + turn.len() <= max_context_messages
                && used_bytes.saturating_add(turn_bytes) <= turn_budget)
        {
            used_messages += turn.len();
            used_bytes = used_bytes.saturating_add(turn_bytes);
            selected_turns.push(turn);
        } else {
            break;
        }
    }
    selected_turns.reverse();

    let mut result = system_messages;
    result.extend(selected_turns.into_iter().flatten());
    Ok(result)
}

pub(crate) fn message_size(message: &Message) -> usize {
    serde_json::to_vec(message)
        .map(|encoded| encoded.len())
        .unwrap_or_else(|_| message.content.as_deref().unwrap_or_default().len())
}
