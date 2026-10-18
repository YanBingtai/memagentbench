//! Context budgeting and message-structure validation.

use std::collections::BTreeSet;

use crate::{agent_loop::AgentError, message::Message};

/// Keep the newest complete user turns within the context budget.
///
/// System messages are retained when they fit the byte budget. A turn starts at a user message and
/// includes every following assistant and tool message until the next user
/// message. If the newest turn is larger than the budget, it is kept whole so
/// an assistant tool call is never separated from its tool results; if it
/// cannot fit, the run fails with a typed context-limit error instead of
/// sending an oversized request. The byte budget is an approximate
/// serialized-message limit; providers should still apply their own
/// token-aware request limit when one is available.
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

/// Validate the message structure sent to a model provider.
///
/// Context compaction is allowed to rewrite messages, but it must preserve the
/// same tool-call protocol as a durable session: system messages stay at the
/// front, every tool result belongs to one preceding assistant call, and no
/// assistant call is left without all of its results.
pub(crate) fn validate_context_structure(messages: &[Message]) -> Result<(), AgentError> {
    let mut saw_non_system = false;
    let mut saw_user = false;
    let mut declared_calls = BTreeSet::new();
    let mut pending_calls = BTreeSet::new();

    for message in messages {
        match message.role.as_str() {
            "system" => {
                if saw_non_system {
                    return Err(AgentError::InvalidContext {
                        reason: "system messages must precede conversation messages".to_string(),
                    });
                }
            }
            "user" => {
                if !pending_calls.is_empty() {
                    return Err(AgentError::InvalidContext {
                        reason: "a user message appears before all tool results".to_string(),
                    });
                }
                saw_non_system = true;
                saw_user = true;
            }
            "assistant" => {
                if !pending_calls.is_empty() {
                    return Err(AgentError::InvalidContext {
                        reason: "an assistant message appears before all tool results".to_string(),
                    });
                }
                saw_non_system = true;
                if let Some(tool_calls) = &message.tool_calls {
                    for call in tool_calls {
                        if call.id.trim().is_empty() || !declared_calls.insert(call.id.as_str()) {
                            return Err(AgentError::InvalidContext {
                                reason: format!(
                                    "assistant tool call id {:?} is invalid or duplicated",
                                    call.id
                                ),
                            });
                        }
                        pending_calls.insert(call.id.clone());
                    }
                }
            }
            "tool" => {
                saw_non_system = true;
                let Some(call_id) = message.tool_call_id.as_deref() else {
                    return Err(AgentError::InvalidContext {
                        reason: "tool result is missing its tool_call_id".to_string(),
                    });
                };
                if !pending_calls.remove(call_id) {
                    return Err(AgentError::InvalidContext {
                        reason: format!(
                            "tool result refers to unknown or completed call {call_id:?}"
                        ),
                    });
                }
            }
            role => {
                return Err(AgentError::InvalidContext {
                    reason: format!("unsupported message role {role:?}"),
                });
            }
        }
    }

    if !saw_user {
        return Err(AgentError::InvalidContext {
            reason: "context must contain at least one user message".to_string(),
        });
    }
    if !pending_calls.is_empty() {
        return Err(AgentError::InvalidContext {
            reason: format!("tool calls are missing results: {pending_calls:?}"),
        });
    }
    Ok(())
}

pub(crate) fn message_size(message: &Message) -> usize {
    serde_json::to_vec(message)
        .map(|encoded| encoded.len())
        .unwrap_or_else(|_| message.content.as_deref().unwrap_or_default().len())
}
