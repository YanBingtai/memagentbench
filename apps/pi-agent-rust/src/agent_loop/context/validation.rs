//! Message protocol validation for model requests.

use std::collections::BTreeSet;

use crate::{agent_loop::AgentError, message::Message};

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
