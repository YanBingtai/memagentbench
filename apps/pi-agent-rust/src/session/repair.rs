//! Transcript identity and tail-orphan validation.

use std::collections::BTreeSet;

use uuid::Uuid;

use super::{error::SessionError, types::SessionRecord};

pub(crate) fn find_tail_orphans(records: &[SessionRecord]) -> Result<Vec<String>, SessionError> {
    validate_tool_result_identity(records)?;
    for (index, record) in records.iter().enumerate() {
        let Some(tool_calls) = record.message.tool_calls.as_ref() else {
            continue;
        };
        if record.message.role != "assistant" || tool_calls.is_empty() {
            continue;
        }

        let expected: BTreeSet<&str> = tool_calls.iter().map(|call| call.id.as_str()).collect();
        let mut found = BTreeSet::new();
        let mut has_non_tool_after_batch = false;
        for following in records.iter().skip(index + 1) {
            if following.message.role == "tool" {
                if let Some(call_id) = following.message.tool_call_id.as_deref() {
                    if expected.contains(call_id) {
                        found.insert(call_id);
                    }
                }
            } else {
                has_non_tool_after_batch = true;
                break;
            }
        }

        let missing: Vec<String> = tool_calls
            .iter()
            .filter(|call| !found.contains(call.id.as_str()))
            .map(|call| call.id.clone())
            .collect();
        if missing.is_empty() {
            continue;
        }
        if has_non_tool_after_batch {
            return Err(SessionError::OrphanToolCall { call_ids: missing });
        }
        return Ok(missing);
    }
    Ok(Vec::new())
}

pub(crate) fn validate_tool_result_identity(records: &[SessionRecord]) -> Result<(), SessionError> {
    let mut declared = BTreeSet::new();
    let mut results = BTreeSet::new();
    for record in records {
        if record.message.role == "assistant" {
            let Some(tool_calls) = record.message.tool_calls.as_ref() else {
                continue;
            };
            for call in tool_calls {
                if call.id.trim().is_empty() {
                    return Err(SessionError::DuplicateToolCall {
                        call_id: call.id.clone(),
                    });
                }
                if !declared.insert(call.id.as_str()) {
                    return Err(SessionError::DuplicateToolCall {
                        call_id: call.id.clone(),
                    });
                }
            }
        } else if record.message.role == "tool" {
            let Some(call_id) = record.message.tool_call_id.as_deref() else {
                return Err(SessionError::ToolResultMissingCallId);
            };
            if !declared.contains(call_id) {
                return Err(SessionError::UnexpectedToolResult {
                    call_id: call_id.to_string(),
                });
            }
            if !results.insert(call_id) {
                return Err(SessionError::DuplicateToolResult {
                    call_id: call_id.to_string(),
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn next_sequence(records: &[SessionRecord], session_id: Uuid) -> u64 {
    records
        .iter()
        .filter(|record| record.session_id == session_id)
        .map(|record| record.sequence.saturating_add(1))
        .max()
        .unwrap_or(0)
}
