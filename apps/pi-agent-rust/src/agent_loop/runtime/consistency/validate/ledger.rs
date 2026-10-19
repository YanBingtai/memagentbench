//! Projection of durable action admission and terminal events.

use std::collections::{HashMap, HashSet};

use pi_agent_runtime::{EventLedger, EventSequence, RuntimeCoordinator};
use serde_json::Value;

use crate::AgentError;

pub(super) struct LedgerActions {
    pub(super) admitted: HashMap<String, String>,
    pub(super) terminal: HashSet<String>,
}

pub(super) fn collect<L: EventLedger>(
    runtime: &RuntimeCoordinator<L>,
) -> Result<LedgerActions, AgentError> {
    let events = runtime
        .ledger()
        .scan_after(EventSequence::INITIAL)
        .map_err(|error| AgentError::Runtime(error.to_string()))?;
    let mut actions = LedgerActions {
        admitted: HashMap::new(),
        terminal: HashSet::new(),
    };
    for event in events {
        let Some(action_id) = event.draft.payload.get("action_id").and_then(Value::as_str) else {
            continue;
        };
        if event.draft.kind.as_str() == pi_agent_runtime::ACTION_ADMITTED_EVENT {
            if let Some(grant_id) = event.draft.payload.get("grant_id").and_then(Value::as_str) {
                actions
                    .admitted
                    .insert(action_id.to_string(), grant_id.to_string());
            }
        }
        if matches!(
            event.draft.kind.as_str(),
            pi_agent_runtime::ACTION_SUCCEEDED_EVENT
                | pi_agent_runtime::ACTION_FAILED_EVENT
                | pi_agent_runtime::ACTION_UNKNOWN_EVENT
                | pi_agent_runtime::ACTION_RECONCILED_EVENT
        ) {
            actions.terminal.insert(action_id.to_string());
        }
    }
    Ok(actions)
}
