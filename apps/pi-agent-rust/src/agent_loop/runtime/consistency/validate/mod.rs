//! Validate session tool results against the durable runtime ledger.

mod ledger;
mod message;

use pi_agent_runtime::{EventLedger, RuntimeCoordinator};

use crate::{message::Message, AgentError};

pub(crate) fn validate_session_runtime_consistency<L: EventLedger>(
    messages: &[Message],
    runtime: &RuntimeCoordinator<L>,
) -> Result<(), AgentError> {
    let actions = ledger::collect(runtime)?;
    for message in messages {
        message::validate(message, &actions)?;
    }
    Ok(())
}
