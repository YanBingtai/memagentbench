//! Inspection of unresolved runtime actions.

use pi_agent_runtime::{EventLedger, JsonlLedger, RuntimeCoordinator};

use crate::{
    model::ChatModel,
    tools::{ToolGateway, ToolReplayPolicy},
};

use super::super::{
    super::{AgentError, AgentLoop, PendingRuntimeAction},
    policy::runtime_policy,
};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    /// Inspect unresolved runtime actions without starting an agent run.
    pub fn pending_runtime_actions(&self) -> Result<Vec<PendingRuntimeAction>, AgentError> {
        self.ensure_idle()?;
        let Some(path) = &self.runtime_ledger_file else {
            return Ok(Vec::new());
        };
        let ledger = JsonlLedger::open(path)
            .map_err(|error| AgentError::Runtime(format!("{}: {error}", path.display())))?;
        let ledger = Box::new(ledger) as Box<dyn EventLedger + Send>;
        let Some(runtime) = RuntimeCoordinator::recover_latest(
            runtime_policy(
                &self.tools.definitions(),
                self.policy.tool_timeout_secs,
                self.allow_irreversible_tools,
            ),
            ledger,
        )
        .map_err(|error| AgentError::Runtime(error.to_string()))?
        else {
            return Ok(Vec::new());
        };
        Ok(runtime
            .pending_actions()
            .iter()
            .map(|grant| pending_runtime_action(grant, &self.tools))
            .collect())
    }
}

fn pending_runtime_action(
    grant: &pi_agent_runtime::CommitGrant,
    tools: &ToolGateway,
) -> PendingRuntimeAction {
    PendingRuntimeAction {
        action_id: grant.action_id.as_uuid(),
        grant_id: grant.grant_id,
        run_id: grant.run_id.as_uuid(),
        tool_name: grant.tool_name.clone(),
        idempotency_key: grant.idempotency_key.clone(),
        argument_digest: grant.argument_digest.clone(),
        risk: grant.risk,
        issued_at_ms: grant.issued_at_ms,
        expires_at_ms: grant.expires_at_ms,
        replay_policy: tools
            .replay_policy(&grant.tool_name)
            .unwrap_or(ToolReplayPolicy::Never),
    }
}
