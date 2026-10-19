//! Explicit reconciliation of unknown runtime actions.

use pi_agent_runtime::{EventLedger, JsonlLedger, ReconciliationOutcome, RuntimeCoordinator};
use uuid::Uuid;

use crate::model::ChatModel;

use super::super::{
    super::{unix_timestamp_ms, AgentError, AgentLoop, RuntimeReconcileOutcome},
    policy::runtime_policy,
};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    /// Close one unknown runtime action using caller-provided external evidence.
    pub fn reconcile_runtime_action(
        &self,
        action_id: Uuid,
        outcome: RuntimeReconcileOutcome,
    ) -> Result<(), AgentError> {
        self.ensure_idle()?;
        let Some(path) = &self.runtime_ledger_file else {
            return Err(AgentError::Runtime(
                "no runtime ledger is configured".to_string(),
            ));
        };
        let ledger = JsonlLedger::open(path)
            .map_err(|error| AgentError::Runtime(format!("{}: {error}", path.display())))?;
        let ledger = Box::new(ledger) as Box<dyn EventLedger + Send>;
        let mut runtime = RuntimeCoordinator::recover_latest(
            runtime_policy(
                &self.tools.definitions(),
                self.policy.tool_timeout_secs,
                self.allow_irreversible_tools,
            ),
            ledger,
        )
        .map_err(|error| AgentError::Runtime(error.to_string()))?
        .ok_or(AgentError::RuntimeActionNotFound { action_id })?;
        let grant = runtime
            .pending_actions()
            .iter()
            .find(|grant| grant.action_id.as_uuid() == action_id)
            .cloned()
            .ok_or(AgentError::RuntimeActionNotFound { action_id })?;
        let outcome = match outcome {
            RuntimeReconcileOutcome::Succeeded { output, evidence } => {
                ReconciliationOutcome::Succeeded { output, evidence }
            }
            RuntimeReconcileOutcome::Failed { error, evidence } => {
                ReconciliationOutcome::Failed { error, evidence }
            }
        };
        runtime
            .reconcile(grant, outcome, unix_timestamp_ms())
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
        Ok(())
    }
}
