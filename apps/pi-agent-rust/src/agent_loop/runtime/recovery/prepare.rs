//! Runtime ledger creation and recovery before an agent run.

use std::path::Path;

use pi_agent_runtime::{
    AdmissionPolicy, EventLedger, InMemoryLedger, JsonlLedger, RunId, RuntimeCoordinator,
};

use crate::model::ChatModel;

use super::super::{
    super::{AgentError, AgentLoop},
    policy::runtime_policy,
};

type RuntimeLedger = Box<dyn EventLedger + Send>;
type Runtime = RuntimeCoordinator<RuntimeLedger>;

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(crate) fn prepare_runtime(
        &self,
        fresh_runtime_run_id: RunId,
        allow_recovery: bool,
    ) -> Result<Runtime, AgentError> {
        let runtime_definitions = self.tools.definitions();
        let policy = runtime_policy(
            &runtime_definitions,
            self.policy.tool_timeout_secs,
            self.allow_irreversible_tools,
        );
        match &self.runtime_ledger_file {
            Some(path) => {
                self.prepare_file_runtime(path, fresh_runtime_run_id, allow_recovery, policy)
            }
            None => Ok(new_runtime(
                fresh_runtime_run_id,
                policy,
                Box::new(InMemoryLedger::default()),
            )),
        }
    }

    fn prepare_file_runtime(
        &self,
        path: &Path,
        fresh_runtime_run_id: RunId,
        allow_recovery: bool,
        policy: AdmissionPolicy,
    ) -> Result<Runtime, AgentError> {
        let recovered = recover_runtime(path, policy.clone())?;
        if allow_recovery && self.runtime_recovery {
            return match recovered {
                Some(runtime) if !runtime.pending_actions().is_empty() => Err(AgentError::Runtime(
                    "runtime recovery found unresolved actions; reconcile them before resuming the agent"
                        .to_string(),
                )),
                Some(runtime) => Ok(runtime),
                None => new_file_runtime(path, fresh_runtime_run_id, policy),
            };
        }

        if recovered
            .as_ref()
            .is_some_and(|runtime| !runtime.pending_actions().is_empty())
        {
            return Err(AgentError::Runtime(
                "runtime ledger contains unresolved actions; use explicit runtime recovery or reconcile them before starting a new run"
                    .to_string(),
            ));
        }
        drop(recovered);
        new_file_runtime(path, fresh_runtime_run_id, policy)
    }
}

fn recover_runtime(path: &Path, policy: AdmissionPolicy) -> Result<Option<Runtime>, AgentError> {
    RuntimeCoordinator::recover_latest(policy, open_runtime_ledger(path)?)
        .map_err(|error| AgentError::Runtime(error.to_string()))
}

fn new_file_runtime(
    path: &Path,
    run_id: RunId,
    policy: AdmissionPolicy,
) -> Result<Runtime, AgentError> {
    Ok(new_runtime(run_id, policy, open_runtime_ledger(path)?))
}

fn new_runtime(run_id: RunId, policy: AdmissionPolicy, ledger: RuntimeLedger) -> Runtime {
    RuntimeCoordinator::new(run_id, policy, ledger)
}

fn open_runtime_ledger(path: &Path) -> Result<RuntimeLedger, AgentError> {
    let ledger = JsonlLedger::open(path)
        .map_err(|error| AgentError::Runtime(format!("{}: {error}", path.display())))?;
    Ok(Box::new(ledger))
}
