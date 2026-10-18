//! Runtime ledger lifecycle and session reconciliation support.
//!
//! This module owns runtime admission recovery and durable reconciliation
//! helpers. The main loop retains scheduling and terminal event ordering.

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};
use uuid::Uuid;

use pi_agent_runtime::{
    AdmissionPolicy, Budget, CommitGrant, EventLedger, EventSequence, InMemoryLedger, JsonlLedger,
    ReconciliationOutcome, ResourceScope, RunId, RuntimeCoordinator,
};

use crate::{
    message::{Message, ToolDefinition},
    model::ChatModel,
    tools::{ToolGateway, ToolReplayPolicy, ToolResult},
};

use super::{
    unix_timestamp_ms, AgentError, AgentLoop, PendingRuntimeAction, RuntimeReconcileOutcome,
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
        .ok_or_else(|| AgentError::RuntimeActionNotFound { action_id })?;
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

    pub(super) fn prepare_runtime(
        &self,
        fresh_runtime_run_id: RunId,
        allow_recovery: bool,
    ) -> Result<RuntimeCoordinator<Box<dyn EventLedger + Send>>, AgentError> {
        let runtime_definitions = self.tools.definitions();
        let runtime_policy = runtime_policy(
            &runtime_definitions,
            self.policy.tool_timeout_secs,
            self.allow_irreversible_tools,
        );
        let (_, runtime): (RunId, RuntimeCoordinator<Box<dyn EventLedger + Send>>) = if let Some(
            path,
        ) =
            &self.runtime_ledger_file
        {
            let ledger = JsonlLedger::open(path)
                .map_err(|error| AgentError::Runtime(format!("{}: {error}", path.display())))?;
            let ledger = Box::new(ledger) as Box<dyn EventLedger + Send>;
            if allow_recovery && self.runtime_recovery {
                match RuntimeCoordinator::recover_latest(runtime_policy.clone(), ledger)
                    .map_err(|error| AgentError::Runtime(error.to_string()))?
                {
                    Some(runtime) if !runtime.pending_actions().is_empty() => {
                        return Err(AgentError::Runtime(
                            "runtime recovery found unresolved actions; reconcile them before resuming the agent"
                                .to_string(),
                        ));
                    }
                    Some(runtime) => (runtime.run_id(), runtime),
                    None => {
                        let ledger = JsonlLedger::open(path).map_err(|error| {
                            AgentError::Runtime(format!("{}: {error}", path.display()))
                        })?;
                        (
                            fresh_runtime_run_id,
                            RuntimeCoordinator::new(
                                fresh_runtime_run_id,
                                runtime_policy,
                                Box::new(ledger) as Box<dyn EventLedger + Send>,
                            ),
                        )
                    }
                }
            } else {
                if let Some(recovered) =
                    RuntimeCoordinator::recover_latest(runtime_policy.clone(), ledger)
                        .map_err(|error| AgentError::Runtime(error.to_string()))?
                {
                    if !recovered.pending_actions().is_empty() {
                        return Err(AgentError::Runtime(
                            "runtime ledger contains unresolved actions; use explicit runtime recovery or reconcile them before starting a new run"
                                .to_string(),
                        ));
                    }
                }
                let ledger = JsonlLedger::open(path)
                    .map_err(|error| AgentError::Runtime(format!("{}: {error}", path.display())))?;
                (
                    fresh_runtime_run_id,
                    RuntimeCoordinator::new(
                        fresh_runtime_run_id,
                        runtime_policy,
                        Box::new(ledger) as Box<dyn EventLedger + Send>,
                    ),
                )
            }
        } else {
            (
                fresh_runtime_run_id,
                RuntimeCoordinator::new(
                    fresh_runtime_run_id,
                    runtime_policy,
                    Box::new(InMemoryLedger::default()) as Box<dyn EventLedger + Send>,
                ),
            )
        };
        Ok(runtime)
    }
}

pub(super) fn pending_runtime_action(
    grant: &CommitGrant,
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

pub(super) fn runtime_policy(
    definitions: &[ToolDefinition],
    tool_timeout_secs: u64,
    allow_irreversible_tools: bool,
) -> AdmissionPolicy {
    let mut policy = AdmissionPolicy::deny_all().with_budget_limit(Budget {
        wall_time_ms: Some(tool_timeout_secs.saturating_mul(1_000)),
        tokens: None,
        tool_calls: Some(1),
    });
    if allow_irreversible_tools {
        policy = policy.allow_irreversible();
    }
    for definition in definitions {
        let name = definition.function.name.clone();
        policy = policy
            .allow_tool(name.clone())
            .allow_resources(ResourceScope::new([format!("tool:{name}")]));
    }
    policy
}

pub(super) fn validate_session_runtime_consistency<L: EventLedger>(
    messages: &[Message],
    runtime: &RuntimeCoordinator<L>,
) -> Result<(), AgentError> {
    let events = runtime
        .ledger()
        .scan_after(EventSequence::INITIAL)
        .map_err(|error| AgentError::Runtime(error.to_string()))?;
    let mut admitted = HashMap::new();
    let mut terminal = HashSet::new();
    for event in events {
        let Some(action_id) = event.draft.payload.get("action_id").and_then(Value::as_str) else {
            continue;
        };
        if event.draft.kind.as_str() == pi_agent_runtime::ACTION_ADMITTED_EVENT {
            if let Some(grant_id) = event.draft.payload.get("grant_id").and_then(Value::as_str) {
                admitted.insert(action_id.to_string(), grant_id.to_string());
            }
        }
        if matches!(
            event.draft.kind.as_str(),
            pi_agent_runtime::ACTION_SUCCEEDED_EVENT
                | pi_agent_runtime::ACTION_FAILED_EVENT
                | pi_agent_runtime::ACTION_UNKNOWN_EVENT
                | pi_agent_runtime::ACTION_RECONCILED_EVENT
        ) {
            terminal.insert(action_id.to_string());
        }
    }

    for message in messages {
        let Some(metadata) = message.tool_result.as_ref() else {
            continue;
        };
        let Some(action_id) = metadata.action_id else {
            if let Some(status) = metadata.details.get("status").and_then(Value::as_str) {
                let anonymous_outcome_is_uncertain = match status {
                    "unknown" | "timed_out" => true,
                    // Cancellation before runtime admission cannot have
                    // caused an external side effect. If the execution bit
                    // is absent or true, the result still needs reconciliation.
                    "cancelled" => metadata.details.get("executed") != Some(&Value::Bool(false)),
                    _ => false,
                };
                if anonymous_outcome_is_uncertain {
                    return Err(AgentError::UnidentifiedToolOutcome {
                        call_id: message.tool_call_id.clone().unwrap_or_default(),
                        status: status.to_string(),
                    });
                }
            }
            continue;
        };
        let action_id_text = action_id.to_string();
        let Some(expected_grant_id) = admitted.get(&action_id_text) else {
            return Err(AgentError::RuntimeSessionMismatch {
                action_id,
                reason: "durable admission event is missing".to_string(),
            });
        };
        if !terminal.contains(&action_id_text) {
            return Err(AgentError::RuntimeSessionMismatch {
                action_id,
                reason: "durable terminal event is missing".to_string(),
            });
        }
        let Some(grant_id) = metadata.grant_id else {
            return Err(AgentError::RuntimeSessionMismatch {
                action_id,
                reason: "session result is missing its grant identity".to_string(),
            });
        };
        if grant_id.to_string() != *expected_grant_id {
            return Err(AgentError::RuntimeSessionMismatch {
                action_id,
                reason: format!(
                    "grant identity {grant_id} does not match durable grant {expected_grant_id}"
                ),
            });
        }
    }
    Ok(())
}

pub(super) fn apply_reconciled_tool_results<L: EventLedger>(
    messages: &mut [Message],
    runtime: &RuntimeCoordinator<L>,
    max_output_bytes: usize,
) -> Result<(), AgentError> {
    let events = runtime
        .ledger()
        .scan_after(EventSequence::INITIAL)
        .map_err(|error| AgentError::Runtime(error.to_string()))?;
    let mut reconciled = HashMap::new();
    for event in events {
        if event.draft.kind.as_str() != pi_agent_runtime::ACTION_RECONCILED_EVENT {
            continue;
        }
        let Some(action_id) = event.draft.payload.get("action_id").and_then(Value::as_str) else {
            continue;
        };
        reconciled.insert(action_id.to_string(), event.draft.payload);
    }

    for message in messages {
        let Some(metadata) = message.tool_result.as_mut() else {
            continue;
        };
        let Some(action_id) = metadata.action_id else {
            continue;
        };
        let Some(event) = reconciled.get(&action_id.to_string()) else {
            continue;
        };
        let outcome = event
            .get("outcome")
            .and_then(Value::as_str)
            .unwrap_or("failed");
        let is_error = outcome == "failed";
        let content = if is_error {
            event
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("runtime action failed during reconciliation")
                .to_string()
        } else {
            match event.get("output").cloned().unwrap_or(Value::Null) {
                Value::String(content) => content,
                output => output.to_string(),
            }
        };
        let bounded = ToolResult {
            content,
            details: json!({
                "status": "reconciled",
                "outcome": outcome,
                "evidence": event.get("evidence").cloned().unwrap_or(Value::Null),
                "executed": true,
            }),
            is_error,
            terminate: metadata.terminate,
        }
        .bounded(max_output_bytes);
        message.content = Some(bounded.content);
        metadata.details = bounded.details;
        metadata.is_error = bounded.is_error;
    }
    Ok(())
}

pub(super) fn commit_tool_result<L: EventLedger>(
    runtime: &mut RuntimeCoordinator<L>,
    grant: CommitGrant,
    result: &ToolResult,
) -> Result<(), AgentError> {
    if result.is_error {
        runtime
            .fail(grant, result.content.clone(), unix_timestamp_ms())
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
    } else {
        let output = serde_json::to_value(result).map_err(|error| {
            AgentError::Runtime(format!(
                "failed to serialize normalized tool result: {error}"
            ))
        })?;
        runtime
            .complete(grant, output, unix_timestamp_ms())
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
    }
    Ok(())
}
