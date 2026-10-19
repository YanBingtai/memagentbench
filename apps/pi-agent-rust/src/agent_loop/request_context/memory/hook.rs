//! Memory hook request construction, execution, and preview validation.

use std::time::Duration;

use pi_agent_runtime::{
    EventLedger, InjectionDecision, MemoryHookContext, MemoryProposal, RunId, RuntimeCoordinator,
};

use crate::{events::RunContext, message::Message};

use super::super::super::{
    context::truncate_context, timeout_error, AgentError, CancellationToken, MemoryHookConfig,
    MemoryHookRequest, MemoryProposalHook,
};

use super::format;

pub(super) fn build_context(
    config: &MemoryHookConfig,
    runtime_run_id: RunId,
    run: RunContext,
    runtime: &RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    now_ms: u64,
) -> MemoryHookContext {
    MemoryHookContext {
        run_id: runtime_run_id,
        turn_id: Some(run.turn_id),
        namespace: config.namespace.clone(),
        current_revision: runtime.current_revision(),
        memory_revision: config.memory_revision,
        policy_revision: config.policy_revision,
        now_ms,
        max_tokens: config.max_tokens,
        max_candidates: config.max_candidates,
        max_projection_bytes: config.max_projection_bytes,
    }
}

pub(super) async fn run_hook(
    hook: &MemoryProposalHook,
    context: MemoryHookContext,
    messages: &[Message],
    deadline: Option<(tokio::time::Instant, Duration)>,
    cancellation: &CancellationToken,
) -> Result<Option<MemoryProposal>, AgentError> {
    let hook_request = hook(MemoryHookRequest {
        context,
        messages: messages.to_vec(),
    });
    match deadline {
        Some((deadline, timeout)) => tokio::select! {
            result = tokio::time::timeout_at(deadline, hook_request) => {
                result
                    .map_err(|_| timeout_error(timeout))?
                    .map_err(AgentError::MemoryHook)
            }
            _ = cancellation.cancelled() => Err(AgentError::RunCancelled),
        },
        None => tokio::select! {
            result = hook_request => result.map_err(AgentError::MemoryHook),
            _ = cancellation.cancelled() => Err(AgentError::RunCancelled),
        },
    }
}

pub(super) fn validate_preview(
    proposal: &MemoryProposal,
    context: &MemoryHookContext,
    messages: &[Message],
    max_context_messages: usize,
    max_context_bytes: usize,
    validate_context_budget: bool,
) -> Result<(), AgentError> {
    let preview = proposal
        .evaluate(context)
        .map_err(|error| AgentError::Runtime(error.to_string()))?;
    if !validate_context_budget {
        return Ok(());
    }

    let InjectionDecision::Inject { injections, .. } = &preview else {
        return Ok(());
    };
    let Some(memory_message) = format::memory_message(injections) else {
        return Ok(());
    };

    let mut preview_messages = messages.to_vec();
    let system_index = format::system_message_index(&preview_messages);
    preview_messages.insert(system_index, memory_message);
    truncate_context(&preview_messages, max_context_messages, max_context_bytes)?;
    Ok(())
}
