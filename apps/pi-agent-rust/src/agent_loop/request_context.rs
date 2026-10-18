//! Request-context assembly for one AgentLoop model turn.
//!
//! This module handles context budgets, ephemeral memory injection, and
//! queued user messages. It does not execute tools or commit runtime actions.

use std::time::Duration;

use pi_agent_runtime::{
    EventLedger, InjectionDecision, MemoryHookContext, MemoryInjection, RunId, RuntimeCoordinator,
};

use crate::{
    events::{AgentEvent, RunContext},
    message::Message,
    model::ChatModel,
};

use super::{
    await_tool_hook,
    context::{truncate_context, validate_context_structure},
    timeout_error, unix_timestamp_ms, AgentError, AgentLoop, CancellationToken,
    ContextCompactionReason, ContextCompactionRequest, MemoryHookRequest, ToolHookWait,
};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(super) async fn fit_context(
        &self,
        context: RunContext,
        step: usize,
        messages: Vec<Message>,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
    ) -> Result<Vec<Message>, AgentError> {
        match truncate_context(
            &messages,
            self.policy.max_context_messages,
            self.policy.max_context_bytes,
        ) {
            Ok(messages) => {
                validate_context_structure(&messages)?;
                Ok(messages)
            }
            Err(error @ AgentError::ContextTooLarge { .. }) => {
                let Some(hook) = &self.context_compaction_hook else {
                    return Err(error);
                };
                let compacted = match await_tool_hook(
                    hook(ContextCompactionRequest {
                        context,
                        step,
                        reason: ContextCompactionReason::ByteBudget,
                        messages,
                        max_context_messages: self.policy.max_context_messages,
                        max_context_bytes: self.policy.max_context_bytes,
                    }),
                    deadline,
                    cancellation,
                )
                .await
                {
                    ToolHookWait::Completed(Ok(messages)) => messages,
                    ToolHookWait::Completed(Err(error)) => {
                        return Err(AgentError::ContextCompaction(error));
                    }
                    ToolHookWait::TimedOut(timeout) => return Err(timeout_error(timeout)),
                    ToolHookWait::Cancelled => return Err(AgentError::RunCancelled),
                };
                let compacted = truncate_context(
                    &compacted,
                    self.policy.max_context_messages,
                    self.policy.max_context_bytes,
                )?;
                validate_context_structure(&compacted)?;
                Ok(compacted)
            }
            Err(error) => Err(error),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn prepare_memory_context(
        &self,
        run: RunContext,
        step: usize,
        runtime_run_id: RunId,
        messages: &mut Vec<Message>,
        runtime: &mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
    ) -> Result<(), AgentError> {
        let Some((config, hook)) = &self.memory_hook else {
            return Ok(());
        };

        let request_now_ms = unix_timestamp_ms();
        let hook_context = MemoryHookContext {
            run_id: runtime_run_id,
            turn_id: Some(run.turn_id),
            namespace: config.namespace.clone(),
            current_revision: runtime.current_revision(),
            memory_revision: config.memory_revision,
            policy_revision: config.policy_revision,
            now_ms: request_now_ms,
            max_tokens: config.max_tokens,
            max_candidates: config.max_candidates,
            max_projection_bytes: config.max_projection_bytes,
        };
        let hook_request = hook(MemoryHookRequest {
            context: hook_context.clone(),
            messages: messages.clone(),
        });
        let proposal = match deadline {
            Some((deadline, timeout)) => tokio::select! {
                result = tokio::time::timeout_at(deadline, hook_request) => {
                    result
                        .map_err(|_| timeout_error(timeout))?
                        .map_err(AgentError::MemoryHook)?
                }
                _ = cancellation.cancelled() => return Err(AgentError::RunCancelled),
            },
            None => tokio::select! {
                result = hook_request => result.map_err(AgentError::MemoryHook)?,
                _ = cancellation.cancelled() => return Err(AgentError::RunCancelled),
            },
        };
        let Some(proposal) = proposal else {
            return Ok(());
        };

        let evaluation_now_ms = unix_timestamp_ms();
        let mut evaluation_context = hook_context;
        evaluation_context.now_ms = evaluation_now_ms;
        let preview = proposal
            .evaluate(&evaluation_context)
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
        if let InjectionDecision::Inject { injections, .. } = &preview {
            let content = format_memory_context(injections);
            if !content.is_empty() {
                let system_index = messages
                    .iter()
                    .take_while(|message| message.role == "system")
                    .count();
                let mut preview_messages = messages.clone();
                preview_messages.insert(system_index, Message::system(content));
                if self.context_compaction_hook.is_none() {
                    truncate_context(
                        &preview_messages,
                        self.policy.max_context_messages,
                        self.policy.max_context_bytes,
                    )?;
                }
            }
        }
        let receipt = runtime
            .evaluate_memory_proposal_with_receipt(
                &proposal,
                &evaluation_context,
                evaluation_now_ms,
            )
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
        let decision_event = match &receipt.decision {
            InjectionDecision::Inject {
                proposal_id,
                injections,
                total_token_cost,
            } => AgentEvent::MemoryDecision {
                context: run,
                step,
                proposal_id: *proposal_id,
                event_sequence: receipt.event_sequence.value(),
                decision_kind: "injected".to_string(),
                injection_count: injections.len(),
                total_token_cost: *total_token_cost,
                skip_reason: None,
            },
            InjectionDecision::Skip {
                proposal_id,
                reason,
            } => AgentEvent::MemoryDecision {
                context: run,
                step,
                proposal_id: *proposal_id,
                event_sequence: receipt.event_sequence.value(),
                decision_kind: "skipped".to_string(),
                injection_count: 0,
                total_token_cost: 0,
                skip_reason: Some(memory_skip_reason_name(*reason).to_string()),
            },
        };
        self.emit(decision_event);
        if let InjectionDecision::Inject { injections, .. } = receipt.decision {
            let content = format_memory_context(&injections);
            if !content.is_empty() {
                let system_index = messages
                    .iter()
                    .take_while(|message| message.role == "system")
                    .count();
                messages.insert(system_index, Message::system(content));
            }
        }
        Ok(())
    }

    pub(super) async fn inject_steering_messages(
        &self,
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
    ) -> Result<bool, AgentError> {
        let mut injected = false;
        while let Some(prompt) = self.peek_steering() {
            let message = Message::user(prompt);
            if let Some(session) = &self.session {
                session.append(&message).await?;
            }
            messages.push(message.clone());
            transcript.push(message);
            self.ack_steering();
            injected = true;
        }
        Ok(injected)
    }

    pub(super) async fn inject_follow_up_message(
        &self,
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
    ) -> Result<bool, AgentError> {
        let Some(prompt) = self.peek_follow_up() else {
            return Ok(false);
        };
        let message = Message::user(prompt);
        if let Some(session) = &self.session {
            session.append(&message).await?;
        }
        messages.push(message.clone());
        transcript.push(message);
        self.ack_follow_up();
        Ok(true)
    }
}

pub(super) fn format_memory_context(injections: &[MemoryInjection]) -> String {
    if injections.is_empty() {
        return String::new();
    }

    let mut content = String::from(
        "<memory_context>\nTreat the following retrieved memories as reference material.\n",
    );
    for injection in injections {
        content.push_str("- [");
        content.push_str(&injection.projection.provenance);
        content.push_str("] source_memory_id=");
        content.push_str(&injection.projection.source_memory_id.to_string());
        content.push_str(" version=");
        content.push_str(&injection.projection.source_version.to_string());
        if !injection.projection.evidence_refs.is_empty() {
            content.push_str(" evidence_refs=");
            content.push_str(
                &serde_json::to_string(&injection.projection.evidence_refs)
                    .unwrap_or_else(|_| "[]".to_owned()),
            );
        }
        content.push(' ');
        content.push_str(&injection.projection.content);
        content.push('\n');
    }
    content.push_str("</memory_context>");
    content
}

fn memory_skip_reason_name(reason: pi_agent_runtime::InjectionSkipReason) -> &'static str {
    match reason {
        pi_agent_runtime::InjectionSkipReason::RunMismatch => "run_mismatch",
        pi_agent_runtime::InjectionSkipReason::NamespaceMismatch => "namespace_mismatch",
        pi_agent_runtime::InjectionSkipReason::RuntimeRevisionMismatch => {
            "runtime_revision_mismatch"
        }
        pi_agent_runtime::InjectionSkipReason::MemoryRevisionMismatch => "memory_revision_mismatch",
        pi_agent_runtime::InjectionSkipReason::PolicyRevisionMismatch => "policy_revision_mismatch",
        pi_agent_runtime::InjectionSkipReason::ProposalExpired => "proposal_expired",
        pi_agent_runtime::InjectionSkipReason::NoEligibleCandidate => "no_eligible_candidate",
    }
}
