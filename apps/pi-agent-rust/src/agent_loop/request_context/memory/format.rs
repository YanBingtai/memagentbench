//! Memory decision events and model-context formatting.

use pi_agent_runtime::{InjectionDecision, InjectionSkipReason, MemoryInjection};

use crate::{
    events::{AgentEvent, RunContext},
    message::Message,
};

pub(super) fn memory_message(injections: &[MemoryInjection]) -> Option<Message> {
    let content = format_memory_context(injections);
    (!content.is_empty()).then(|| Message::system(content))
}

pub(super) fn insert_memory_context(messages: &mut Vec<Message>, decision: &InjectionDecision) {
    let InjectionDecision::Inject { injections, .. } = decision else {
        return;
    };
    let Some(memory_message) = memory_message(injections) else {
        return;
    };
    let system_index = system_message_index(messages);
    messages.insert(system_index, memory_message);
}

pub(super) fn system_message_index(messages: &[Message]) -> usize {
    messages
        .iter()
        .take_while(|message| message.role == "system")
        .count()
}

pub(super) fn decision_event(
    context: RunContext,
    step: usize,
    event_sequence: u64,
    decision: &InjectionDecision,
) -> AgentEvent {
    match decision {
        InjectionDecision::Inject {
            proposal_id,
            injections,
            total_token_cost,
        } => AgentEvent::MemoryDecision {
            context,
            step,
            proposal_id: *proposal_id,
            event_sequence,
            decision_kind: "injected".to_string(),
            injection_count: injections.len(),
            total_token_cost: *total_token_cost,
            skip_reason: None,
        },
        InjectionDecision::Skip {
            proposal_id,
            reason,
        } => AgentEvent::MemoryDecision {
            context,
            step,
            proposal_id: *proposal_id,
            event_sequence,
            decision_kind: "skipped".to_string(),
            injection_count: 0,
            total_token_cost: 0,
            skip_reason: Some(memory_skip_reason_name(*reason).to_string()),
        },
    }
}

pub(in crate::agent_loop) fn format_memory_context(injections: &[MemoryInjection]) -> String {
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

fn memory_skip_reason_name(reason: InjectionSkipReason) -> &'static str {
    match reason {
        InjectionSkipReason::RunMismatch => "run_mismatch",
        InjectionSkipReason::NamespaceMismatch => "namespace_mismatch",
        InjectionSkipReason::RuntimeRevisionMismatch => "runtime_revision_mismatch",
        InjectionSkipReason::MemoryRevisionMismatch => "memory_revision_mismatch",
        InjectionSkipReason::PolicyRevisionMismatch => "policy_revision_mismatch",
        InjectionSkipReason::ProposalExpired => "proposal_expired",
        InjectionSkipReason::NoEligibleCandidate => "no_eligible_candidate",
    }
}
