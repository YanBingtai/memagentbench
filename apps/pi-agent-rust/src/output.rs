use pi_agent_rust::{AgentEvent, RunOutcome};

pub(super) fn print_event(event: AgentEvent) {
    match event {
        AgentEvent::ModelStarted { step, model, .. } => {
            eprintln!("[agent] model step {step}: {model}");
        }
        AgentEvent::ModelRetry {
            step,
            attempt,
            delay_ms,
            error,
            ..
        } => {
            eprintln!("[agent] model retry {attempt} after step {step} in {delay_ms}ms: {error}");
        }
        AgentEvent::ToolStarted { step, call, .. } => {
            eprintln!(
                "[agent] tool step {step}: {} ({})",
                call.function.name, call.id
            );
        }
        AgentEvent::ToolFinished {
            tool_name,
            status,
            duration_ms,
            ..
        } => {
            eprintln!("[agent] tool {tool_name}: {status} ({duration_ms}ms)");
        }
        AgentEvent::ToolUpdate {
            tool_name, content, ..
        } => {
            eprintln!("[agent] tool {tool_name}: {content}");
        }
        AgentEvent::RunFinished {
            steps,
            outcome,
            error,
            ..
        } => match (outcome, error) {
            (RunOutcome::Completed, _) => eprintln!("[agent] completed in {steps} step(s)"),
            (RunOutcome::Unknown, Some(error)) => {
                eprintln!("[agent] outcome unknown after {steps} step(s): {error}")
            }
            (outcome, Some(error)) => {
                eprintln!("[agent] {outcome:?} after {steps} step(s): {error}")
            }
            (outcome, None) => eprintln!("[agent] {outcome:?} after {steps} step(s)"),
        },
        AgentEvent::TurnStarted { step, .. } => {
            eprintln!("[agent] turn {step} started");
        }
        AgentEvent::TurnFinished { step, .. } => {
            eprintln!("[agent] turn {step} finished");
        }
        AgentEvent::MemoryDecision {
            step,
            decision_kind,
            injection_count,
            ..
        } => {
            eprintln!(
                "[agent] memory step {step}: {decision_kind} ({injection_count} injection(s))"
            );
        }
        AgentEvent::RunStarted { .. }
        | AgentEvent::AssistantDelta { .. }
        | AgentEvent::ReasoningDelta { .. }
        | AgentEvent::ToolCallDelta { .. }
        | AgentEvent::AssistantStreamReset { .. }
        | AgentEvent::AssistantMessage { .. } => {}
    }
}
