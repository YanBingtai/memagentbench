use crate::{
    events::{AgentEvent, RunContext},
    message::ToolCall,
    model::ChatModel,
    tools::ToolExecutionContext,
};

use super::super::{AgentLoop, CancellationToken};

/// Build the progress and lineage context shared by every tool execution path.
pub(crate) fn build_context<M>(
    agent: &AgentLoop<M>,
    context: RunContext,
    step: usize,
    call: &ToolCall,
    cancellation: CancellationToken,
) -> ToolExecutionContext
where
    M: ChatModel,
{
    let event_sink = agent.event_sink.clone();
    let call_id = call.id.clone();
    let tool_name = call.function.name.clone();

    ToolExecutionContext::new(cancellation)
        .with_output_limit(agent.tools.output_limit())
        .with_parent_identity(context.run_id, context.turn_id, step, call_id.clone())
        .with_delegation_depth(agent.delegation_depth)
        .with_update_sink(move |update| {
            if let Some(sink) = &event_sink {
                sink.emit(AgentEvent::ToolUpdate {
                    context,
                    step,
                    call_id: call_id.clone(),
                    tool_name: tool_name.clone(),
                    content: update.content,
                    details: update.details,
                });
            }
        })
}
