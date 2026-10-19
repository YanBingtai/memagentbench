use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::{events::AgentEvent, model::ToolCallDelta};

pub(super) struct AttemptEvents<'a, F> {
    context: crate::events::RunContext,
    step: usize,
    emit: &'a F,
    output: Arc<AtomicBool>,
}

impl<'a, F> AttemptEvents<'a, F>
where
    F: Fn(AgentEvent) + Sync,
{
    pub(super) fn new(context: crate::events::RunContext, step: usize, emit: &'a F) -> Self {
        Self {
            context,
            step,
            emit,
            output: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(super) fn had_output(&self) -> bool {
        self.output.load(Ordering::Acquire)
    }

    pub(super) fn assistant_delta(&self, content: &str) {
        self.mark_output();
        (self.emit)(AgentEvent::AssistantDelta {
            context: self.context,
            step: self.step,
            content: content.to_string(),
        });
    }

    pub(super) fn reasoning_delta(&self, content: &str) {
        self.mark_output();
        (self.emit)(AgentEvent::ReasoningDelta {
            context: self.context,
            step: self.step,
            content: content.to_string(),
        });
    }

    pub(super) fn tool_call_delta(&self, delta: &ToolCallDelta) {
        self.mark_output();
        (self.emit)(AgentEvent::ToolCallDelta {
            context: self.context,
            step: self.step,
            index: delta.index,
            id: delta.id.clone(),
            call_type: delta.call_type.clone(),
            name: delta.name.clone(),
            arguments: delta.arguments.clone(),
        });
    }

    fn mark_output(&self) {
        self.output.store(true, Ordering::Release);
    }
}
