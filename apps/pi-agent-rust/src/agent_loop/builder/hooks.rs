//! Asynchronous event and lifecycle hook configuration.

use std::sync::Arc;

use crate::model::ChatModel;

use super::super::{
    AfterRunFuture, AfterRunRequest, AgentLoop, BeforeToolDecision, ContextCompactionFuture,
    ContextCompactionRequest, EventSink, MemoryHookConfig, MemoryHookFuture, MemoryHookRequest,
    ToolHookContext, ToolHookFuture, ToolResultContext, ToolResultOverride,
};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    /// Attach a sink for ephemeral runtime events.
    pub fn with_event_sink<S>(mut self, sink: S) -> Self
    where
        S: EventSink + 'static,
    {
        self.event_sink = Some(Arc::new(sink));
        self
    }

    /// Install an asynchronous hook that can block a tool before admission.
    pub fn with_before_tool_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(ToolHookContext) -> ToolHookFuture<BeforeToolDecision> + Send + Sync + 'static,
    {
        self.before_tool_hook = Some(Arc::new(hook));
        self
    }

    /// Install an asynchronous hook that can audit or replace a tool result.
    pub fn with_after_tool_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(ToolResultContext) -> ToolHookFuture<ToolResultOverride> + Send + Sync + 'static,
    {
        self.after_tool_hook = Some(Arc::new(hook));
        self
    }

    /// Install an asynchronous provider-agnostic memory proposal hook.
    ///
    /// The hook prepares a proposal only. The runtime validates its run,
    /// revision, expiry, candidate count, projection size, and token budget
    /// immediately before the model request. Accepted projections are added
    /// as ephemeral system context and are not written to the session.
    pub fn with_memory_hook<F>(mut self, config: MemoryHookConfig, hook: F) -> Self
    where
        F: Fn(MemoryHookRequest) -> MemoryHookFuture + Send + Sync + 'static,
    {
        self.memory_hook = Some((config, Arc::new(hook)));
        self
    }

    /// Install an asynchronous context compactor used only after a context
    /// budget is exceeded. The transformed messages stay ephemeral and are
    /// checked against the same message and byte limits before the request.
    pub fn with_context_compactor<F>(mut self, hook: F) -> Self
    where
        F: Fn(ContextCompactionRequest) -> ContextCompactionFuture + Send + Sync + 'static,
    {
        self.context_compaction_hook = Some(Arc::new(hook));
        self
    }

    /// Install an asynchronous hook that observes the completed run.
    ///
    /// The hook runs before the terminal `RunFinished` event is emitted. A
    /// hook error is returned to the caller as [`crate::agent_loop::AgentError::AfterRunHook`],
    /// so persistence failures cannot disappear behind a successful model
    /// response.
    pub fn with_after_run_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(AfterRunRequest) -> AfterRunFuture + Send + Sync + 'static,
    {
        self.after_run_hook = Some(Arc::new(hook));
        self
    }
}
