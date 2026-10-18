//! Cooperative cancellation and bounded progress updates.

use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::agent_loop::CancellationToken;

use super::super::errors::ToolError;
use super::super::limits::DEFAULT_TOOL_UPDATE_BYTES;
use super::result::ToolResult;

pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + 'a>>;
/// An incremental update emitted while a tool is running.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolUpdate {
    pub content: String,
    #[serde(default)]
    pub details: Value,
}

impl ToolUpdate {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            details: Value::Null,
        }
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }

    /// Bound a progress update before it reaches an event sink.
    pub fn bounded(self, max_bytes: usize) -> Self {
        let result = ToolResult::text(self.content)
            .with_details(self.details)
            .bounded(max_bytes);
        Self {
            content: result.content,
            details: result.details,
        }
    }
}

pub type ToolUpdateSink = Arc<dyn Fn(ToolUpdate) + Send + Sync>;

/// Cooperative cancellation and progress reporting for one tool call.
#[derive(Clone)]
pub struct ToolExecutionContext {
    cancellation: CancellationToken,
    update_sink: Option<ToolUpdateSink>,
    max_update_bytes: usize,
    update_gate: Arc<UpdateGate>,
    parent_run_id: Option<Uuid>,
    parent_turn_id: Option<Uuid>,
    parent_step: Option<usize>,
    parent_tool_call_id: Option<String>,
    delegation_depth: u16,
}

#[derive(Debug, Default)]
pub(crate) struct UpdateGate {
    open: AtomicBool,
}

pub(crate) struct UpdateLease {
    gate: Arc<UpdateGate>,
}

impl Drop for UpdateLease {
    fn drop(&mut self) {
        self.gate.open.store(false, Ordering::Release);
    }
}

impl ToolExecutionContext {
    pub fn new(cancellation: CancellationToken) -> Self {
        Self {
            cancellation,
            update_sink: None,
            max_update_bytes: DEFAULT_TOOL_UPDATE_BYTES,
            update_gate: Arc::new(UpdateGate {
                open: AtomicBool::new(true),
            }),
            parent_run_id: None,
            parent_turn_id: None,
            parent_step: None,
            parent_tool_call_id: None,
            delegation_depth: 0,
        }
    }

    /// Set the maximum content/details size for progress updates.
    pub fn with_output_limit(mut self, max_bytes: usize) -> Self {
        self.max_update_bytes = max_bytes;
        self
    }

    pub fn with_update_sink<F>(mut self, sink: F) -> Self
    where
        F: Fn(ToolUpdate) + Send + Sync + 'static,
    {
        self.update_sink = Some(Arc::new(sink));
        self
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Clone the run-wide cancellation signal for a delegated operation.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    /// Attach the identity of the parent agent call that owns this execution.
    ///
    /// Tools may use this for child-agent lineage and telemetry. It is
    /// metadata only; runtime admission and authorization remain outside the
    /// tool context.
    pub fn with_parent_identity(
        mut self,
        run_id: Uuid,
        turn_id: Uuid,
        step: usize,
        tool_call_id: impl Into<String>,
    ) -> Self {
        self.parent_run_id = Some(run_id);
        self.parent_turn_id = Some(turn_id);
        self.parent_step = Some(step);
        self.parent_tool_call_id = Some(tool_call_id.into());
        self
    }

    pub fn parent_run_id(&self) -> Option<Uuid> {
        self.parent_run_id
    }

    pub fn parent_turn_id(&self) -> Option<Uuid> {
        self.parent_turn_id
    }

    pub fn parent_step(&self) -> Option<usize> {
        self.parent_step
    }

    pub fn parent_tool_call_id(&self) -> Option<&str> {
        self.parent_tool_call_id.as_deref()
    }

    /// Set the current agent's delegation depth for child-agent lineage.
    pub fn with_delegation_depth(mut self, depth: u16) -> Self {
        self.delegation_depth = depth;
        self
    }

    pub fn delegation_depth(&self) -> u16 {
        self.delegation_depth
    }

    /// Close the progress channel after a tool reaches a terminal state.
    ///
    /// A tool may retain a cloned context for cleanup work. Once this method
    /// is called, those late updates are ignored instead of appearing after
    /// the corresponding `ToolFinished` event.
    pub fn close_updates(&self) {
        self.update_gate.open.store(false, Ordering::Release);
    }

    pub fn update(&self, update: ToolUpdate) {
        if !self.update_gate.open.load(Ordering::Acquire) {
            return;
        }
        if let Some(sink) = &self.update_sink {
            sink(update.bounded(self.max_update_bytes));
        }
    }

    pub(crate) fn lease_updates(&self) -> UpdateLease {
        UpdateLease {
            gate: Arc::clone(&self.update_gate),
        }
    }
}
