use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use super::super::updates::ToolUpdate;
use super::ToolExecutionContext;

#[derive(Debug, Default)]
pub(crate) struct UpdateGate {
    open: AtomicBool,
}

impl UpdateGate {
    pub(super) fn new() -> Self {
        Self {
            open: AtomicBool::new(true),
        }
    }
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
