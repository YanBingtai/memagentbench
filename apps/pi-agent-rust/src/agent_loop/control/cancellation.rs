//! Cooperative cancellation shared by the agent and its callers.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use tokio::sync::Notify;

/// A clonable cancellation signal shared by an agent run and its caller.
///
/// Cancellation is cooperative: the loop stops waiting for the current model
/// or tool future, then persists structural tool results before returning.
/// Clones all observe the same request, so a CLI, TUI, or service handler can
/// retain one clone while the agent owns another.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    state: Arc<CancellationState>,
}

#[derive(Debug, Default)]
struct CancellationState {
    requested: AtomicBool,
    notify: Notify,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Calling this more than once is harmless.
    pub fn cancel(&self) {
        if !self.state.requested.swap(true, Ordering::Release) {
            self.state.notify.notify_waiters();
            // Keep one permit for a waiter that starts between the atomic
            // check and registration with Notify.
            self.state.notify.notify_one();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.requested.load(Ordering::Acquire)
    }

    /// Wait until cancellation is requested.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let notified = self.state.notify.notified();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}
