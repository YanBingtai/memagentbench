//! Cooperative cancellation shared by the agent and its callers.

use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex as StdMutex,
    },
};

use tokio::sync::Notify;

use super::{AgentError, AgentLoop};

const MAX_QUEUED_PROMPTS: usize = 32;
const MAX_QUEUED_PROMPT_BYTES: usize = 64 * 1024;

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

#[derive(Debug, Default)]
pub(super) struct ControlState {
    steering: VecDeque<String>,
    follow_up: VecDeque<String>,
    active: bool,
    has_run: bool,
}

pub(super) struct RunGuard {
    control: Arc<StdMutex<ControlState>>,
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        let mut control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        control.steering.clear();
        control.follow_up.clear();
        control.active = false;
    }
}

impl<M> AgentLoop<M> {
    /// Queue a user message for the next safe model boundary in this run.
    pub fn steer(&self, prompt: impl Into<String>) -> Result<(), AgentError> {
        self.enqueue_prompt("steering", prompt.into(), |control| &mut control.steering)
    }

    /// Queue one user message to run after the current assistant turn finishes.
    pub fn follow_up(&self, prompt: impl Into<String>) -> Result<(), AgentError> {
        self.enqueue_prompt("follow-up", prompt.into(), |control| &mut control.follow_up)
    }

    fn enqueue_prompt(
        &self,
        kind: &'static str,
        prompt: String,
        queue: impl FnOnce(&mut ControlState) -> &mut VecDeque<String>,
    ) -> Result<(), AgentError> {
        if prompt.trim().is_empty() {
            return Err(AgentError::EmptyQueuedPrompt { kind });
        }
        if prompt.len() > MAX_QUEUED_PROMPT_BYTES {
            return Err(AgentError::QueuedPromptTooLarge {
                kind,
                limit: MAX_QUEUED_PROMPT_BYTES,
            });
        }
        let mut control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !control.active && control.has_run {
            return Err(AgentError::ControlQueueRequiresActiveRun { kind });
        }
        let queue = queue(&mut control);
        if queue.len() >= MAX_QUEUED_PROMPTS {
            return Err(AgentError::QueueFull {
                kind,
                limit: MAX_QUEUED_PROMPTS,
            });
        }
        queue.push_back(prompt);
        Ok(())
    }

    pub(super) fn peek_steering(&self) -> Option<String> {
        self.control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .steering
            .front()
            .cloned()
    }

    pub(super) fn ack_steering(&self) {
        let _ = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .steering
            .pop_front();
    }

    pub(super) fn peek_follow_up(&self) -> Option<String> {
        self.control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .follow_up
            .front()
            .cloned()
    }

    pub(super) fn ack_follow_up(&self) {
        let _ = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .follow_up
            .pop_front();
    }

    pub(super) fn start_run(&self) -> Result<RunGuard, AgentError> {
        let mut control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if control.active {
            return Err(AgentError::RunBusy);
        }
        control.active = true;
        Ok(RunGuard {
            control: Arc::clone(&self.control),
        })
    }

    pub(super) fn mark_run_started(&self) {
        self.control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .has_run = true;
    }

    pub(super) fn ensure_idle(&self) -> Result<(), AgentError> {
        let control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if control.active {
            return Err(AgentError::RunBusy);
        }
        Ok(())
    }
}
