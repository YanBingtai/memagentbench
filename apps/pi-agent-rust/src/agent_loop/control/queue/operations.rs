use std::collections::VecDeque;
use std::sync::{Arc, MutexGuard as StdMutexGuard};

use crate::agent_loop::{AgentError, AgentLoop};

use super::{ControlState, RunGuard, MAX_QUEUED_PROMPTS, MAX_QUEUED_PROMPT_BYTES};

impl<M> AgentLoop<M> {
    fn lock_control(&self) -> StdMutexGuard<'_, ControlState> {
        self.control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

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
        let mut control = self.lock_control();
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

    pub(in crate::agent_loop) fn peek_steering(&self) -> Option<String> {
        self.lock_control().steering.front().cloned()
    }

    pub(in crate::agent_loop) fn ack_steering(&self) {
        let _ = self.lock_control().steering.pop_front();
    }

    pub(in crate::agent_loop) fn peek_follow_up(&self) -> Option<String> {
        self.lock_control().follow_up.front().cloned()
    }

    pub(in crate::agent_loop) fn ack_follow_up(&self) {
        let _ = self.lock_control().follow_up.pop_front();
    }

    pub(in crate::agent_loop) fn start_run(&self) -> Result<RunGuard, AgentError> {
        let mut control = self.lock_control();
        if control.active {
            return Err(AgentError::RunBusy);
        }
        control.active = true;
        Ok(RunGuard::new(Arc::clone(&self.control)))
    }

    pub(in crate::agent_loop) fn mark_run_started(&self) {
        self.lock_control().has_run = true;
    }

    pub(in crate::agent_loop) fn ensure_idle(&self) -> Result<(), AgentError> {
        let control = self.lock_control();
        if control.active {
            return Err(AgentError::RunBusy);
        }
        Ok(())
    }
}
