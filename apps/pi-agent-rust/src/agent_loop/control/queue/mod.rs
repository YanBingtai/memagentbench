//! Steering and follow-up prompt queues for an agent run.

mod operations;
mod state;

const MAX_QUEUED_PROMPTS: usize = 32;
const MAX_QUEUED_PROMPT_BYTES: usize = 64 * 1024;

pub(in crate::agent_loop) use state::ControlState;
use state::RunGuard;
