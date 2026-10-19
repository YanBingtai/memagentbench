//! Per-call lifecycle and accumulated state for a sequential tool batch.
//!
//! The sequential coordinator owns only batch order. Call execution and
//! terminal-state accumulation live in focused siblings below.

mod call;
mod state;

pub(in crate::agent_loop::sequential) use call::{execute_tool, preflight_calls, ToolStepRequest};
pub(in crate::agent_loop::sequential) use state::BatchState;
