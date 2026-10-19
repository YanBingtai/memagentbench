//! Transcript, event, and runtime compensation for parallel results.

mod remaining;
mod worker_error;

pub(super) use remaining::persist_remaining_tool_results;
pub(super) use worker_error::persist_worker_error_results;
