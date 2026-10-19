//! Lifecycle for one call in a sequential tool batch.
//!
//! The facade exposes the small API used by the batch coordinator. Each stage
//! lives in a focused sibling so request plumbing, preflight, execution, and
//! result finalization can be read independently.

mod execute;
mod finalize;
mod preflight;
mod request;

pub(in crate::agent_loop::sequential) use execute::execute_tool;
pub(in crate::agent_loop::sequential) use preflight::preflight_calls;
pub(in crate::agent_loop::sequential) use request::{ToolStepOutcome, ToolStepRequest};
