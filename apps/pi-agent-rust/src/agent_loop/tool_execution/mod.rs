//! Shared execution primitives for one prepared tool call.
//!
//! Scheduling belongs to the sequential and parallel coordinators.  This
//! module owns the mechanics that must stay identical in both paths:
//! constructing progress context and applying tool/run timeout policy while
//! observing the run cancellation token.

mod context;
mod request;
mod terminal;
mod timeout;

pub(crate) use context::build_context;
pub(crate) use request::ToolExecutionRequest;
pub(crate) use terminal::terminalize;
pub(crate) use timeout::{execute, ToolExecutionOutcome};
