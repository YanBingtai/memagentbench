//! Execution context and progress contracts shared by tools.

use std::{future::Future, pin::Pin};

use super::super::errors::ToolError;

mod context;
mod updates;

pub use context::ToolExecutionContext;
// Keep the crate-visible paths stable for runtime code that may need to
// inspect the update gate while the implementation lives in `context`.
#[allow(unused_imports)]
pub(crate) use context::{UpdateGate, UpdateLease};
pub use updates::{ToolUpdate, ToolUpdateSink};

/// Future returned by a tool execution.
pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + 'a>>;
