//! Stable contracts shared by the registry and built-in tools.

mod capabilities;
mod execution;
mod result;

pub use capabilities::{
    ToolCapabilities, ToolConcurrency, ToolContext, ToolExposure, ToolReplayPolicy,
};
pub use execution::{ToolExecutionContext, ToolFuture, ToolUpdate, ToolUpdateSink};
pub use result::{ToolResult, ToolResultFuture};
