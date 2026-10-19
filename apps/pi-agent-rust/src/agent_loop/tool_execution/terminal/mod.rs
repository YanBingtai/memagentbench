//! Terminal mapping for one prepared tool execution.
//!
//! The execution schedulers share this contract but keep their own persistence
//! policy. Sequential execution may append synthetic cancellation results,
//! while parallel execution commits each worker result independently.

mod mapping;
mod unknown;

pub(crate) use mapping::terminalize;

use crate::{agent_loop::AgentError, tools::ToolResult};

/// The result and stop policy shared by all execution schedulers.
pub(crate) struct TerminalOutcome {
    pub(crate) tool_result: ToolResult,
    pub(crate) unknown_reason: Option<String>,
    pub(crate) stop_error: Option<AgentError>,
    pub(crate) should_commit: bool,
}
