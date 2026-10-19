mod contracts;
mod errors;
mod filesystem;
mod limits;
mod registry;
mod schema;
mod shell;
mod skill;
mod web_search;

pub use filesystem::{EditFileTool, ListFilesTool, ReadFileTool, SearchFilesTool, WriteFileTool};
pub use registry::{PreparedToolCall, Tool, ToolGateway, ToolRegistry};
#[cfg(test)]
use schema::*;
pub use shell::ShellTool;
pub use skill::ReadSkillTool;
pub use web_search::WebSearchTool;

pub use contracts::{
    ToolCapabilities, ToolConcurrency, ToolContext, ToolExecutionContext, ToolExposure, ToolFuture,
    ToolReplayPolicy, ToolResult, ToolResultFuture, ToolUpdate, ToolUpdateSink,
};
pub use errors::{RegistryError, ToolError};

#[cfg(test)]
mod tests;
