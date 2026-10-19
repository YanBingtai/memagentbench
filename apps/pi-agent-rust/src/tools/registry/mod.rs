//! Registration, validation, and execution facade for tools.

mod builtins;
mod catalog;
mod execution;
mod gateway;
mod traits;
mod types;
mod validation;

pub use traits::Tool;
pub(super) use super::contracts::{ToolCapabilities, ToolConcurrency, ToolExposure, ToolReplayPolicy};
pub(super) use super::errors::{RegistryError, ToolError};
pub(super) use types::RegisteredTool;
pub use types::{PreparedToolCall, ToolGateway, ToolRegistry};
