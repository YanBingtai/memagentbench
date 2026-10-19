//! Data types shared by registry, validation, and gateway implementations.

use std::{collections::BTreeMap, sync::Arc};

use serde_json::Value;

use crate::protocol::ToolDefinition;

use super::{super::contracts::ToolCapabilities, traits::Tool};

/// A tool call whose JSON arguments have already been parsed and validated.
pub struct PreparedToolCall {
    pub(super) tool: Arc<dyn Tool>,
    pub(super) arguments: Value,
}

impl PreparedToolCall {
    pub fn arguments(&self) -> &Value {
        &self.arguments
    }
}

#[derive(Clone)]
pub(crate) struct RegisteredTool {
    pub(super) tool: Arc<dyn Tool>,
    pub(super) definition: ToolDefinition,
    pub(super) capabilities: ToolCapabilities,
}

/// Registry used by the agent to advertise and execute tools.
#[derive(Clone)]
pub struct ToolRegistry {
    pub(super) tools: BTreeMap<String, RegisteredTool>,
    pub(super) max_tool_output_bytes: usize,
}

/// Execution facade built from a [`ToolRegistry`].
///
/// Keeping this facade separate gives callers a clear boundary: the registry
/// advertises stable model definitions, while the gateway prepares and runs
/// already-validated calls. Runtime admission and sandbox execution remain
/// outside this module.
#[derive(Clone)]
pub struct ToolGateway {
    pub(super) registry: ToolRegistry,
}
