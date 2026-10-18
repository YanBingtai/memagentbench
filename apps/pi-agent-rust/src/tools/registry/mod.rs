//! Registration, validation, and execution facade for tools.

mod execution;
mod gateway;
mod registration;
mod validation;

use std::{collections::BTreeMap, sync::Arc};

use pi_agent_runtime::RiskClass;
use serde_json::Value;

use crate::{
    protocol::{ToolCall, ToolDefinition},
    skills::SkillCatalog,
};

use super::{
    contracts::*,
    errors::{RegistryError, ToolError},
};

/// A dynamically dispatchable asynchronous tool.
pub trait Tool: Send + Sync {
    /// Return a stable, side-effect-free definition for the lifetime of the
    /// registered tool. The registry validates this definition at registration
    /// and providers receive the same logical schema on every request.
    fn definition(&self) -> ToolDefinition;
    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a>;

    /// Validate arguments before hooks, runtime admission, or execution.
    ///
    /// Built-in tools override this with typed deserialization and business
    /// rules. The registry applies the registered definition snapshot for the
    /// generic schema check, so custom tools do not need to re-read a dynamic
    /// definition here.
    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let _ = arguments;
        Ok(())
    }

    /// Execute with structured output while preserving the original string
    /// tool API for existing implementations.
    fn execute_result<'a>(&'a self, arguments: Value) -> ToolResultFuture<'a> {
        Box::pin(async move { self.execute(arguments).await.map(ToolResult::text) })
    }

    /// Execute with cooperative cancellation and progress reporting.
    ///
    /// Existing tools keep their original implementation automatically. A
    /// long-running tool can override this method and periodically inspect
    /// `context.is_cancelled()` or call `context.update(...)`.
    fn execute_result_with_context<'a>(
        &'a self,
        arguments: Value,
        _context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(async move { self.execute_result(arguments).await })
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    /// Declare the recovery policy for an interrupted invocation.
    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Never
    }

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Direct
    }

    /// Declares the side-effect class used by runtime admission.
    ///
    /// Custom tools are denied until they state their behavior explicitly.
    /// This keeps a newly registered tool from accidentally receiving the
    /// permissions intended for a read-only tool.
    fn risk_class(&self) -> RiskClass {
        RiskClass::Unknown
    }
}

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
pub(super) struct RegisteredTool {
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
