//! The dynamic tool contract used by the registry and built-in tools.

use pi_agent_runtime::RiskClass;
use serde_json::Value;

use crate::protocol::ToolDefinition;

use super::super::{
    contracts::{
        ToolConcurrency, ToolExecutionContext, ToolExposure, ToolFuture, ToolReplayPolicy,
        ToolResult, ToolResultFuture,
    },
    errors::ToolError,
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
