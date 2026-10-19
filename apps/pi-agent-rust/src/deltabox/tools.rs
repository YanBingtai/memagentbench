use std::sync::Arc;

use serde_json::Value;

use crate::{
    CancellationToken, Tool, ToolConcurrency, ToolContext, ToolDefinition, ToolError,
    ToolExecutionContext, ToolFuture, ToolReplayPolicy, ToolResultFuture,
};

use super::DeltaBoxSandbox;

pub(super) struct DeltaBoxTool {
    pub(super) native: Arc<dyn Tool>,
    pub(super) sandbox: DeltaBoxSandbox,
    pub(super) context: ToolContext,
}

impl Tool for DeltaBoxTool {
    fn definition(&self) -> ToolDefinition {
        let mut definition = self.native.definition();
        if definition.function.name == "bash" {
            definition.function.description = "Execute bash in the DeltaBox workspace. Use relative paths. Network is denied; changes follow the session review/commit policy.".into();
        }
        definition
    }
    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        self.native.validate_arguments(arguments)
    }
    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }
    fn replay_policy(&self) -> ToolReplayPolicy {
        self.native.replay_policy()
    }
    fn risk_class(&self) -> pi_agent_runtime::RiskClass {
        self.native.risk_class()
    }
    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move { Ok(self.execute_result(arguments).await?.content) })
    }
    fn execute_result<'a>(&'a self, arguments: Value) -> ToolResultFuture<'a> {
        self.execute_result_with_context(
            arguments,
            ToolExecutionContext::new(CancellationToken::new()),
        )
    }
    fn execute_result_with_context<'a>(
        &'a self,
        arguments: Value,
        context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(self.sandbox.execute(
            self.native.definition().function.name,
            arguments,
            self.context.clone(),
            context,
        ))
    }
}
