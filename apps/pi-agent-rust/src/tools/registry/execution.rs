//! Prepared tool execution and result normalization.

use crate::{
    protocol::ToolCall,
    tools::{contracts::ToolExecutionContext, errors::ToolError},
};

use super::{PreparedToolCall, ToolRegistry};
use super::super::contracts::ToolResult;

impl ToolRegistry {
    pub async fn execute(&self, call: &ToolCall) -> Result<String, ToolError> {
        self.execute_result(call).await.map(|result| result.content)
    }

    pub async fn execute_result(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let prepared = self.prepare_call(call)?;
        self.execute_prepared_result(prepared).await
    }

    pub async fn execute_result_with_context(
        &self,
        call: &ToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        let prepared = self.prepare_call(call)?;
        self.execute_prepared_result_with_context(prepared, context)
            .await
    }

    pub async fn execute_prepared_result(
        &self,
        prepared: PreparedToolCall,
    ) -> Result<ToolResult, ToolError> {
        prepared
            .tool
            .execute_result(prepared.arguments)
            .await
            .map(|result| self.normalize_result(result))
    }

    pub async fn execute_prepared_result_with_context(
        &self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        let _update_lease = context.lease_updates();
        let result = prepared
            .tool
            .execute_result_with_context(prepared.arguments, context)
            .await?;
        Ok(self.normalize_result(result))
    }

    /// Apply the registry-wide output bound after a tool or hook changes a result.
    pub fn normalize_result(&self, result: ToolResult) -> ToolResult {
        result.bounded(self.max_tool_output_bytes)
    }

    /// Return the registry-wide output limit used by tool results and updates.
    pub fn output_limit(&self) -> usize {
        self.max_tool_output_bytes
    }
}
