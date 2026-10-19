//! Gateway preparation and execution methods.

use serde_json::Value;

use crate::message::ToolCall;
use crate::tools::contracts::{ToolExecutionContext, ToolResult};
use crate::tools::errors::ToolError;

use super::super::{PreparedToolCall, ToolGateway};

impl ToolGateway {
    /// Prepare a call produced by the model. Hidden tools are rejected here.
    pub fn prepare_model_call(
        &self,
        call: &ToolCall,
        arguments: Value,
    ) -> Result<PreparedToolCall, ToolError> {
        self.registry.prepare_model_call(call, arguments)
    }

    /// Prepare a trusted in-process call, including hidden capabilities.
    pub fn prepare_trusted_call(
        &self,
        call: &ToolCall,
        arguments: Value,
    ) -> Result<PreparedToolCall, ToolError> {
        self.registry.prepare_parsed_call(call, arguments)
    }

    pub async fn execute_prepared(
        &self,
        prepared: PreparedToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.registry.execute_prepared_result(prepared).await
    }

    pub async fn execute_prepared_result(
        &self,
        prepared: PreparedToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.execute_prepared(prepared).await
    }

    pub async fn execute_prepared_with_context(
        &self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        self.registry
            .execute_prepared_result_with_context(prepared, context)
            .await
    }

    pub async fn execute_prepared_result_with_context(
        &self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        self.execute_prepared_with_context(prepared, context).await
    }
}
