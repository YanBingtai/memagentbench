use std::sync::Arc;
use std::time::Duration;

use pi_agent_runtime::{DeltaBoxClient, DeltaBoxConfig, DeltaBoxError};
use serde_json::Value;

use crate::{
    EditFileTool, ListFilesTool, ReadFileTool, RegistryError, SearchFilesTool, ShellTool, Tool,
    ToolContext, ToolError, ToolExecutionContext, ToolRegistry, ToolResult, WriteFileTool,
};

use super::{session::DeltaBoxSession, tools::DeltaBoxTool, DeltaBoxDisposition};

mod command;
mod lifecycle;
mod policy;
mod probe;

#[derive(Clone, Debug)]
pub struct DeltaBoxSandbox {
    pub(super) inner: Arc<DeltaBoxSession>,
}

impl DeltaBoxSandbox {
    /// Create a confined session with network denied and a scrubbed child
    /// environment. The generated policy stays available for durable review.
    pub fn start(config: DeltaBoxConfig, timeout: Duration) -> Result<Self, ToolError> {
        lifecycle::start(config, timeout)
    }

    pub fn client(&self) -> &DeltaBoxClient {
        &self.inner.client
    }

    /// Register all six workspace tools against one consistent sandbox view.
    pub fn tools(&self, context: ToolContext) -> Result<ToolRegistry, RegistryError> {
        let mut tools = ToolRegistry::new().with_output_limit(context.max_tool_output_bytes);
        let native: Vec<Arc<dyn Tool>> = vec![
            Arc::new(ReadFileTool::new(context.clone())),
            Arc::new(ListFilesTool::new(context.clone())),
            Arc::new(SearchFilesTool::new(context.clone())),
            Arc::new(WriteFileTool::new(context.clone())),
            Arc::new(EditFileTool::new(context.clone())),
            Arc::new(ShellTool::new(context.clone())),
        ];
        for tool in native {
            tools.register(DeltaBoxTool {
                native: tool,
                sandbox: self.clone(),
                context: context.clone(),
            })?;
        }
        Ok(tools)
    }

    /// Finish only after tools have quiesced. A failed commit preserves the
    /// session and policy for inspection instead of discarding pending work.
    pub fn finish(&self, disposition: DeltaBoxDisposition) -> Result<Value, DeltaBoxError> {
        lifecycle::finish(self, disposition)
    }

    pub(super) async fn execute(
        &self,
        name: String,
        arguments: Value,
        limits: ToolContext,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        command::execute(self, name, arguments, limits, context).await
    }
}
