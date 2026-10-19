//! Execute one bounded delegated task through a fresh child [`AgentLoop`].

mod result;
mod run;
mod setup;

use serde_json::Value;

use crate::tools::{ToolError, ToolExecutionContext, ToolResult};

use super::SubagentTool;

impl SubagentTool {
    pub(super) async fn run_child(
        &self,
        arguments: Value,
        context: Option<ToolExecutionContext>,
    ) -> Result<ToolResult, ToolError> {
        let setup::PreparedChild { agent, spec } =
            setup::prepare(self, &arguments, context.as_ref())?;
        let (observation, outcome) = run::execute(agent, &spec.task, context.as_ref()).await;
        result::build(
            spec,
            context.as_ref(),
            observation,
            outcome,
            self.max_output_bytes,
        )
    }
}
