//! Admission and execution of one sequential tool call.
//!
//! The facade keeps the sequential batch caller small while the implementation
//! is split by responsibility: request/result types, admission and outcome
//! mapping, and the timeout/cancellation runner.

mod admission;
mod context;
mod decision;
mod outcome;
mod types;

pub(super) use types::{ToolExecutionOutcome, ToolExecutionRequest};

use crate::{agent_loop::AgentLoop, model::ChatModel};

use super::super::AgentError;

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(super) async fn admit_and_execute_tool(
        &self,
        request: ToolExecutionRequest<'_>,
    ) -> Result<ToolExecutionOutcome, AgentError> {
        admission::admit_and_execute(self, request).await
    }
}
