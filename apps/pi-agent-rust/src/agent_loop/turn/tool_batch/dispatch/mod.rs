//! Parallel-first dispatch and terminal handling for an executable tool batch.

use crate::model::ChatModel;

use super::super::{
    super::{AgentError, AgentLoop},
    TurnOutcome,
};
use super::ToolBatchRequest;

mod parallel;
mod sequential;

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(super) async fn dispatch_tool_batch(
        &self,
        request: ToolBatchRequest<'_>,
    ) -> Result<TurnOutcome, AgentError> {
        let ToolBatchRequest {
            context,
            step,
            assistant_message,
            tool_calls,
            messages,
            transcript,
            stats,
            deadline,
            cancellation,
            runtime_run_id,
            runtime,
        } = request;

        let mut turn_tool_results = Vec::new();
        let parallel = self
            .try_run_parallel_tool_batch(
                context,
                step,
                &tool_calls,
                messages,
                transcript,
                &mut turn_tool_results,
                stats,
                deadline,
                cancellation,
                runtime_run_id,
                runtime,
            )
            .await?;

        let Some(outcome) = parallel else {
            return self
                .dispatch_sequential_batch(
                    context,
                    step,
                    tool_calls,
                    assistant_message,
                    messages,
                    transcript,
                    turn_tool_results,
                    stats,
                    deadline,
                    cancellation,
                    runtime_run_id,
                    runtime,
                )
                .await;
        };

        self.finish_parallel_batch(
            context,
            step,
            assistant_message,
            messages,
            stats,
            turn_tool_results,
            outcome,
        )
    }
}
