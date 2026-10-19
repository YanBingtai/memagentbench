//! Orchestration for one prepared agent run.

use crate::events::AgentEvent;
use crate::model::ChatModel;

use super::super::{AgentError, AgentLoop, RunResult, RunStats};
use super::start::{PreparedRun, RunRequest};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(super) async fn run_request(&self, request: RunRequest) -> Result<RunResult, AgentError> {
        let _run_guard = self.start_run()?;
        let PreparedRun {
            prompt,
            context,
            deadline,
            cancellation,
            runtime,
        } = self.prepare_run(request);

        self.emit(AgentEvent::RunStarted { context });

        let mut stats = RunStats::default();
        let mut transcript = Vec::new();
        let result = match runtime {
            Ok(runtime) => {
                self.run_inner(
                    prompt,
                    context,
                    &mut stats,
                    &mut transcript,
                    deadline,
                    &cancellation,
                    runtime,
                )
                .await
            }
            Err(error) => Err(error),
        };

        self.finish_run(context, transcript, stats, result).await
    }
}
