//! Request validation and runtime preparation for one agent run.

use std::time::Duration;

use pi_agent_runtime::{EventLedger, RunId, RuntimeCoordinator};

use crate::{events::RunContext, model::ChatModel, session::SessionStore};

use super::super::{AgentError, AgentLoop, CancellationToken};

pub(super) type AgentRuntime = RuntimeCoordinator<Box<dyn EventLedger + Send>>;

pub(super) struct RunRequest {
    pub(super) prompt: String,
    pub(super) timeout: Option<Duration>,
    pub(super) cancellation: CancellationToken,
    pub(super) supplied_context: Option<RunContext>,
}

pub(super) struct PreparedRun {
    pub(super) prompt: String,
    pub(super) context: RunContext,
    pub(super) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(super) cancellation: CancellationToken,
    pub(super) runtime: Result<AgentRuntime, AgentError>,
}

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(super) fn prepare_run(&self, request: RunRequest) -> PreparedRun {
        let RunRequest {
            prompt,
            timeout,
            cancellation,
            supplied_context,
        } = request;
        let externally_supplied_context = supplied_context.is_some();
        let mut context = supplied_context.unwrap_or_else(|| {
            RunContext::new(self.session.as_ref().map(SessionStore::session_id))
        });
        let deadline = timeout.map(|timeout| (tokio::time::Instant::now() + timeout, timeout));

        let runtime = match self.preflight_error(&prompt, &cancellation, context) {
            Some(error) => Err(error),
            None => {
                self.mark_run_started();
                self.prepare_runtime(
                    RunId::from_uuid(context.run_id),
                    !externally_supplied_context,
                )
            }
        };
        if !externally_supplied_context {
            if let Ok(runtime) = &runtime {
                context.run_id = runtime.run_id().as_uuid();
            }
        }

        PreparedRun {
            prompt,
            context,
            deadline,
            cancellation,
            runtime,
        }
    }

    fn preflight_error(
        &self,
        prompt: &str,
        cancellation: &CancellationToken,
        context: RunContext,
    ) -> Option<AgentError> {
        if prompt.trim().is_empty() {
            return Some(AgentError::EmptyPrompt);
        }
        if cancellation.is_cancelled() {
            return Some(AgentError::RunCancelled);
        }
        self.session_mismatch(context)
    }

    fn session_mismatch(&self, context: RunContext) -> Option<AgentError> {
        self.session.as_ref().and_then(|session| {
            (context.session_id != Some(session.session_id())).then_some(
                AgentError::SessionContextMismatch {
                    expected: session.session_id(),
                    provided: context.session_id,
                },
            )
        })
    }
}
