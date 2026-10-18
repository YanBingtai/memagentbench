//! Public run entry points and the lifecycle around one agent execution.

use std::time::Duration;

use pi_agent_runtime::RunId;

use crate::{
    events::{AgentEvent, RunContext, RunOutcome},
    model::ChatModel,
    session::SessionStore,
};

use super::{
    outcomes::run_outcome_for_error_with_runtime, AfterRunRequest, AgentError, AgentLoop,
    CancellationToken, RunResult, RunStats,
};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    /// Run one prompt until the model returns an assistant message without tools.
    pub async fn run(&self, prompt: impl Into<String>) -> Result<RunResult, AgentError> {
        let cancellation = CancellationToken::new();
        self.run_with_deadline_and_context(prompt.into(), self.run_timeout, &cancellation, None)
            .await
    }

    /// Run one prompt using a caller-supplied execution identity.
    ///
    /// This is intended for orchestration and benchmark adapters that need to
    /// correlate model steps with an external operation. The supplied
    /// `run_id` and `turn_id` are preserved in lifecycle events and the fresh
    /// runtime coordinator. Runtime recovery is intentionally bypassed for
    /// this path: recovering an older ledger run would make the requested
    /// identity ambiguous. Configure reconciliation before starting a linked
    /// run when the ledger contains unresolved actions.
    pub async fn run_with_context(
        &self,
        prompt: impl Into<String>,
        context: RunContext,
    ) -> Result<RunResult, AgentError> {
        let cancellation = CancellationToken::new();
        self.run_with_deadline_and_context(
            prompt.into(),
            self.run_timeout,
            &cancellation,
            Some(context),
        )
        .await
    }

    /// Run one prompt with a deadline for model and tool execution.
    ///
    /// Session writes are deliberately outside the cancellable futures. If a
    /// tool is interrupted, synthetic tool results are persisted for all
    /// outstanding calls so a resumed transcript remains structurally valid.
    pub async fn run_with_timeout(
        &self,
        prompt: impl Into<String>,
        timeout: Duration,
    ) -> Result<RunResult, AgentError> {
        let cancellation = CancellationToken::new();
        self.run_with_deadline_and_context(prompt.into(), Some(timeout), &cancellation, None)
            .await
    }

    /// Run one prompt until completion, timeout, or an external cancellation.
    pub async fn run_with_cancellation(
        &self,
        prompt: impl Into<String>,
        cancellation: CancellationToken,
    ) -> Result<RunResult, AgentError> {
        self.run_with_deadline_and_context(prompt.into(), self.run_timeout, &cancellation, None)
            .await
    }

    async fn run_with_deadline_and_context(
        &self,
        prompt: String,
        timeout: Option<Duration>,
        cancellation: &CancellationToken,
        supplied_context: Option<RunContext>,
    ) -> Result<RunResult, AgentError> {
        let _run_guard = self.start_run()?;
        let deadline = timeout.map(|timeout| (tokio::time::Instant::now() + timeout, timeout));
        let externally_supplied_context = supplied_context.is_some();
        let mut context = supplied_context.unwrap_or_else(|| {
            RunContext::new(self.session.as_ref().map(SessionStore::session_id))
        });
        let session_mismatch = self.session.as_ref().and_then(|session| {
            (context.session_id != Some(session.session_id())).then_some(
                AgentError::SessionContextMismatch {
                    expected: session.session_id(),
                    provided: context.session_id,
                },
            )
        });
        let runtime = if prompt.trim().is_empty() {
            Err(AgentError::EmptyPrompt)
        } else if cancellation.is_cancelled() {
            Err(AgentError::RunCancelled)
        } else if let Some(error) = session_mismatch {
            Err(error)
        } else {
            self.mark_run_started();
            self.prepare_runtime(
                RunId::from_uuid(context.run_id),
                !externally_supplied_context,
            )
        };
        if !externally_supplied_context {
            if let Ok(runtime) = &runtime {
                context.run_id = runtime.run_id().as_uuid();
            }
        }
        self.emit(AgentEvent::RunStarted { context });

        let mut stats = RunStats::default();
        let mut transcript = Vec::new();
        let mut result = match runtime {
            Ok(runtime) => {
                self.run_inner(
                    prompt,
                    context,
                    &mut stats,
                    &mut transcript,
                    deadline,
                    cancellation,
                    runtime,
                )
                .await
            }
            Err(error) => Err(error),
        };
        let mut after_run_hook_error = None;
        if let Some(hook) = &self.after_run_hook {
            let outcome = match &result {
                Ok(_) => RunOutcome::Completed,
                Err(error) => run_outcome_for_error_with_runtime(error, stats.has_unknown_action),
            };
            let request = AfterRunRequest {
                context,
                transcript,
                steps: stats.steps,
                model_retries: stats.model_retries,
                outcome,
                result: result.as_ref().ok().cloned(),
                error: result.as_ref().err().map(ToString::to_string),
            };
            if let Err(error) = hook(request).await {
                if result.is_ok() {
                    result = Err(AgentError::AfterRunHook(error));
                } else {
                    after_run_hook_error = Some(error);
                }
            }
        }
        match &result {
            Ok(result) => self.emit(AgentEvent::RunFinished {
                context,
                steps: result.steps,
                outcome: RunOutcome::Completed,
                final_message: Some(result.final_message.clone()),
                error: None,
            }),
            Err(error) => self.emit(AgentEvent::RunFinished {
                context,
                steps: stats.steps,
                outcome: run_outcome_for_error_with_runtime(error, stats.has_unknown_action),
                final_message: None,
                error: Some(match after_run_hook_error {
                    Some(hook_error) => {
                        format!("{error}; after-run hook failed: {hook_error}")
                    }
                    None => error.to_string(),
                }),
            }),
        }
        result
    }
}
