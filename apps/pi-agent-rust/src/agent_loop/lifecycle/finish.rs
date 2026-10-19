//! After-run persistence hooks and terminal lifecycle events.

use crate::events::{AgentEvent, RunContext, RunOutcome};

use super::super::{
    outcomes::run_outcome_for_error_with_runtime, AfterRunRequest, AgentError, AgentLoop,
    RunResult, RunStats,
};

impl<M> AgentLoop<M> {
    pub(super) async fn finish_run(
        &self,
        context: RunContext,
        transcript: Vec<crate::message::Message>,
        stats: RunStats,
        mut result: Result<RunResult, AgentError>,
    ) -> Result<RunResult, AgentError> {
        let after_run_hook_error = self
            .run_after_hook(context, transcript, stats, &mut result)
            .await;
        self.emit_run_finished(context, stats, &result, after_run_hook_error);
        result
    }

    async fn run_after_hook(
        &self,
        context: RunContext,
        transcript: Vec<crate::message::Message>,
        stats: RunStats,
        result: &mut Result<RunResult, AgentError>,
    ) -> Option<String> {
        let Some(hook) = &self.after_run_hook else {
            return None;
        };
        let outcome = match &*result {
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
        match hook(request).await {
            Ok(()) => None,
            Err(error) if result.is_ok() => {
                *result = Err(AgentError::AfterRunHook(error));
                None
            }
            Err(error) => Some(error),
        }
    }

    fn emit_run_finished(
        &self,
        context: RunContext,
        stats: RunStats,
        result: &Result<RunResult, AgentError>,
        after_run_hook_error: Option<String>,
    ) {
        let event = match result {
            Ok(result) => AgentEvent::RunFinished {
                context,
                steps: result.steps,
                outcome: RunOutcome::Completed,
                final_message: Some(result.final_message.clone()),
                error: None,
            },
            Err(error) => AgentEvent::RunFinished {
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
            },
        };
        self.emit(event);
    }
}
