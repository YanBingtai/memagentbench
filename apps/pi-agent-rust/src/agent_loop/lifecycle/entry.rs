//! Public entry points for one agent execution.

use std::time::Duration;

use crate::events::RunContext;
use crate::model::ChatModel;

use super::super::{AgentError, AgentLoop, CancellationToken, RunResult};
use super::start::RunRequest;

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    /// Run one prompt until the model returns an assistant message without tools.
    pub async fn run(&self, prompt: impl Into<String>) -> Result<RunResult, AgentError> {
        self.run_request(RunRequest {
            prompt: prompt.into(),
            timeout: self.run_timeout,
            cancellation: CancellationToken::new(),
            supplied_context: None,
        })
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
        self.run_request(RunRequest {
            prompt: prompt.into(),
            timeout: self.run_timeout,
            cancellation: CancellationToken::new(),
            supplied_context: Some(context),
        })
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
        self.run_request(RunRequest {
            prompt: prompt.into(),
            timeout: Some(timeout),
            cancellation: CancellationToken::new(),
            supplied_context: None,
        })
        .await
    }

    /// Run one prompt until completion, timeout, or an external cancellation.
    pub async fn run_with_cancellation(
        &self,
        prompt: impl Into<String>,
        cancellation: CancellationToken,
    ) -> Result<RunResult, AgentError> {
        self.run_request(RunRequest {
            prompt: prompt.into(),
            timeout: self.run_timeout,
            cancellation,
            supplied_context: None,
        })
        .await
    }
}
