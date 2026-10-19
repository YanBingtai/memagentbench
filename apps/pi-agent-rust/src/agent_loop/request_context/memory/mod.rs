//! Memory proposal evaluation and ephemeral context injection.
//!
//! The module keeps the turn-facing method small by separating hook execution,
//! preview validation, and output formatting from the runtime commit path.

mod format;
mod hook;

use std::time::Duration;

use pi_agent_runtime::{EventLedger, RunId, RuntimeCoordinator};

use crate::{events::RunContext, message::Message, model::ChatModel};

use super::super::{unix_timestamp_ms, AgentError, AgentLoop, CancellationToken};

#[cfg(test)]
pub(in crate::agent_loop) use format::format_memory_context;

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    #[allow(clippy::too_many_arguments)]
    pub(in crate::agent_loop) async fn prepare_memory_context(
        &self,
        run: RunContext,
        step: usize,
        runtime_run_id: RunId,
        messages: &mut Vec<Message>,
        runtime: &mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
    ) -> Result<(), AgentError> {
        let Some((config, hook)) = &self.memory_hook else {
            return Ok(());
        };

        let request_now_ms = unix_timestamp_ms();
        let hook_context =
            hook::build_context(config, runtime_run_id, run, runtime, request_now_ms);
        let proposal =
            hook::run_hook(hook, hook_context.clone(), messages, deadline, cancellation).await?;
        let Some(proposal) = proposal else {
            return Ok(());
        };

        let evaluation_now_ms = unix_timestamp_ms();
        let mut evaluation_context = hook_context;
        evaluation_context.now_ms = evaluation_now_ms;
        hook::validate_preview(
            &proposal,
            &evaluation_context,
            messages,
            self.policy.max_context_messages,
            self.policy.max_context_bytes,
            self.context_compaction_hook.is_none(),
        )?;

        let receipt = runtime
            .evaluate_memory_proposal_with_receipt(
                &proposal,
                &evaluation_context,
                evaluation_now_ms,
            )
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
        self.emit(format::decision_event(
            run,
            step,
            receipt.event_sequence.value(),
            &receipt.decision,
        ));
        format::insert_memory_context(messages, &receipt.decision);
        Ok(())
    }
}
