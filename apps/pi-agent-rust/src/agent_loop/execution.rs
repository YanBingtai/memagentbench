//! Main model/tool turn orchestration.

use std::time::Duration;

use pi_agent_runtime::{EventLedger, RuntimeCoordinator};

use crate::{events::RunContext, message::Message, model::ChatModel};

use super::{
    runtime::{apply_reconciled_tool_results, validate_session_runtime_consistency},
    turn::{TurnOutcome, TurnRequest},
    AgentError, AgentLoop, CancellationToken, RunResult, RunStats,
};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn run_inner(
        &self,
        prompt: String,
        context: RunContext,
        stats: &mut RunStats,
        transcript: &mut Vec<Message>,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
        mut runtime: RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    ) -> Result<RunResult, AgentError> {
        let definitions = self.tools.model_definitions();
        let runtime_run_id = runtime.run_id();
        let mut messages = match &self.session {
            Some(session) => session
                .load_repaired()
                .await?
                .into_iter()
                .map(|record| record.message)
                .collect(),
            None => Vec::new(),
        };
        if self.runtime_ledger_file.is_some() {
            validate_session_runtime_consistency(&messages, &runtime)?;
            apply_reconciled_tool_results(&mut messages, &runtime, self.tools.output_limit())?;
        }
        if let Some(system_prompt) = &self.system_prompt {
            messages.insert(0, Message::system(system_prompt.clone()));
        }
        if let Some(skills) = &self.skills {
            let skill_index = skills.format_for_system_prompt();
            if !skill_index.is_empty() {
                let system_end = messages
                    .iter()
                    .take_while(|message| message.role == "system")
                    .count();
                messages.insert(system_end, Message::system(skill_index));
            }
        }
        if cancellation.is_cancelled() {
            return Err(AgentError::RunCancelled);
        }
        let user_message = Message::user(prompt);
        messages.push(user_message.clone());
        transcript.push(user_message.clone());
        if let Some(session) = &self.session {
            session.append(&user_message).await?;
        }

        for step in 1..=self.policy.max_steps {
            match self
                .run_turn(TurnRequest {
                    context,
                    step,
                    definitions: &definitions,
                    messages: &mut messages,
                    transcript,
                    stats,
                    deadline,
                    cancellation,
                    runtime_run_id,
                    runtime: &mut runtime,
                })
                .await?
            {
                TurnOutcome::Continue => {}
                TurnOutcome::Completed(result) => return Ok(*result),
            }
        }

        Err(AgentError::MaxStepsExceeded {
            max_steps: self.policy.max_steps,
        })
    }
}
