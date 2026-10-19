use std::sync::{Arc, Mutex};

use uuid::Uuid;

use crate::{
    agent_loop::{AgentError, AgentLoop},
    events::{AgentEvent, RunOutcome},
    model::ModelClient,
    tools::ToolExecutionContext,
};

use super::super::progress::subagent_progress;

/// Ephemeral child lifecycle state captured from its event stream.
#[derive(Clone)]
pub(super) struct Observation {
    outcome: Arc<Mutex<Option<RunOutcome>>>,
    run_id: Arc<Mutex<Option<Uuid>>>,
}

impl Observation {
    fn new() -> Self {
        Self {
            outcome: Arc::new(Mutex::new(None)),
            run_id: Arc::new(Mutex::new(None)),
        }
    }

    pub(super) fn outcome(&self) -> Option<RunOutcome> {
        *self
            .outcome
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    pub(super) fn run_id(&self) -> Option<Uuid> {
        *self
            .run_id
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

pub(super) async fn execute(
    child: AgentLoop<ModelClient>,
    task: &str,
    context: Option<&ToolExecutionContext>,
) -> (
    Observation,
    Result<crate::agent_loop::RunResult, AgentError>,
) {
    let observation = Observation::new();
    let observed_outcome = Arc::clone(&observation.outcome);
    let observed_run_id = Arc::clone(&observation.run_id);
    let progress = context.cloned();
    let child = child.with_event_sink(move |event| {
        observe_lifecycle(&observed_run_id, &observed_outcome, &event);
        if let (Some(progress), Some(update)) = (progress.as_ref(), subagent_progress(&event)) {
            progress.update(update);
        }
    });

    let result = match context.map(ToolExecutionContext::cancellation_token) {
        Some(cancellation) => child.run_with_cancellation(task, cancellation).await,
        None => child.run(task).await,
    };
    (observation, result)
}

fn observe_lifecycle(
    run_id: &Arc<Mutex<Option<Uuid>>>,
    outcome: &Arc<Mutex<Option<RunOutcome>>>,
    event: &AgentEvent,
) {
    match event {
        AgentEvent::RunStarted { context, .. } => {
            *run_id.lock().unwrap_or_else(|error| error.into_inner()) = Some(context.run_id);
        }
        AgentEvent::RunFinished { outcome: value, .. } => {
            *outcome.lock().unwrap_or_else(|error| error.into_inner()) = Some(*value);
        }
        _ => {}
    }
}
