use std::time::Duration;

use crate::model::ChatFuture;

use crate::agent_loop::helpers::timeout_error;

use super::{AgentError, CancellationToken};

pub(super) async fn wait_for_model(
    model_request: ChatFuture<'_>,
    deadline: Option<(tokio::time::Instant, Duration)>,
    cancellation: &CancellationToken,
) -> Result<crate::message::Message, AgentError> {
    match deadline {
        Some((deadline, timeout)) => tokio::select! {
            result = tokio::time::timeout_at(deadline, model_request) => {
                result
                    .map_err(|_| timeout_error(timeout))
                    .and_then(|result| result.map_err(AgentError::Model))
            }
            _ = cancellation.cancelled() => Err(AgentError::RunCancelled),
        },
        None => tokio::select! {
            result = model_request => result.map_err(AgentError::Model),
            _ = cancellation.cancelled() => Err(AgentError::RunCancelled),
        },
    }
}
