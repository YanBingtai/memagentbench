use std::time::Duration;

use crate::tools::{ToolError, ToolResult};

use super::request::ToolExecutionRequest;

/// Terminal state returned by the shared execution runner.
#[derive(Debug)]
pub(crate) enum ToolExecutionOutcome {
    Completed(Result<ToolResult, ToolError>),
    ToolTimedOut,
    RunTimedOut(Duration),
    Cancelled,
}

/// Execute a prepared call with the tool timeout, run deadline, and caller
/// cancellation applied in one place.
pub(crate) async fn execute(request: ToolExecutionRequest) -> ToolExecutionOutcome {
    let ToolExecutionRequest {
        tools,
        prepared,
        context,
        tool_timeout,
        deadline,
        cancellation,
    } = request;

    match deadline {
        Some((deadline, run_timeout)) => tokio::select! {
            result = tokio::time::timeout_at(
                deadline,
                tokio::time::timeout(
                    tool_timeout,
                    tools.execute_prepared_result_with_context(prepared, context),
                ),
            ) => match result {
                Ok(Ok(result)) => ToolExecutionOutcome::Completed(result),
                Ok(Err(_)) => ToolExecutionOutcome::ToolTimedOut,
                Err(_) => ToolExecutionOutcome::RunTimedOut(run_timeout),
            },
            _ = cancellation.cancelled() => ToolExecutionOutcome::Cancelled,
        },
        None => tokio::select! {
            result = tokio::time::timeout(
                tool_timeout,
                tools.execute_prepared_result_with_context(prepared, context),
            ) => match result {
                Ok(result) => ToolExecutionOutcome::Completed(result),
                Err(_) => ToolExecutionOutcome::ToolTimedOut,
            },
            _ = cancellation.cancelled() => ToolExecutionOutcome::Cancelled,
        },
    }
}
