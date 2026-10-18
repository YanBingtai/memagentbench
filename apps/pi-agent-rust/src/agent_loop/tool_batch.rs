//! Small execution primitives shared by sequential and parallel tool paths.

use std::time::Duration;

use crate::tools::{PreparedToolCall, ToolExecutionContext, ToolGateway, ToolResult};

use super::{CancellationToken, ToolHookFuture};

pub(crate) enum ToolHookWait<T> {
    Completed(Result<T, String>),
    TimedOut(Duration),
    Cancelled,
}

#[derive(Debug)]
pub(crate) enum ParallelExecutionOutcome {
    Completed(Result<ToolResult, crate::tools::ToolError>),
    ToolTimedOut,
    RunTimedOut(Duration),
    Cancelled,
}

pub(crate) async fn await_tool_hook<T>(
    hook: ToolHookFuture<T>,
    deadline: Option<(tokio::time::Instant, Duration)>,
    cancellation: &CancellationToken,
) -> ToolHookWait<T> {
    match deadline {
        Some((deadline, run_timeout)) => tokio::select! {
            result = tokio::time::timeout_at(deadline, hook) => match result {
                Ok(_) if cancellation.is_cancelled() => ToolHookWait::Cancelled,
                Ok(_) if tokio::time::Instant::now() >= deadline => {
                    ToolHookWait::TimedOut(run_timeout)
                }
                Ok(result) => ToolHookWait::Completed(result),
                Err(_) => ToolHookWait::TimedOut(run_timeout),
            },
            _ = cancellation.cancelled() => ToolHookWait::Cancelled,
        },
        None => tokio::select! {
            result = hook => {
                if cancellation.is_cancelled() {
                    ToolHookWait::Cancelled
                } else {
                    ToolHookWait::Completed(result)
                }
            },
            _ = cancellation.cancelled() => ToolHookWait::Cancelled,
        },
    }
}

pub(crate) async fn execute_parallel_tool(
    tools: ToolGateway,
    prepared: PreparedToolCall,
    tool_context: ToolExecutionContext,
    tool_timeout: Duration,
    deadline: Option<(tokio::time::Instant, Duration)>,
    cancellation: CancellationToken,
) -> ParallelExecutionOutcome {
    match deadline {
        Some((deadline, run_timeout)) => {
            tokio::select! {
                result = tokio::time::timeout_at(
                    deadline,
                    tokio::time::timeout(
                        tool_timeout,
                        tools.execute_prepared_result_with_context(prepared, tool_context),
                    ),
                ) => match result {
                    Ok(Ok(result)) => ParallelExecutionOutcome::Completed(result),
                    Ok(Err(_)) => ParallelExecutionOutcome::ToolTimedOut,
                    Err(_) => ParallelExecutionOutcome::RunTimedOut(run_timeout),
                },
                _ = cancellation.cancelled() => ParallelExecutionOutcome::Cancelled,
            }
        }
        None => {
            tokio::select! {
                result = tokio::time::timeout(
                    tool_timeout,
                    tools.execute_prepared_result_with_context(prepared, tool_context),
                ) => match result {
                    Ok(result) => ParallelExecutionOutcome::Completed(result),
                    Err(_) => ParallelExecutionOutcome::ToolTimedOut,
                },
                _ = cancellation.cancelled() => ParallelExecutionOutcome::Cancelled,
            }
        }
    }
}
