use std::time::Duration;

use super::super::{CancellationToken, ToolHookFuture};

pub(crate) enum ToolHookWait<T> {
    Completed(Result<T, String>),
    TimedOut(Duration),
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
