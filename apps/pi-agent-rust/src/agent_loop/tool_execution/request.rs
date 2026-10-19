use std::time::Duration;

use crate::tools::{PreparedToolCall, ToolExecutionContext, ToolGateway};

use super::super::CancellationToken;

/// Inputs needed to execute one prepared call.
///
/// The request owns the gateway and cancellation token so it can be moved
/// into a parallel worker.  Sequential callers clone the inexpensive gateway
/// facade and use the same request type, keeping timeout semantics aligned.
pub(crate) struct ToolExecutionRequest {
    pub(crate) tools: ToolGateway,
    pub(crate) prepared: PreparedToolCall,
    pub(crate) context: ToolExecutionContext,
    pub(crate) tool_timeout: Duration,
    pub(crate) deadline: Option<(tokio::time::Instant, Duration)>,
    pub(crate) cancellation: CancellationToken,
}

impl ToolExecutionRequest {
    pub(crate) fn new(
        tools: ToolGateway,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
        tool_timeout: Duration,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            tools,
            prepared,
            context,
            tool_timeout,
            deadline,
            cancellation,
        }
    }
}
