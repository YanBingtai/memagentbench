//! Typed errors that terminate an agent run.

use thiserror::Error;
use uuid::Uuid;

use crate::{model::ModelError, session::SessionError};

/// Errors that stop an agent run before it produces a final assistant message.
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("prompt must not be empty")]
    EmptyPrompt,

    #[error("cannot start another run while this agent is already active")]
    RunBusy,

    #[error("{kind} prompt must not be empty")]
    EmptyQueuedPrompt { kind: &'static str },

    #[error("{kind} queue is full (limit {limit})")]
    QueueFull { kind: &'static str, limit: usize },

    #[error("{kind} prompt exceeds the {limit}-byte limit")]
    QueuedPromptTooLarge { kind: &'static str, limit: usize },

    #[error("{kind} prompts can only be queued before the first run or during an active run")]
    ControlQueueRequiresActiveRun { kind: &'static str },

    #[error("max_steps must be greater than zero")]
    InvalidMaxSteps,

    #[error("run policy field {field} must be greater than zero")]
    InvalidRunPolicy { field: &'static str },

    #[error("agent reached its maximum of {max_steps} model steps")]
    MaxStepsExceeded { max_steps: usize },

    #[error("model response ended with finish_reason={reason}")]
    ModelTerminated { reason: String },

    #[error("agent run timed out after {timeout_ms} milliseconds")]
    RunTimedOut { timeout_ms: u64 },

    #[error("agent run was cancelled by the caller")]
    RunCancelled,

    #[error(
        "model context requires {required_bytes} serialized bytes, which exceeds the {limit_bytes}-byte limit"
    )]
    ContextTooLarge {
        required_bytes: usize,
        limit_bytes: usize,
    },

    #[error("context compaction failed: {0}")]
    ContextCompaction(String),

    #[error("model context has invalid message structure: {reason}")]
    InvalidContext { reason: String },

    #[error("runtime ledger failed: {0}")]
    Runtime(String),

    #[error(
        "supplied session identity {provided:?} does not match the attached session {expected}"
    )]
    SessionContextMismatch {
        expected: Uuid,
        provided: Option<Uuid>,
    },

    #[error("session/runtime mismatch for action {action_id}: {reason}")]
    RuntimeSessionMismatch { action_id: Uuid, reason: String },

    #[error(
        "session contains tool result {call_id:?} with outcome {status:?} but no runtime identity"
    )]
    UnidentifiedToolOutcome { call_id: String, status: String },

    #[error("runtime action {action_id} was not found among pending actions")]
    RuntimeActionNotFound { action_id: Uuid },

    #[error("runtime action {action_id} for tool {tool_name} has unknown outcome: {reason}")]
    RuntimeActionUnknown {
        action_id: Uuid,
        tool_name: String,
        reason: String,
    },

    #[error("tool execution was terminated by policy: {reason}")]
    ToolTerminated { reason: String },

    #[error("memory hook failed: {0}")]
    MemoryHook(String),

    #[error("after-run hook failed: {0}")]
    AfterRunHook(String),

    #[error("model requested {count} tool calls in step {step}; limit is {limit}")]
    TooManyToolCalls {
        step: usize,
        count: usize,
        limit: usize,
    },

    #[error("invalid tool call in step {step} at index {index}: {reason}")]
    InvalidToolCall {
        step: usize,
        index: usize,
        reason: String,
    },

    #[error(transparent)]
    Model(#[from] ModelError),

    #[error(transparent)]
    Session(#[from] SessionError),
}
