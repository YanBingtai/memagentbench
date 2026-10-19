//! Hook contracts for tool and run lifecycle callbacks.

use std::{future::Future, pin::Pin, sync::Arc};

use serde_json::Value;

use super::results::RunResult;
use crate::{
    events::{RunContext, RunOutcome},
    message::{Message, ToolCall},
};

pub type ToolHookFuture<T> = Pin<Box<dyn Future<Output = Result<T, String>> + Send>>;

#[derive(Debug, Clone)]
pub struct ToolHookContext {
    pub run: RunContext,
    pub step: usize,
    pub call: ToolCall,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeforeToolDecision {
    Allow,
    Block {
        reason: String,
    },
    /// Stop the current tool batch and end the run before admission.
    Terminate {
        reason: String,
    },
}

#[derive(Debug, Clone)]
pub struct ToolResultContext {
    pub run: RunContext,
    pub step: usize,
    pub call: ToolCall,
    pub arguments: Value,
    pub content: String,
    pub is_error: bool,
    pub details: Value,
    pub terminate: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ToolResultOverride {
    pub content: Option<String>,
    pub is_error: Option<bool>,
    pub details: Option<Value>,
    pub terminate: Option<bool>,
}

pub(crate) type BeforeToolHook =
    Arc<dyn Fn(ToolHookContext) -> ToolHookFuture<BeforeToolDecision> + Send + Sync>;
pub(crate) type AfterToolHook =
    Arc<dyn Fn(ToolResultContext) -> ToolHookFuture<ToolResultOverride> + Send + Sync>;

/// Snapshot supplied to an optional hook after one agent run.
///
/// `transcript` is the canonical runtime transcript accumulated before any
/// ephemeral memory or skill context is added to a model request. It remains
/// available when the run fails after producing assistant or tool messages.
/// The core loop keeps this contract independent from any memory backend.
#[derive(Debug, Clone)]
pub struct AfterRunRequest {
    pub context: RunContext,
    pub transcript: Vec<Message>,
    pub steps: usize,
    pub model_retries: usize,
    pub outcome: RunOutcome,
    pub result: Option<RunResult>,
    pub error: Option<String>,
}

pub type AfterRunFuture = Pin<Box<dyn Future<Output = Result<(), String>> + Send>>;

pub(crate) type AfterRunHook = Arc<dyn Fn(AfterRunRequest) -> AfterRunFuture + Send + Sync>;
