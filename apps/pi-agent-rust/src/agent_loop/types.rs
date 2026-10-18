//! Public contracts shared by the agent loop implementation.

use std::{future::Future, pin::Pin, sync::Arc};

use pi_agent_runtime::{
    MemoryHookContext, MemoryProposal, MemoryRevision, PolicyRevision, RiskClass,
};
use serde_json::Value;
use uuid::Uuid;

use crate::{
    events::{RunContext, RunOutcome},
    message::{Message, ToolCall},
    model::ModelUsage,
    tools::ToolReplayPolicy,
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

pub(super) type BeforeToolHook =
    Arc<dyn Fn(ToolHookContext) -> ToolHookFuture<BeforeToolDecision> + Send + Sync>;
pub(super) type AfterToolHook =
    Arc<dyn Fn(ToolResultContext) -> ToolHookFuture<ToolResultOverride> + Send + Sync>;

/// The completed transcript and final answer from one agent run.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub messages: Vec<Message>,
    pub final_message: Message,
    pub steps: usize,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RunStats {
    pub(super) steps: usize,
    pub(super) model_retries: usize,
    pub(super) has_unknown_action: bool,
    model_requests: usize,
    input_usage_complete: bool,
    output_usage_complete: bool,
    total_usage_complete: bool,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

impl Default for RunStats {
    fn default() -> Self {
        Self {
            steps: 0,
            model_retries: 0,
            has_unknown_action: false,
            model_requests: 0,
            input_usage_complete: true,
            output_usage_complete: true,
            total_usage_complete: true,
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
        }
    }
}

impl RunStats {
    pub(crate) fn record_model_usage(&mut self, usage: Option<ModelUsage>) {
        self.model_requests = self.model_requests.saturating_add(1);
        let Some(usage) = usage else {
            self.input_usage_complete = false;
            self.output_usage_complete = false;
            self.total_usage_complete = false;
            return;
        };

        record_token(
            &mut self.input_tokens,
            &mut self.input_usage_complete,
            usage.input_tokens,
        );
        record_token(
            &mut self.output_tokens,
            &mut self.output_usage_complete,
            usage.output_tokens,
        );
        record_token(
            &mut self.total_tokens,
            &mut self.total_usage_complete,
            usage.total_tokens,
        );
    }

    pub(crate) fn record_retry(&mut self) {
        self.model_retries = self.model_retries.saturating_add(1);
    }

    pub(crate) fn model_usage(&self) -> (Option<u64>, Option<u64>, Option<u64>) {
        if self.model_requests == 0 {
            return (None, None, None);
        }
        (
            self.input_tokens.filter(|_| self.input_usage_complete),
            self.output_tokens.filter(|_| self.output_usage_complete),
            self.total_tokens.filter(|_| self.total_usage_complete),
        )
    }
}

fn record_token(slot: &mut Option<u64>, complete: &mut bool, value: Option<u64>) {
    let Some(value) = value else {
        *complete = false;
        return;
    };
    *slot = Some(slot.unwrap_or_default().saturating_add(value));
}

/// A runtime action that stopped with an unknown external outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingRuntimeAction {
    pub action_id: Uuid,
    pub grant_id: Uuid,
    pub run_id: Uuid,
    pub tool_name: String,
    pub idempotency_key: String,
    pub argument_digest: String,
    pub risk: RiskClass,
    pub issued_at_ms: u64,
    pub expires_at_ms: Option<u64>,
    /// Whether the registered tool permits an automatic replay policy.
    ///
    /// This is advisory metadata for a reconciler. The runtime still owns the
    /// final decision and unknown actions remain blocked until reconciliation.
    pub replay_policy: ToolReplayPolicy,
}

/// Evidence supplied by an external reconciler for an unknown action.
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeReconcileOutcome {
    Succeeded { output: Value, evidence: String },
    Failed { error: String, evidence: String },
}

/// Input supplied to a memory hook before one model request.
///
/// The hook receives an owned, already-truncated context so it can perform
/// asynchronous retrieval without borrowing the agent loop across an await.
#[derive(Debug, Clone)]
pub struct MemoryHookRequest {
    pub context: MemoryHookContext,
    pub messages: Vec<Message>,
}

pub type MemoryHookFuture =
    Pin<Box<dyn Future<Output = Result<Option<MemoryProposal>, String>> + Send>>;

pub(super) type MemoryProposalHook =
    Arc<dyn Fn(MemoryHookRequest) -> MemoryHookFuture + Send + Sync>;

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

pub(super) type AfterRunHook = Arc<dyn Fn(AfterRunRequest) -> AfterRunFuture + Send + Sync>;

/// Why the agent asked a caller-provided compactor to transform context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextCompactionReason {
    ByteBudget,
}

/// Input supplied to the optional context compaction hook.
///
/// The returned messages are request-only context. They are not written to
/// the session automatically; the durable transcript remains the source of
/// truth for recovery and auditing.
#[derive(Debug, Clone)]
pub struct ContextCompactionRequest {
    pub context: RunContext,
    pub step: usize,
    pub reason: ContextCompactionReason,
    pub messages: Vec<Message>,
    pub max_context_messages: usize,
    pub max_context_bytes: usize,
}

pub type ContextCompactionFuture =
    Pin<Box<dyn Future<Output = Result<Vec<Message>, String>> + Send>>;

pub(super) type ContextCompactionHook =
    Arc<dyn Fn(ContextCompactionRequest) -> ContextCompactionFuture + Send + Sync>;

/// Static limits and revisions used to build each runtime memory context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryHookConfig {
    pub namespace: String,
    pub memory_revision: MemoryRevision,
    pub policy_revision: PolicyRevision,
    pub max_tokens: u32,
    pub max_candidates: u32,
    pub max_projection_bytes: u32,
}

impl Default for MemoryHookConfig {
    fn default() -> Self {
        Self {
            namespace: "default".to_owned(),
            memory_revision: MemoryRevision::INITIAL,
            policy_revision: PolicyRevision::INITIAL,
            max_tokens: 1_024,
            max_candidates: 16,
            max_projection_bytes: 32 * 1024,
        }
    }
}
