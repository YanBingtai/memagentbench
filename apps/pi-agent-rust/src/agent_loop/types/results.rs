//! Runtime and transcript results returned by the agent loop.

use pi_agent_runtime::RiskClass;
use serde_json::Value;
use uuid::Uuid;

use crate::{message::Message, tools::ToolReplayPolicy};

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
