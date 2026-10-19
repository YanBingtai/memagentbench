use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::message::{Message, ToolCall};

use super::{RunContext, RunOutcome};

/// Runtime events consumed by presentation and service layers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Emitted before the first model request.
    RunStarted { context: RunContext },

    /// Emitted before one assistant response and its tool-result batch.
    TurnStarted { context: RunContext, step: usize },

    /// Emitted immediately before one model request.
    ModelStarted {
        context: RunContext,
        step: usize,
        model: String,
    },

    /// A transient model request failed and will be retried.
    ModelRetry {
        context: RunContext,
        step: usize,
        attempt: usize,
        delay_ms: u64,
        error: String,
    },

    /// The runtime evaluated one memory proposal for the current model step.
    ///
    /// This is an ephemeral correlation event. The durable runtime ledger is
    /// authoritative for the decision itself; this event exposes its sequence
    /// so adapters can link a model step to the corresponding ledger record.
    MemoryDecision {
        context: RunContext,
        step: usize,
        proposal_id: Uuid,
        event_sequence: u64,
        decision_kind: String,
        injection_count: usize,
        total_token_cost: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        skip_reason: Option<String>,
    },

    /// An ephemeral text fragment from a streaming model response.
    AssistantDelta {
        context: RunContext,
        step: usize,
        content: String,
    },

    /// An ephemeral reasoning fragment from a thinking-capable model.
    ReasoningDelta {
        context: RunContext,
        step: usize,
        content: String,
    },

    /// An ephemeral fragment of a streamed tool call.
    ///
    /// `name` and `arguments` may be partial fragments. The complete,
    /// validated tool call is emitted later as [`Self::AssistantMessage`].
    ToolCallDelta {
        context: RunContext,
        step: usize,
        index: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        call_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        arguments: Option<String>,
    },

    /// Clear partial assistant and reasoning output from one failed model
    /// attempt before a retry or terminal error.
    AssistantStreamReset {
        context: RunContext,
        step: usize,
        attempt: usize,
        reason: String,
    },

    /// The complete assistant message after a model response is assembled.
    AssistantMessage {
        context: RunContext,
        step: usize,
        message: Message,
    },

    /// Emitted after one assistant response and all of its tool results.
    TurnFinished {
        context: RunContext,
        step: usize,
        message: Message,
        tool_results: Vec<Message>,
    },

    /// Emitted when the agent starts handling one model tool call.
    ///
    /// The call may still be rejected by argument validation, a hook, or
    /// runtime admission; consumers should use [`Self::ToolFinished`]'s
    /// status and `executed` details to determine whether execution occurred.
    ToolStarted {
        context: RunContext,
        step: usize,
        call: ToolCall,
    },

    /// An incremental update emitted while a tool is still running.
    ToolUpdate {
        context: RunContext,
        step: usize,
        call_id: String,
        tool_name: String,
        content: String,
        details: Value,
    },

    /// Emitted after a tool succeeds or fails.
    ToolFinished {
        context: RunContext,
        step: usize,
        call_id: String,
        tool_name: String,
        content: String,
        is_error: bool,
        details: Value,
        terminate: bool,
        /// Elapsed wall-clock time from `ToolStarted` to this terminal event.
        /// Synthetic recovery results use zero because no live execution was
        /// observed in the current process.
        #[serde(default)]
        duration_ms: u64,
        /// Stable terminal category for consumers that should not parse text.
        #[serde(default)]
        status: String,
        /// Runtime action identity when admission succeeded for this call.
        #[serde(skip_serializing_if = "Option::is_none")]
        action_id: Option<Uuid>,
        /// Runtime grant identity associated with `action_id`.
        #[serde(skip_serializing_if = "Option::is_none")]
        grant_id: Option<Uuid>,
    },

    /// The single terminal event for a run.
    RunFinished {
        context: RunContext,
        steps: usize,
        outcome: RunOutcome,
        final_message: Option<Message>,
        error: Option<String>,
    },
}
