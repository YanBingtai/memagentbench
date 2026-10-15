//! Events emitted by the agent runtime.
//!
//! Events are an ephemeral protocol for a CLI, TUI, or service consumer.
//! Durable session storage keeps canonical messages separately; streaming
//! deltas should not be persisted as independent transcript messages.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::message::{Message, ToolCall};

/// Durable provider-neutral retrieval evidence carried by a memory decision.
///
/// Runtime ledger events remain the source of truth. The alias keeps the
/// event-facing API discoverable without making the core agent depend on a
/// concrete memory backend.
pub type MemoryTrace = pi_agent_runtime::MemoryRetrievalSummary;

/// Identifies one logical execution and lets consumers correlate its events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RunContext {
    /// The durable conversation, when this run is attached to one.
    pub session_id: Option<Uuid>,
    /// The process-level execution of one or more turns.
    pub run_id: Uuid,
    /// The user turn currently being processed.
    pub turn_id: Uuid,
}

impl RunContext {
    /// Create a fresh run and turn context for an optional session.
    pub fn new(session_id: Option<Uuid>) -> Self {
        Self {
            session_id,
            run_id: Uuid::new_v4(),
            turn_id: Uuid::new_v4(),
        }
    }
}

/// Terminal state of one agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    Completed,
    Failed,
    Cancelled,
    Unknown,
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::FunctionCall;

    #[test]
    fn serializes_tool_lifecycle_with_stable_event_names() {
        let context = RunContext::new(None);
        let event = AgentEvent::ToolStarted {
            context,
            step: 1,
            call: ToolCall {
                id: "call-1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: r#"{"path":"Cargo.toml"}"#.to_string(),
                },
            },
        };

        let json = serde_json::to_value(event).expect("event should serialize");

        assert_eq!(json["type"], "tool_started");
        assert_eq!(json["call"]["function"]["name"], "read_file");
        assert!(json["context"]["run_id"].is_string());
    }

    #[test]
    fn serializes_tool_updates_with_call_identity() {
        let event = AgentEvent::ToolUpdate {
            context: RunContext::new(None),
            step: 2,
            call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            content: "running".to_string(),
            details: serde_json::json!({"elapsed_ms": 10}),
        };

        let json = serde_json::to_value(event).expect("tool update should serialize");

        assert_eq!(json["type"], "tool_update");
        assert_eq!(json["call_id"], "call-1");
        assert_eq!(json["details"]["elapsed_ms"], 10);
    }

    #[test]
    fn serializes_tool_terminal_timing_and_status() {
        let action_id = Uuid::new_v4();
        let grant_id = Uuid::new_v4();
        let event = AgentEvent::ToolFinished {
            context: RunContext::new(None),
            step: 1,
            call_id: "call-1".to_string(),
            tool_name: "read_file".to_string(),
            content: "done".to_string(),
            is_error: false,
            details: Value::Null,
            terminate: false,
            duration_ms: 12,
            status: "succeeded".to_string(),
            action_id: Some(action_id),
            grant_id: Some(grant_id),
        };

        let json = serde_json::to_value(event).expect("tool terminal event should serialize");

        assert_eq!(json["type"], "tool_finished");
        assert_eq!(json["duration_ms"], 12);
        assert_eq!(json["status"], "succeeded");
        assert_eq!(json["action_id"], action_id.to_string());
        assert_eq!(json["grant_id"], grant_id.to_string());
    }

    #[test]
    fn serializes_reasoning_deltas_with_stable_event_names() {
        let event = AgentEvent::ReasoningDelta {
            context: RunContext::new(None),
            step: 1,
            content: "先分析".to_string(),
        };

        let json = serde_json::to_value(event).expect("reasoning delta should serialize");

        assert_eq!(json["type"], "reasoning_delta");
        assert_eq!(json["content"], "先分析");
    }

    #[test]
    fn serializes_memory_decision_correlation_fields() {
        let event = AgentEvent::MemoryDecision {
            context: RunContext::new(None),
            step: 2,
            proposal_id: Uuid::from_u128(7),
            event_sequence: 11,
            decision_kind: "injected".to_string(),
            injection_count: 1,
            total_token_cost: 5,
            skip_reason: None,
        };

        let json = serde_json::to_value(event).expect("memory decision should serialize");

        assert_eq!(json["type"], "memory_decision");
        assert_eq!(json["step"], 2);
        assert_eq!(json["event_sequence"], 11);
        assert_eq!(json["decision_kind"], "injected");
        assert_eq!(json["total_token_cost"], 5);
    }

    #[test]
    fn serializes_tool_call_deltas_with_partial_fields() {
        let event = AgentEvent::ToolCallDelta {
            context: RunContext::new(None),
            step: 1,
            index: 0,
            id: Some("call-1".to_string()),
            call_type: Some("function".to_string()),
            name: Some("read_file".to_string()),
            arguments: Some("{\"path\":\"Ca".to_string()),
        };

        let json = serde_json::to_value(event).expect("tool-call delta should serialize");

        assert_eq!(json["type"], "tool_call_delta");
        assert_eq!(json["index"], 0);
        assert_eq!(json["arguments"], "{\"path\":\"Ca");
    }

    #[test]
    fn serializes_assistant_stream_resets_with_attempt_identity() {
        let event = AgentEvent::AssistantStreamReset {
            context: RunContext::new(None),
            step: 2,
            attempt: 1,
            reason: "stream ended before completion".to_string(),
        };

        let json = serde_json::to_value(event).expect("stream reset should serialize");

        assert_eq!(json["type"], "assistant_stream_reset");
        assert_eq!(json["step"], 2);
        assert_eq!(json["attempt"], 1);
        assert_eq!(json["reason"], "stream ended before completion");
    }

    #[test]
    fn serializes_turn_boundaries_with_tool_results() {
        let event = AgentEvent::TurnFinished {
            context: RunContext::new(None),
            step: 1,
            message: Message::assistant(Some("done".to_string()), None),
            tool_results: vec![Message::tool("call-1", "ok")],
        };

        let json = serde_json::to_value(event).expect("turn event should serialize");

        assert_eq!(json["type"], "turn_finished");
        assert_eq!(json["step"], 1);
        assert_eq!(json["tool_results"][0]["tool_call_id"], "call-1");
    }

    #[test]
    fn preserves_terminal_outcome_and_error() {
        let event = AgentEvent::RunFinished {
            context: RunContext::new(Some(Uuid::new_v4())),
            steps: 2,
            outcome: RunOutcome::Failed,
            final_message: None,
            error: Some("model unavailable".to_string()),
        };

        let restored: AgentEvent =
            serde_json::from_value(serde_json::to_value(event).expect("event should serialize"))
                .expect("event should deserialize");

        assert!(matches!(
            restored,
            AgentEvent::RunFinished {
                outcome: RunOutcome::Failed,
                error: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn serializes_unknown_terminal_outcome() {
        let event = AgentEvent::RunFinished {
            context: RunContext::new(None),
            steps: 1,
            outcome: RunOutcome::Unknown,
            final_message: None,
            error: Some("external outcome is unknown".to_string()),
        };

        let json = serde_json::to_value(event).expect("unknown outcome should serialize");

        assert_eq!(json["outcome"], "unknown");
    }
}
