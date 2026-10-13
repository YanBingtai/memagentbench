//! Events emitted by the agent runtime.
//!
//! Events are an ephemeral protocol for a CLI, TUI, or service consumer.
//! Durable session storage keeps canonical messages separately; streaming
//! deltas should not be persisted as independent transcript messages.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::message::{Message, ToolCall};

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
}

/// Runtime events consumed by presentation and service layers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Emitted before the first model request.
    RunStarted { context: RunContext },

    /// Emitted immediately before one model request.
    ModelStarted {
        context: RunContext,
        step: usize,
        model: String,
    },

    /// An ephemeral text fragment from a streaming model response.
    AssistantDelta {
        context: RunContext,
        step: usize,
        content: String,
    },

    /// The complete assistant message after a model response is assembled.
    AssistantMessage {
        context: RunContext,
        step: usize,
        message: Message,
    },

    /// Emitted before a tool is executed.
    ToolStarted {
        context: RunContext,
        step: usize,
        call: ToolCall,
    },

    /// Emitted after a tool succeeds or fails.
    ToolFinished {
        context: RunContext,
        step: usize,
        call_id: String,
        tool_name: String,
        content: String,
        is_error: bool,
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
        assert_eq!(json["context"]["run_id"].is_string(), true);
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
}
