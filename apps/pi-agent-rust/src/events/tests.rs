use super::*;
use crate::message::{FunctionCall, Message, ToolCall};
use serde_json::Value;
use uuid::Uuid;

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
