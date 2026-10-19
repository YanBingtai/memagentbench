use super::*;

#[test]
fn tool_call_serializes_with_openai_field_names() {
    let message = Message::assistant(
        None,
        Some(vec![ToolCall {
            id: "call-1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: r#"{"path":"hello.txt"}"#.to_string(),
            },
        }]),
    );
    let json = serde_json::to_value(message).unwrap();
    assert_eq!(json["role"], "assistant");
    assert_eq!(json["tool_calls"][0]["type"], "function");
    assert_eq!(json["tool_calls"][0]["function"]["name"], "read_file");
}

#[test]
fn tool_result_keeps_call_id() {
    let message = Message::tool("call-1", "hello");
    assert_eq!(message.role, "tool");
    assert_eq!(message.tool_call_id.as_deref(), Some("call-1"));
    assert!(message.tool_result.is_none());
}

#[test]
fn structured_tool_result_round_trips() {
    let message = Message::tool_with_result(
        "call-1",
        "hello",
        ToolResultMetadata::new(serde_json::json!({"source": "test"}), true, false),
    );

    let restored: Message =
        serde_json::from_value(serde_json::to_value(&message).expect("message should serialize"))
            .expect("message should deserialize");

    assert_eq!(restored, message);
    assert!(restored.tool_result.as_ref().unwrap().is_error);
}

#[test]
fn structured_tool_result_preserves_runtime_identity() {
    let action_id = uuid::Uuid::new_v4();
    let grant_id = uuid::Uuid::new_v4();
    let metadata = ToolResultMetadata::new(serde_json::json!({"source": "runtime"}), false, false)
        .with_runtime_identity(action_id, grant_id);
    let message = Message::tool_with_result("call-1", "hello", metadata);

    let restored: Message =
        serde_json::from_value(serde_json::to_value(&message).expect("message should serialize"))
            .expect("message should deserialize");

    let result = restored
        .tool_result
        .expect("structured result should be present");
    assert_eq!(result.action_id, Some(action_id));
    assert_eq!(result.grant_id, Some(grant_id));
}

#[test]
fn legacy_tool_message_deserializes_without_structured_metadata() {
    let message: Message = serde_json::from_value(serde_json::json!({
        "role": "tool",
        "content": "hello",
        "tool_call_id": "call-1"
    }))
    .expect("legacy tool message should remain readable");

    assert!(message.tool_result.is_none());
}

#[test]
fn reasoning_content_is_optional_for_legacy_messages() {
    let message: Message = serde_json::from_value(serde_json::json!({
        "role": "assistant",
        "content": "answer"
    }))
    .expect("legacy assistant message should remain readable");

    assert!(message.reasoning_content.is_none());
}
