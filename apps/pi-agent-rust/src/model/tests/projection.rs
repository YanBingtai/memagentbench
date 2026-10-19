use super::super::options::*;
use crate::protocol::{Message, ToolResultMetadata};
use serde_json::json;

#[test]
fn provider_projection_omits_internal_tool_result_metadata() {
    let messages = [Message::tool_with_result(
        "call-1",
        "hello",
        ToolResultMetadata::new(json!({"source": "test"}), false, false),
    )];
    let projected = provider_messages(&messages, false);
    let request = ChatCompletionRequest {
        model: "demo",
        messages: &projected,
        tools: None,
        tool_choice: None,
        temperature: 0.0,
        stream: None,
        chat_template_kwargs: None,
    };

    let json = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(json["messages"][0]["content"], "hello");
    assert_eq!(json["messages"][0]["tool_call_id"], "call-1");
    assert!(json["messages"][0].get("tool_result").is_none());
}

#[test]
fn provider_projection_omits_internal_reasoning_content() {
    let mut message = Message::assistant(Some("answer".to_string()), None);
    message.reasoning_content = Some("private reasoning".to_string());
    let messages = [message];
    let projected = provider_messages(&messages, false);
    let request = ChatCompletionRequest {
        model: "demo",
        messages: &projected,
        tools: None,
        tool_choice: None,
        temperature: 0.0,
        stream: None,
        chat_template_kwargs: None,
    };

    let json = serde_json::to_value(request).expect("request should serialize");

    assert!(json["messages"][0].get("reasoning_content").is_none());
}

#[test]
fn provider_projection_keeps_reasoning_content_when_thinking_is_enabled() {
    let mut message = Message::assistant(Some("answer".to_string()), None);
    message.reasoning_content = Some("private reasoning".to_string());
    let messages = [message];
    let projected = provider_messages(&messages, true);
    let request = ChatCompletionRequest {
        model: "qwen3.8",
        messages: &projected,
        tools: None,
        tool_choice: None,
        temperature: 0.0,
        stream: None,
        chat_template_kwargs: Some(ChatTemplateKwargs {
            enable_thinking: true,
        }),
    };

    let json = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(
        json["messages"][0]["reasoning_content"],
        "private reasoning"
    );
    assert!(json["messages"][0].get("tool_result").is_none());
}

#[test]
fn mimo_default_thinking_keeps_prior_reasoning_content() {
    let mut message = Message::assistant(Some("answer".to_string()), None);
    message.reasoning_content = Some("prior reasoning".to_string());
    let messages = [message];
    let projected = provider_messages(
        &messages,
        thinking_enabled_for_request("mimo-v2.6-flash", None),
    );

    assert_eq!(
        projected[0].reasoning_content.map(String::as_str),
        Some("prior reasoning")
    );
}
