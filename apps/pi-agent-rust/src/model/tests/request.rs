use super::super::{error::ModelError, options::*};
use crate::protocol::ToolDefinition;
use serde_json::{json, Value};

#[test]
fn request_omits_empty_tools() {
    let request = ChatCompletionRequest {
        model: "demo",
        messages: &[],
        tools: None,
        tool_choice: None,
        temperature: 0.0,
        stream: None,
        chat_template_kwargs: None,
    };
    let json = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(json["model"], Value::String("demo".to_string()));
    assert!(json.get("tools").is_none());
    assert!(json.get("tool_choice").is_none());
    assert!(json.get("stream").is_none());
}

#[test]
fn streaming_request_sets_stream_flag() {
    let request = ChatCompletionRequest {
        model: "demo",
        messages: &[],
        tools: None,
        tool_choice: None,
        temperature: 0.0,
        stream: Some(true),
        chat_template_kwargs: None,
    };
    let json = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(json["stream"], true);
}

#[test]
fn request_sets_auto_tool_choice_when_tools_are_present() {
    let tools = [ToolDefinition::function(
        "read_file",
        "Read a workspace file",
        serde_json::json!({"type": "object"}),
    )];
    let request = ChatCompletionRequest {
        model: "demo",
        messages: &[],
        tools: Some(&tools),
        tool_choice: Some(ToolChoice::Auto),
        temperature: 0.0,
        stream: None,
        chat_template_kwargs: None,
    };
    let json = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(json["tool_choice"], Value::String("auto".to_string()));
}

#[test]
fn request_can_disable_tool_choice_for_provider_compatibility() {
    let tools = [ToolDefinition::function(
        "read_file",
        "Read a workspace file",
        serde_json::json!({"type": "object"}),
    )];
    let request = ChatCompletionRequest {
        model: "demo",
        messages: &[],
        tools: Some(&tools),
        tool_choice: Some(ToolChoice::None),
        temperature: 0.0,
        stream: None,
        chat_template_kwargs: None,
    };
    let json = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(json["tool_choice"], Value::String("none".to_string()));
}

#[test]
fn request_can_require_a_tool_call() {
    let tools = [ToolDefinition::function(
        "read_file",
        "Read a workspace file",
        serde_json::json!({"type": "object"}),
    )];
    let request = ChatCompletionRequest {
        model: "demo",
        messages: &[],
        tools: Some(&tools),
        tool_choice: Some(ToolChoice::Required),
        temperature: 0.0,
        stream: None,
        chat_template_kwargs: None,
    };
    let json = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(json["tool_choice"], Value::String("required".to_string()));
}

#[test]
fn request_can_select_one_function() {
    let tools = [ToolDefinition::function(
        "read_file",
        "Read a workspace file",
        serde_json::json!({"type": "object"}),
    )];
    let request = ChatCompletionRequest {
        model: "demo",
        messages: &[],
        tools: Some(&tools),
        tool_choice: Some(ToolChoice::Function {
            name: "read_file".to_string(),
        }),
        temperature: 0.0,
        stream: None,
        chat_template_kwargs: None,
    };
    let json = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(json["tool_choice"]["type"], "function");
    assert_eq!(json["tool_choice"]["function"]["name"], "read_file");
}

#[test]
fn request_can_enable_thinking_for_local_models() {
    let request = ChatCompletionRequest {
        model: "qwen3.8",
        messages: &[],
        tools: None,
        tool_choice: None,
        temperature: 0.7,
        stream: None,
        chat_template_kwargs: Some(ChatTemplateKwargs {
            enable_thinking: true,
        }),
    };
    let json = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(json["chat_template_kwargs"]["enable_thinking"], true);
}

#[test]
fn mimo_request_uses_thinking_contract() {
    let request = ChatCompletionRequest {
        model: "mimo-v2.6-flash",
        messages: &[],
        tools: None,
        tool_choice: None,
        temperature: 0.7,
        stream: None,
        chat_template_kwargs: Some(ChatTemplateKwargs {
            enable_thinking: true,
        }),
    };

    let body = provider_request_body("mimo-v2.6-flash", &request, Some(true))
        .expect("MiMo request should serialize");

    assert_eq!(body["thinking"]["type"], "enabled");
    assert!(body.get("chat_template_kwargs").is_none());
    assert!(body.get("temperature").is_none());
}

#[test]
fn mimo_request_can_disable_thinking() {
    let request = ChatCompletionRequest {
        model: "mimo-v2.6-flash",
        messages: &[],
        tools: None,
        tool_choice: None,
        temperature: 0.2,
        stream: None,
        chat_template_kwargs: Some(ChatTemplateKwargs {
            enable_thinking: false,
        }),
    };

    let body = provider_request_body("mimo-v2.6-flash", &request, Some(false))
        .expect("MiMo request should serialize");

    assert_eq!(body["thinking"]["type"], "disabled");
    assert!(body.get("chat_template_kwargs").is_none());
    assert!(
        (body["temperature"]
            .as_f64()
            .expect("temperature must be numeric")
            - 0.2)
            .abs()
            < 1e-6
    );
}

#[test]
fn mimo_rejects_non_auto_tool_choice() {
    let request = ChatCompletionRequest {
        model: "mimo-v2.6-flash",
        messages: &[],
        tools: None,
        tool_choice: Some(ToolChoice::Function {
            name: "read_file".to_owned(),
        }),
        temperature: 0.2,
        stream: None,
        chat_template_kwargs: None,
    };

    let error = provider_request_body("mimo-v2.6-flash", &request, None)
        .expect_err("MiMo must reject unsupported tool selection modes");

    assert!(matches!(
        error,
        ModelError::UnsupportedToolChoice { model } if model == "mimo-v2.6-flash"
    ));
}

#[test]
fn mimo_keeps_auto_tool_choice_and_tools() {
    let tools = [ToolDefinition::function(
        "read_file",
        "Read a workspace file",
        json!({"type": "object"}),
    )];
    let request = ChatCompletionRequest {
        model: "mimo-v2.6-flash",
        messages: &[],
        tools: Some(&tools),
        tool_choice: Some(ToolChoice::Auto),
        temperature: 0.2,
        stream: None,
        chat_template_kwargs: None,
    };

    let body = provider_request_body("mimo-v2.6-flash", &request, None)
        .expect("MiMo should accept automatic tool selection");

    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["tools"][0]["function"]["name"], "read_file");
}

#[test]
fn mimo_default_request_explicitly_enables_thinking() {
    let request = ChatCompletionRequest {
        model: "mimo-v2.6-flash",
        messages: &[],
        tools: None,
        tool_choice: None,
        temperature: 0.2,
        stream: None,
        chat_template_kwargs: None,
    };

    let body = provider_request_body("mimo-v2.6-flash", &request, None)
        .expect("MiMo default request should serialize");

    assert_eq!(body["thinking"]["type"], "enabled");
    assert!(body.get("temperature").is_none());
    assert!(body.get("chat_template_kwargs").is_none());
}
