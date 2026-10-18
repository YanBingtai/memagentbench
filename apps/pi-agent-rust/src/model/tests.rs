use super::*;
use super::{api::*, client::*, error::*, limits::*, options::*, sse::*};
use crate::protocol::{FunctionCall, Message, ToolCall, ToolDefinition, ToolResultMetadata};
use reqwest::StatusCode;
use serde_json::json;
use serde_json::Value;
use std::time::Duration;
use wiremock::{matchers::method, Mock, MockServer, ResponseTemplate};

#[test]
fn client_appends_chat_completions_path() {
    let client = ModelClient::new("http://localhost:6273/v1/", Duration::from_secs(30))
        .expect("endpoint should be valid");

    assert_eq!(client.endpoint, "http://localhost:6273/v1/chat/completions");
}

#[test]
fn invalid_endpoint_is_rejected() {
    let error = ModelClient::new("localhost:6273", Duration::from_secs(30))
        .expect_err("endpoint without a scheme should fail");

    assert!(matches!(error, ModelError::InvalidEndpoint(_)));
}

#[test]
fn zero_request_timeout_is_rejected() {
    let error = ModelClient::new("http://localhost:6273/v1", Duration::ZERO)
        .expect_err("zero timeout should fail");

    assert!(matches!(error, ModelError::InvalidTimeout));
}

#[test]
fn invalid_temperature_is_rejected_before_request() {
    let error = validate_chat_options(ChatOptions {
        temperature: f32::NAN,
        enable_thinking: None,
    })
    .expect_err("NaN temperature should fail");
    assert!(matches!(error, ModelError::InvalidTemperature));

    let error = validate_chat_options(ChatOptions {
        temperature: 2.1,
        enable_thinking: None,
    })
    .expect_err("temperature above provider range should fail");
    assert!(matches!(error, ModelError::InvalidTemperature));
}

#[test]
fn retryability_is_limited_to_transient_provider_failures() {
    assert!(ModelError::Api {
        status: StatusCode::TOO_MANY_REQUESTS,
        message: "busy".to_string(),
    }
    .is_retryable());
    assert!(ModelError::Api {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: "overloaded".to_string(),
    }
    .is_retryable());
    assert!(ModelError::StreamIncomplete.is_retryable());
    assert!(!ModelError::Api {
        status: StatusCode::BAD_REQUEST,
        message: "invalid".to_string(),
    }
    .is_retryable());
    assert!(
        !ModelError::Decode(serde_json::from_str::<Value>("{").expect_err("invalid JSON"))
            .is_retryable()
    );
}

#[tokio::test]
async fn model_client_implements_provider_independent_trait() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "trait adapter works"
                }
            }],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client =
        ModelClient::new(&server.uri(), Duration::from_secs(5)).expect("endpoint should be valid");
    let messages = [Message::user("hello")];
    let response = ChatModel::complete(
        &client,
        ChatRequest {
            model: "test-model",
            api_key: None,
            messages: &messages,
            tools: &[],
            options: ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
        },
    )
    .await
    .expect("trait adapter should return the model response");

    assert_eq!(response.content.as_deref(), Some("trait adapter works"));
    assert_eq!(
        client.take_last_usage(),
        Some(ModelUsage {
            input_tokens: Some(7),
            output_tokens: Some(3),
            total_tokens: Some(10),
        })
    );
}

#[tokio::test]
async fn non_streaming_finish_reason_is_preserved() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "partial"
                },
                "finish_reason": "length"
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client =
        ModelClient::new(&server.uri(), Duration::from_secs(5)).expect("endpoint should be valid");
    let messages = [Message::user("hello")];
    let response = client
        .complete_with_options(
            "test-model",
            None,
            &messages,
            &[],
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
        )
        .await
        .expect("completion should decode");

    assert_eq!(response.stop_reason.as_deref(), Some("length"));
}

#[tokio::test]
async fn non_streaming_response_body_is_bounded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string("x".repeat(MAX_RESPONSE_BODY_BYTES + 1)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client =
        ModelClient::new(&server.uri(), Duration::from_secs(5)).expect("endpoint should be valid");
    let messages = [Message::user("hello")];
    let error = client
        .complete_with_options(
            "test-model",
            None,
            &messages,
            &[],
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
        )
        .await
        .expect_err("oversized response should be rejected");

    assert!(matches!(
        error,
        ModelError::ResponseLimit {
            limit: MAX_RESPONSE_BODY_BYTES
        }
    ));
}

#[test]
fn non_streaming_tool_arguments_are_bounded() {
    let message = Message::assistant(
        None,
        Some(vec![ToolCall {
            id: "call-1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: "x".repeat(MAX_STREAM_TOOL_BYTES + 1),
            },
        }]),
    );

    let error = validate_non_streaming_message(&message)
        .expect_err("oversized tool arguments should be rejected");

    assert!(matches!(
        error,
        ModelError::ResponseLimit {
            limit: MAX_STREAM_TOOL_BYTES
        }
    ));
}

#[tokio::test]
async fn streaming_response_assembles_text_and_split_tool_arguments() {
    let server = MockServer::start().await;
    let body = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"你\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"Ca\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"好\",\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"rgo.toml\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":5,\"total_tokens\":16}}\n\n",
            "data: [DONE]\n\n",
        );
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client =
        ModelClient::new(&server.uri(), Duration::from_secs(5)).expect("endpoint should be valid");
    let messages = [Message::user("stream")];
    let mut deltas = Vec::new();
    let response = ChatModel::stream(
        &client,
        ChatRequest {
            model: "test-model",
            api_key: None,
            messages: &messages,
            tools: &[],
            options: ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
        },
        &mut |delta| deltas.push(delta.to_string()),
    )
    .await
    .expect("stream should complete");

    assert_eq!(deltas, ["你", "好"]);
    assert_eq!(response.content.as_deref(), Some("你好"));
    let call = &response.tool_calls.as_ref().expect("tool call")[0];
    assert_eq!(call.id, "call-1");
    assert_eq!(call.function.name, "read_file");
    assert_eq!(call.function.arguments, r#"{"path":"Cargo.toml"}"#);
    assert_eq!(
        client.take_last_usage(),
        Some(ModelUsage {
            input_tokens: Some(11),
            output_tokens: Some(5),
            total_tokens: Some(16),
        })
    );
}

#[tokio::test]
async fn streaming_eof_without_done_is_rejected() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"stop\"}]}\n\n",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

    let client =
        ModelClient::new(&server.uri(), Duration::from_secs(5)).expect("endpoint should be valid");
    let messages = [Message::user("stream")];
    let error = client
        .complete_streaming(
            ChatRequest {
                model: "test-model",
                api_key: None,
                messages: &messages,
                tools: &[],
                options: ChatOptions {
                    temperature: 0.0,
                    enable_thinking: None,
                },
            },
            &mut |_| {},
        )
        .await
        .expect_err("missing completion marker should fail");

    assert!(matches!(error, ModelError::StreamIncomplete));
}

#[test]
fn sse_decoder_handles_crlf_and_split_utf8() {
    let mut decoder = SseDecoder::default();
    let bytes = "data: 你好\r\n\r\n".as_bytes();
    let split = "data: ".len() + 1;
    assert!(decoder.push(&bytes[..split]).unwrap().is_empty());
    let payloads = decoder.push(&bytes[split..]).unwrap();

    assert_eq!(payloads, ["你好"]);
    decoder.finish().expect("complete event should be flushed");
}

#[test]
fn repeated_tool_metadata_is_not_appended() {
    let mut aggregate = StreamAggregate::default();
    let first = serde_json::json!({
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\\\"path\\\": \\\""}
            }]}
        }]
    });
    let second = serde_json::json!({
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"arguments": "probe.txt\\\"}"}
            }]}
        }]
    });
    fn ignore_delta(_: &str) {}
    let mut on_delta = ignore_delta;
    aggregate
        .apply_json(&first.to_string(), &mut on_delta)
        .expect("first tool chunk should parse");
    aggregate
        .apply_json(&second.to_string(), &mut on_delta)
        .expect("repeated metadata should be ignored");

    let message = aggregate.finish().expect("tool call should be complete");
    let call = &message.tool_calls.expect("tool call should exist")[0];
    assert_eq!(call.id, "call-1");
    assert_eq!(call.call_type, "function");
    assert_eq!(call.function.name, "read_file");
    assert_eq!(call.function.arguments, r#"{\"path\": \"probe.txt\"}"#);
}

#[test]
fn streaming_tool_call_deltas_are_forwarded_before_aggregation() {
    let mut aggregate = StreamAggregate::default();
    let first = serde_json::json!({
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"Ca"}
            }]}
        }]
    });
    let second = serde_json::json!({
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": 0,
                "function": {"arguments": "rgo.toml\"}"}
            }]}
        }]
    });
    fn ignore_text(_: &str) {}
    let mut text = ignore_text;
    let mut reasoning = ignore_text;
    let mut deltas = Vec::new();
    let mut on_tool_call = |delta: &ToolCallDelta| deltas.push(delta.clone());

    aggregate
        .apply_json_with_reasoning_and_tools(
            &first.to_string(),
            &mut text,
            &mut reasoning,
            &mut on_tool_call,
        )
        .expect("first tool chunk should parse");
    aggregate
        .apply_json_with_reasoning_and_tools(
            &second.to_string(),
            &mut text,
            &mut reasoning,
            &mut on_tool_call,
        )
        .expect("second tool chunk should parse");

    assert_eq!(
        deltas,
        vec![
            ToolCallDelta {
                index: 0,
                id: Some("call-1".to_string()),
                call_type: Some("function".to_string()),
                name: Some("read_file".to_string()),
                arguments: Some("{\"path\":\"Ca".to_string()),
            },
            ToolCallDelta {
                index: 0,
                id: None,
                call_type: None,
                name: None,
                arguments: Some("rgo.toml\"}".to_string()),
            },
        ]
    );

    let message = aggregate.finish().expect("tool call should be complete");
    assert_eq!(
        message.tool_calls.expect("tool call should exist")[0]
            .function
            .arguments,
        r#"{"path":"Cargo.toml"}"#
    );
}

#[test]
fn streaming_reasoning_content_is_preserved_separately_from_answer() {
    let mut aggregate = StreamAggregate::default();
    let payload = serde_json::json!({
        "choices": [{
            "index": 0,
            "delta": {
                "reasoning_content": "先分析",
                "content": "答案"
            }
        }]
    });
    fn ignore_delta(_: &str) {}
    let mut on_delta = ignore_delta;

    aggregate
        .apply_json(&payload.to_string(), &mut on_delta)
        .expect("reasoning chunk should parse");
    let message = aggregate.finish().expect("aggregate should finish");

    assert_eq!(message.reasoning_content.as_deref(), Some("先分析"));
    assert_eq!(message.content.as_deref(), Some("答案"));
}

#[test]
fn qwen_inline_thinking_is_moved_out_of_content() {
    let message = Message::assistant(
        Some("<think>internal reasoning</think>\n\nfinal answer".to_string()),
        None,
    );
    let normalized = normalize_inline_thinking(message);

    assert_eq!(
        normalized.reasoning_content.as_deref(),
        Some("internal reasoning")
    );
    assert_eq!(normalized.content.as_deref(), Some("final answer"));
}

#[test]
fn qwen_inline_thinking_accepts_a_missing_opening_tag() {
    let mut message = Message::assistant(
        Some("internal reasoning\n</think>\n\nfinal answer".to_string()),
        None,
    );
    message.reasoning_content = Some("provider reasoning".to_string());
    let normalized = normalize_inline_thinking(message);

    assert_eq!(
        normalized.reasoning_content.as_deref(),
        Some("provider reasoning")
    );
    assert_eq!(normalized.content.as_deref(), Some("final answer"));
}

#[test]
fn streaming_qwen_inline_thinking_is_normalized_after_aggregation() {
    let mut aggregate = StreamAggregate::default();
    let payload = serde_json::json!({
        "choices": [{
            "index": 0,
            "delta": {"content": "internal</think>answer"}
        }]
    });
    fn ignore_delta(_: &str) {}
    let mut on_delta = ignore_delta;

    aggregate
        .apply_json(&payload.to_string(), &mut on_delta)
        .expect("inline thinking chunk should parse");
    let normalized = normalize_inline_thinking(
        aggregate
            .finish()
            .expect("aggregate should finish before normalization"),
    );

    assert_eq!(normalized.reasoning_content.as_deref(), Some("internal"));
    assert_eq!(normalized.content.as_deref(), Some("answer"));
}

#[test]
fn streaming_terminal_reasons_are_preserved_in_the_assistant_message() {
    for reason in ["length", "content_filter"] {
        let mut aggregate = StreamAggregate::default();
        let payload = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "partial",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\""
                        }
                    }]
                },
                "finish_reason": reason
            }]
        });
        fn ignore_delta(_: &str) {}
        let mut on_delta = ignore_delta;

        aggregate
            .apply_json(&payload.to_string(), &mut on_delta)
            .expect("terminal reason should still produce a message");
        let message = aggregate.finish().expect("partial message should finish");

        assert_eq!(message.stop_reason.as_deref(), Some(reason));
        assert_eq!(message.content.as_deref(), Some("partial"));
        assert!(message.tool_calls.is_some());
    }
}

#[test]
fn truncated_tool_calls_with_incomplete_metadata_become_terminal_messages() {
    let incomplete_calls = [
        json!({
            "index": 0,
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"}
        }),
        json!({
            "index": 0,
            "id": "call-1",
            "function": {"name": "read_file", "arguments": "{}"}
        }),
        json!({
            "index": 0,
            "id": "call-1",
            "type": "function",
            "function": {"arguments": "{}"}
        }),
        json!({
            "index": 0,
            "id": "call-1",
            "type": "custom",
            "function": {"name": "read_file", "arguments": "{}"}
        }),
    ];

    for tool_call in incomplete_calls {
        let mut aggregate = StreamAggregate::default();
        let payload = json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "partial",
                    "tool_calls": [tool_call]
                },
                "finish_reason": "length"
            }]
        });
        fn ignore_delta(_: &str) {}
        let mut on_delta = ignore_delta;

        aggregate
            .apply_json(&payload.to_string(), &mut on_delta)
            .expect("truncated tool metadata should parse");
        let message = aggregate
            .finish()
            .expect("truncated message should remain recoverable");

        assert_eq!(message.stop_reason.as_deref(), Some("length"));
        assert_eq!(message.content.as_deref(), Some("partial"));
        assert!(message.tool_calls.is_none());
    }
}

#[test]
fn api_error_prefers_provider_message() {
    let body = r#"{"error":{"message":"rate limit exceeded"}}"#;

    assert_eq!(api_error_message(body), "rate limit exceeded");
}

#[test]
fn unknown_api_error_body_is_truncated() {
    let body = "x".repeat(MAX_ERROR_BODY_BYTES + 1);

    assert_eq!(
        api_error_message(&body).len(),
        MAX_ERROR_BODY_BYTES + "…".len()
    );
}

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
