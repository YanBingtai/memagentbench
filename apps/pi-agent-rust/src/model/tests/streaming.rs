use super::super::{api::*, client::*, error::*, options::ChatOptions, sse::*};
use crate::protocol::Message;
use serde_json::json;
use std::time::Duration;
use wiremock::{matchers::method, Mock, MockServer, ResponseTemplate};

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
