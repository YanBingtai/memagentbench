use super::super::{
    api::*, client::*, error::*, limits::*, options::*, sse::validate_non_streaming_message,
};
use crate::protocol::{FunctionCall, Message, ToolCall};
use serde_json::json;
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
