use super::super::{error::*, limits::*, options::*, sse::api_error_message};
use reqwest::StatusCode;
use serde_json::Value;

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
