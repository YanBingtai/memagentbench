use serde::Deserialize;

use crate::model::limits::MAX_ERROR_BODY_BYTES;

/// Extract a provider error message and bound an otherwise unstructured body.
pub(crate) fn api_error_message(body: &str) -> String {
    #[derive(Deserialize)]
    struct ErrorEnvelope {
        error: Option<ApiErrorBody>,
    }

    #[derive(Deserialize)]
    struct ApiErrorBody {
        message: Option<String>,
    }

    let parsed_message = serde_json::from_str::<ErrorEnvelope>(body)
        .ok()
        .and_then(|envelope| envelope.error)
        .and_then(|error| error.message);

    parsed_message.unwrap_or_else(|| truncate_for_error(body))
}

pub(crate) fn truncate_for_error(body: &str) -> String {
    let mut end = body.len().min(MAX_ERROR_BODY_BYTES);
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }

    let truncated = &body[..end];
    if end < body.len() {
        format!("{truncated}…")
    } else {
        truncated.to_string()
    }
}
