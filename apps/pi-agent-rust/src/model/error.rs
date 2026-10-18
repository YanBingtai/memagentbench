//! Errors and retry classification for model requests.

use reqwest::StatusCode;
use thiserror::Error;

/// Errors returned while constructing a model client or calling a provider.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("invalid model endpoint: {0}")]
    InvalidEndpoint(String),

    #[error("model request timeout must be greater than zero")]
    InvalidTimeout,

    #[error("temperature must be finite and between 0.0 and 2.0")]
    InvalidTemperature,

    #[error("model {model} only supports tool_choice=auto")]
    UnsupportedToolChoice { model: String },

    #[error("failed to build HTTP client: {0}")]
    BuildClient(#[source] reqwest::Error),

    #[error("model request failed: {0}")]
    Http(#[source] reqwest::Error),

    #[error("model returned HTTP {status}: {message}")]
    Api { status: StatusCode, message: String },

    #[error("model returned no choices")]
    EmptyResponse,

    #[error("failed to decode model response: {0}")]
    Decode(#[source] serde_json::Error),

    #[error("failed to encode model request: {0}")]
    Encode(#[source] serde_json::Error),

    #[error("model response is not valid UTF-8")]
    ResponseEncoding,

    #[error("model response exceeded the {limit}-byte limit")]
    ResponseLimit { limit: usize },

    #[error("model response contained more than {limit} tool calls")]
    ToolCallCountLimit { limit: usize },

    #[error("model stream ended before a completion signal")]
    StreamIncomplete,

    #[error("invalid model stream: {0}")]
    StreamProtocol(String),

    #[error("model stream exceeded the {limit} limit")]
    StreamLimit { limit: &'static str },
}

impl ModelError {
    /// Whether retrying the same read-only model request may succeed later.
    ///
    /// Model requests are safe to retry because the provider call does not
    /// execute an agent tool. Decode and protocol errors remain terminal so a
    /// malformed response is not hidden by repeated requests.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Http(error) => error.is_connect() || error.is_timeout(),
            Self::Api { status, .. } => {
                status.is_server_error()
                    || matches!(
                        *status,
                        StatusCode::REQUEST_TIMEOUT
                            | StatusCode::TOO_EARLY
                            | StatusCode::TOO_MANY_REQUESTS
                    )
            }
            Self::StreamIncomplete => true,
            Self::InvalidEndpoint(_)
            | Self::InvalidTimeout
            | Self::InvalidTemperature
            | Self::UnsupportedToolChoice { .. }
            | Self::BuildClient(_)
            | Self::EmptyResponse
            | Self::Decode(_)
            | Self::Encode(_)
            | Self::ResponseEncoding
            | Self::ResponseLimit { .. }
            | Self::ToolCallCountLimit { .. }
            | Self::StreamProtocol(_)
            | Self::StreamLimit { .. } => false,
        }
    }
}
