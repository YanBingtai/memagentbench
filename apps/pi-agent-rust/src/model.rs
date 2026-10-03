use std::time::Duration;

use reqwest::{header, StatusCode};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::message::{Message, ToolDefinition};

const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
const MAX_ERROR_BODY_BYTES: usize = 8 * 1024;

/// Errors returned while constructing a model client or calling a provider.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("invalid model endpoint: {0}")]
    InvalidEndpoint(String),

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
}

/// A small OpenAI-compatible chat-completions client.
///
/// The client owns transport configuration, while the agent owns conversation
/// state and decides when a request should be made. Keeping these concerns
/// separate lets us add other providers without changing the agent loop.
#[derive(Clone, Debug)]
pub struct ModelClient {
    http: reqwest::Client,
    endpoint: String,
}

impl ModelClient {
    /// Build a client from a provider base URL such as `http://localhost:8000/v1`.
    pub fn new(base_url: &str, timeout: Duration) -> Result<Self, ModelError> {
        let base_url = base_url.trim_end_matches('/');
        if base_url.is_empty() || !base_url.contains("://") {
            return Err(ModelError::InvalidEndpoint(base_url.to_string()));
        }

        let mut default_headers = header::HeaderMap::new();
        default_headers.insert(
            header::ACCEPT,
            header::HeaderValue::from_static("application/json"),
        );

        let http = reqwest::Client::builder()
            .default_headers(default_headers)
            .timeout(timeout)
            .build()
            .map_err(ModelError::BuildClient)?;

        Ok(Self {
            http,
            endpoint: format!("{base_url}{CHAT_COMPLETIONS_PATH}"),
        })
    }

    /// Send the current conversation and return the assistant message.
    pub async fn complete(
        &self,
        model: &str,
        api_key: Option<&str>,
        messages: &[Message],
        tools: &[ToolDefinition],
        temperature: f32,
    ) -> Result<Message, ModelError> {
        self.complete_with_options(
            model,
            api_key,
            messages,
            tools,
            ChatOptions {
                temperature,
                enable_thinking: None,
            },
        )
        .await
    }

    /// Send a chat request with provider-specific generation options.
    pub async fn complete_with_options(
        &self,
        model: &str,
        api_key: Option<&str>,
        messages: &[Message],
        tools: &[ToolDefinition],
        options: ChatOptions,
    ) -> Result<Message, ModelError> {
        let request = ChatCompletionRequest {
            model,
            messages,
            tools: (!tools.is_empty()).then_some(tools),
            temperature: options.temperature,
            chat_template_kwargs: options
                .enable_thinking
                .map(|enable_thinking| ChatTemplateKwargs { enable_thinking }),
        };

        let mut builder = self.http.post(&self.endpoint).json(&request);
        if let Some(api_key) = api_key.filter(|key| !key.is_empty()) {
            builder = builder.bearer_auth(api_key);
        }

        let response = builder.send().await.map_err(ModelError::Http)?;
        let status = response.status();
        let body = response.text().await.map_err(ModelError::Http)?;

        if !status.is_success() {
            return Err(ModelError::Api {
                status,
                message: api_error_message(&body),
            });
        }

        let parsed: ChatCompletionResponse =
            serde_json::from_str(&body).map_err(ModelError::Decode)?;
        parsed
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message)
            .ok_or(ModelError::EmptyResponse)
    }
}

/// Generation settings understood by OpenAI-compatible local model servers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChatOptions {
    pub temperature: f32,
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDefinition]>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
}

#[derive(Debug, Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

fn api_error_message(body: &str) -> String {
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

fn truncate_for_error(body: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

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
            temperature: 0.0,
            chat_template_kwargs: None,
        };
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["model"], Value::String("demo".to_string()));
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn request_can_enable_thinking_for_local_models() {
        let request = ChatCompletionRequest {
            model: "qwen3.8",
            messages: &[],
            tools: None,
            temperature: 0.7,
            chat_template_kwargs: Some(ChatTemplateKwargs {
                enable_thinking: true,
            }),
        };
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["chat_template_kwargs"]["enable_thinking"], true);
    }
}
