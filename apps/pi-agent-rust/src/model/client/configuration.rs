use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use reqwest::header;

use super::super::{
    api::ModelUsage, error::ModelError, limits::CHAT_COMPLETIONS_PATH, options::ToolChoice,
};

/// A small OpenAI-compatible chat-completions client.
///
/// The client owns transport configuration, while the agent owns conversation
/// state and decides when a request should be made. Keeping these concerns
/// separate lets us add other providers without changing the agent loop.
#[derive(Clone, Debug)]
pub struct ModelClient {
    pub(crate) http: reqwest::Client,
    pub(crate) endpoint: String,
    pub(crate) tool_choice: ToolChoice,
    pub(crate) last_usage: Arc<Mutex<Option<ModelUsage>>>,
}

impl ModelClient {
    /// Build a client from a provider base URL such as `http://localhost:8000/v1`.
    pub fn new(base_url: &str, timeout: Duration) -> Result<Self, ModelError> {
        let base_url = base_url.trim_end_matches('/');
        if base_url.is_empty() || !base_url.contains("://") {
            return Err(ModelError::InvalidEndpoint(base_url.to_string()));
        }
        if timeout.is_zero() {
            return Err(ModelError::InvalidTimeout);
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
            tool_choice: ToolChoice::Auto,
            last_usage: Arc::new(Mutex::new(None)),
        })
    }

    /// Select how the provider should handle advertised tools.
    ///
    /// `Auto` preserves agent behavior. `None` is useful for providers that
    /// accept tool definitions but do not enable automatic tool selection.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }
}
