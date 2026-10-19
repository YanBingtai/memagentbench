use std::{env, time::Duration};

mod arguments;
mod definition;
mod execution;
mod format;
mod provider;

use super::limits::*;
use reqwest::Client;

#[derive(Clone)]
pub struct WebSearchTool {
    client: Client,
    endpoint: String,
    api_key: String,
    timeout: Duration,
    max_output_bytes: usize,
}

impl std::fmt::Debug for WebSearchTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WebSearchTool")
            .field("endpoint", &self.endpoint)
            .field("timeout", &self.timeout)
            .field("max_output_bytes", &self.max_output_bytes)
            .finish_non_exhaustive()
    }
}

impl WebSearchTool {
    pub fn new(api_key: impl Into<String>, timeout: Duration, max_output_bytes: usize) -> Self {
        Self {
            client: Client::new(),
            endpoint: BRAVE_SEARCH_ENDPOINT.to_string(),
            api_key: api_key.into(),
            timeout: if timeout.is_zero() {
                Duration::from_secs(15)
            } else {
                timeout
            },
            max_output_bytes: if max_output_bytes == 0 {
                DEFAULT_TOOL_OUTPUT_BYTES
            } else {
                max_output_bytes
            },
        }
    }

    /// Build the default Brave-backed tool from the process environment.
    pub fn from_env(max_output_bytes: usize) -> Option<Self> {
        let api_key = env::var("BRAVE_SEARCH_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())?;
        let timeout = env::var("BRAVE_SEARCH_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|seconds| *seconds > 0)
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(15));
        Some(Self::new(api_key, timeout, max_output_bytes))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}
