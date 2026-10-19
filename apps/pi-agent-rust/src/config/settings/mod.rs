//! Runtime configuration and its small, composable accessors.

use std::path::PathBuf;

use serde::Deserialize;

mod credentials;
mod defaults;
mod load;
mod runtime;

/// Configuration loaded from TOML or built from the production defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub base_url: String,
    pub model: String,
    pub api_key_env: Option<String>,
    pub workspace: PathBuf,
    pub session_file: PathBuf,
    pub max_steps: usize,
    pub max_tool_calls_per_step: usize,
    pub max_parallel_tool_calls: usize,
    pub max_model_retries: usize,
    pub model_retry_backoff_ms: u64,
    pub tool_timeout_secs: u64,
    pub run_timeout_secs: Option<u64>,
    pub max_context_messages: usize,
    pub max_context_bytes: usize,
    pub temperature: f32,
    /// Optional stable system prompt inserted before the session transcript.
    pub system_prompt: Option<String>,
    /// Control thinking for compatible local model servers.
    ///
    /// Omitted TOML settings use `Some(true)` from `Default`.
    /// `Some(false)` explicitly disables thinking; callers can set `None`
    /// programmatically to leave the choice to the model server.
    pub enable_thinking: Option<bool>,
    pub request_timeout_secs: u64,
    pub max_file_bytes: usize,
    pub max_tool_output_bytes: usize,
}
