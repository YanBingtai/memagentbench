use serde::Deserialize;

/// Limits and deadlines that govern one agent run.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RunPolicy {
    /// Maximum number of model/tool rounds in one run.
    pub max_steps: usize,
    /// Maximum number of tool calls accepted from one assistant message.
    pub max_tool_calls_per_step: usize,
    /// Maximum number of parallel-safe tools that may run at once.
    pub max_parallel_tool_calls: usize,
    /// Maximum number of retries for transient model request failures.
    pub max_model_retries: usize,
    /// Base delay for exponential model retry backoff.
    pub model_retry_backoff_ms: u64,
    /// Maximum duration allowed for one tool execution.
    pub tool_timeout_secs: u64,
    /// Optional wall-clock budget for model and tool execution in one run.
    /// `None` preserves an unlimited run deadline.
    pub run_timeout_secs: Option<u64>,
    /// Maximum number of historical messages sent to the model.
    pub max_context_messages: usize,
    /// Approximate serialized byte budget for historical messages sent to the model.
    pub max_context_bytes: usize,
}

impl Default for RunPolicy {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_tool_calls_per_step: 16,
            max_parallel_tool_calls: 4,
            max_model_retries: 2,
            model_retry_backoff_ms: 250,
            tool_timeout_secs: 120,
            run_timeout_secs: None,
            max_context_messages: 128,
            max_context_bytes: 1_048_576,
        }
    }
}
