use crate::config::RunPolicy;

use super::Config;

impl Config {
    /// Return the runtime policy represented by this configuration.
    pub fn run_policy(&self) -> RunPolicy {
        RunPolicy {
            max_steps: self.max_steps,
            max_tool_calls_per_step: self.max_tool_calls_per_step,
            max_parallel_tool_calls: self.max_parallel_tool_calls,
            max_model_retries: self.max_model_retries,
            model_retry_backoff_ms: self.model_retry_backoff_ms,
            tool_timeout_secs: self.tool_timeout_secs,
            run_timeout_secs: self.run_timeout_secs,
            max_context_messages: self.max_context_messages,
            max_context_bytes: self.max_context_bytes,
        }
    }
}
