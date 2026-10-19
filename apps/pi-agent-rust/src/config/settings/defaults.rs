use super::Config;

impl Default for Config {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            model: "your-model".to_string(),
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            workspace: "./workspace".into(),
            session_file: "./sessions/session.jsonl".into(),
            max_steps: 8,
            max_tool_calls_per_step: 16,
            max_parallel_tool_calls: 4,
            max_model_retries: 2,
            model_retry_backoff_ms: 250,
            tool_timeout_secs: 120,
            run_timeout_secs: None,
            max_context_messages: 128,
            max_context_bytes: 1_048_576,
            temperature: 0.0,
            system_prompt: None,
            enable_thinking: Some(true),
            request_timeout_secs: 120,
            max_file_bytes: 1_048_576,
            max_tool_output_bytes: 65_536,
        }
    }
}
