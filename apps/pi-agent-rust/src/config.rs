use std::{
    env, fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid TOML in {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
}

/// Limits and deadlines that govern one agent run.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RunPolicy {
    /// Maximum number of model/tool rounds in one run.
    pub max_steps: usize,
    /// Maximum number of tool calls accepted from one assistant message.
    pub max_tool_calls_per_step: usize,
    /// Maximum duration allowed for one tool execution.
    pub tool_timeout_secs: u64,
    /// Optional wall-clock budget for model and tool execution in one run.
    /// `None` preserves an unlimited run deadline.
    pub run_timeout_secs: Option<u64>,
    /// Maximum number of historical messages sent to the model.
    pub max_context_messages: usize,
}

impl Default for RunPolicy {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_tool_calls_per_step: 16,
            tool_timeout_secs: 120,
            run_timeout_secs: None,
            max_context_messages: 128,
        }
    }
}

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
    pub tool_timeout_secs: u64,
    pub run_timeout_secs: Option<u64>,
    pub max_context_messages: usize,
    pub temperature: f32,
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

impl Default for Config {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            model: "your-model".to_string(),
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            workspace: PathBuf::from("./workspace"),
            session_file: PathBuf::from("./sessions/session.jsonl"),
            max_steps: 8,
            max_tool_calls_per_step: 16,
            tool_timeout_secs: 120,
            run_timeout_secs: None,
            max_context_messages: 128,
            temperature: 0.0,
            enable_thinking: Some(true),
            request_timeout_secs: 120,
            max_file_bytes: 1_048_576,
            max_tool_output_bytes: 65_536,
        }
    }
}

impl Config {
    /// Return the runtime policy represented by this configuration.
    pub fn run_policy(&self) -> RunPolicy {
        RunPolicy {
            max_steps: self.max_steps,
            max_tool_calls_per_step: self.max_tool_calls_per_step,
            tool_timeout_secs: self.tool_timeout_secs,
            run_timeout_secs: self.run_timeout_secs,
            max_context_messages: self.max_context_messages,
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref().to_path_buf();
        let raw = fs::read_to_string(&path).map_err(|source| ConfigError::Read {
            path: path.clone(),
            source,
        })?;
        let mut config: Self = toml::from_str(&raw).map_err(|source| ConfigError::Parse {
            path: path.clone(),
            source,
        })?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        config.workspace = resolve_path(base, config.workspace);
        config.session_file = resolve_path(base, config.session_file);
        Ok(config)
    }

    pub fn api_key(&self) -> Option<String> {
        self.api_key_env
            .as_deref()
            .and_then(|name| env::var(name).ok())
            .filter(|value| !value.is_empty())
    }
}

fn resolve_path(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn thinking_setting_loads_with_defaults_and_explicit_overrides() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("config.toml");

        for (raw, expected) in [
            ("", Some(true)),
            ("enable_thinking = true", Some(true)),
            ("enable_thinking = false", Some(false)),
        ] {
            fs::write(&path, raw).expect("config fixture should be written");

            let config = Config::load(&path).expect("config should load");

            assert_eq!(config.enable_thinking, expected, "config: {raw:?}");
        }
    }

    #[test]
    fn thinking_setting_rejects_non_boolean_values() {
        let error = toml::from_str::<Config>("enable_thinking = \"false\"")
            .expect_err("thinking must be a TOML boolean, not a string");

        assert!(error.message().contains("boolean"));
    }

    #[test]
    fn run_policy_uses_safe_defaults() {
        let policy = Config::default().run_policy();

        assert_eq!(policy.max_steps, 8);
        assert_eq!(policy.max_tool_calls_per_step, 16);
        assert_eq!(policy.tool_timeout_secs, 120);
        assert_eq!(policy.run_timeout_secs, None);
        assert_eq!(policy.max_context_messages, 128);
    }

    #[test]
    fn run_policy_loads_explicit_toml_values() {
        let config: Config = toml::from_str(
            r#"
                max_steps = 4
                max_tool_calls_per_step = 6
                tool_timeout_secs = 30
                run_timeout_secs = 300
                max_context_messages = 64
            "#,
        )
        .expect("runtime policy should parse");
        let policy = config.run_policy();

        assert_eq!(
            policy,
            RunPolicy {
                max_steps: 4,
                max_tool_calls_per_step: 6,
                tool_timeout_secs: 30,
                run_timeout_secs: Some(300),
                max_context_messages: 64,
            }
        );
    }
}
