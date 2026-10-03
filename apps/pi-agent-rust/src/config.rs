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

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub base_url: String,
    pub model: String,
    pub api_key_env: Option<String>,
    pub workspace: PathBuf,
    pub session_file: PathBuf,
    pub max_steps: usize,
    pub temperature: f32,
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
            temperature: 0.0,
            request_timeout_secs: 120,
            max_file_bytes: 1_048_576,
            max_tool_output_bytes: 65_536,
        }
    }
}

impl Config {
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
