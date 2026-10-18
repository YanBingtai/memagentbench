use std::path::PathBuf;

use pi_agent_rust::{
    AgentError, ConfigError, ModelError, RegistryError, SessionError, SkillLoaderError,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub(super) enum AppError {
    #[error(transparent)]
    Agent(#[from] AgentError),

    #[error(transparent)]
    Config(#[from] ConfigError),

    #[error(transparent)]
    Model(#[from] ModelError),

    #[error("failed to register built-in tool: {0}")]
    Registry(#[from] RegistryError),

    #[error("failed to initialize bash sandbox: {message}")]
    ShellSandbox { message: String },

    #[error("failed to create workspace {path}: {source}")]
    Workspace {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(transparent)]
    Session(#[from] SessionError),

    #[error(transparent)]
    Skills(#[from] SkillLoaderError),

    #[error("failed to install Ctrl-C handler: {0}")]
    Signal(#[source] std::io::Error),

    #[error("invalid session id {value}: {source}")]
    InvalidSessionId {
        value: String,
        #[source]
        source: uuid::Error,
    },

    #[error("failed to serialize assistant message: {0}")]
    Serialize(#[from] serde_json::Error),

    #[error("a prompt is required; pass --prompt or provide positional prompt words")]
    MissingPrompt,

    #[error("--reconcile-evidence must not be empty")]
    MissingReconcileEvidence,

    #[error("invalid --reconcile-output JSON: {0}")]
    InvalidReconcileOutput(#[source] serde_json::Error),

    #[error("tool choice refers to an unavailable model-visible tool: {name}")]
    UnknownToolChoiceFunction { name: String },
}
