use std::{path::PathBuf, process::ExitCode};

use clap::Parser;
use thiserror::Error;
use uuid::Uuid;

use pi_agent_rust::{
    AgentError, AgentLoop, Config, RegistryError, SessionError, SessionStore, ToolContext,
    ToolRegistry,
};

#[derive(Debug, Parser)]
#[command(
    name = "pi-agent-rust",
    version,
    about = "A small, OpenAI-compatible Pi agent client"
)]
struct Args {
    /// OpenAI-compatible API base URL.
    #[arg(long, default_value = "http://127.0.0.1:6273/v1")]
    base_url: String,

    /// Model identifier sent to the provider.
    #[arg(long, default_value = "qwen3.8")]
    model: String,

    /// Environment variable containing an optional bearer token.
    #[arg(long, default_value = "OPENAI_API_KEY")]
    api_key_env: String,

    /// Request timeout in seconds.
    #[arg(long, default_value_t = 120)]
    timeout_secs: u64,

    /// Sampling temperature forwarded to the provider.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Maximum number of model/tool rounds in one run.
    #[arg(long, default_value_t = 8)]
    max_steps: usize,

    /// Workspace root exposed to built-in tools.
    #[arg(long, default_value = "./workspace")]
    workspace: PathBuf,

    /// JSONL file used to persist the conversation transcript.
    #[arg(long, default_value = "./sessions/session.jsonl")]
    session_file: PathBuf,

    /// Existing session UUID used to group records in the transcript.
    #[arg(long)]
    session_id: Option<String>,

    /// Explicitly enable thinking; this is already the default.
    #[arg(long, conflicts_with = "no_think")]
    think: bool,

    /// Disable thinking for providers that support this option.
    #[arg(long = "no-think", conflicts_with = "think")]
    no_think: bool,

    /// A prompt supplied as one argument.
    #[arg(long, value_name = "PROMPT", conflicts_with = "prompt_words")]
    prompt: Option<String>,

    /// Prompt words. Multiple words are joined with a single space.
    #[arg(value_name = "PROMPT", num_args = 0..)]
    prompt_words: Vec<String>,
}

impl Args {
    fn thinking_enabled(&self) -> bool {
        self.think || !self.no_think
    }
}

#[derive(Debug, Error)]
enum AppError {
    #[error(transparent)]
    Agent(#[from] AgentError),

    #[error("failed to register built-in tool: {0}")]
    Registry(#[from] RegistryError),

    #[error("failed to create workspace {path}: {source}")]
    Workspace {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(transparent)]
    Session(#[from] SessionError),

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
}

fn resolve_session_id(raw: Option<&str>) -> Result<Uuid, AppError> {
    match raw {
        Some(value) => Uuid::parse_str(value).map_err(|source| AppError::InvalidSessionId {
            value: value.to_owned(),
            source,
        }),
        None => Ok(Uuid::new_v4()),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), AppError> {
    let args = Args::parse();
    let thinking_enabled = args.thinking_enabled();
    let prompt = args
        .prompt
        .clone()
        .or_else(|| (!args.prompt_words.is_empty()).then(|| args.prompt_words.join(" ")))
        .ok_or(AppError::MissingPrompt)?;
    let mut config = Config::default();
    config.base_url = args.base_url;
    config.model = args.model;
    config.api_key_env = (!args.api_key_env.is_empty()).then_some(args.api_key_env);
    config.workspace = args.workspace;
    config.session_file = args.session_file;
    config.max_steps = args.max_steps;
    config.temperature = args.temperature;
    config.request_timeout_secs = args.timeout_secs;
    config.enable_thinking = Some(thinking_enabled);
    let session_id = resolve_session_id(args.session_id.as_deref())?;

    tokio::fs::create_dir_all(&config.workspace)
        .await
        .map_err(|source| AppError::Workspace {
            path: config.workspace.clone(),
            source,
        })?;
    let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))?;
    let session = SessionStore::open(config.session_file.clone(), session_id).await?;
    let agent = AgentLoop::from_config(&config, tools)?.with_session(session);
    let result = agent.run(prompt).await?;

    if let Some(content) = result.final_message.content {
        println!("{content}");
    } else {
        println!("{}", serde_json::to_string_pretty(&result.final_message)?);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_is_enabled_by_default() {
        let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
            .expect("arguments should parse");

        assert!(args.thinking_enabled());
    }

    #[test]
    fn think_flag_can_be_enabled() {
        let args = Args::try_parse_from(["pi-agent-rust", "--think", "--prompt", "hello"])
            .expect("arguments should parse");

        assert!(args.thinking_enabled());
    }

    #[test]
    fn think_mode_can_be_disabled() {
        let args = Args::try_parse_from(["pi-agent-rust", "--no-think", "--prompt", "hello"])
            .expect("arguments should parse");

        assert!(!args.thinking_enabled());
    }

    #[test]
    fn missing_session_id_creates_a_new_id() {
        let id = resolve_session_id(None).expect("a missing id should create one");

        assert_ne!(id, Uuid::nil());
    }

    #[test]
    fn valid_session_id_is_preserved() {
        let expected = Uuid::new_v4();

        assert_eq!(
            resolve_session_id(Some(&expected.to_string())).expect("id should parse"),
            expected
        );
    }

    #[test]
    fn invalid_session_id_is_rejected() {
        assert!(matches!(
            resolve_session_id(Some("not-a-uuid")),
            Err(AppError::InvalidSessionId { .. })
        ));
    }
}
