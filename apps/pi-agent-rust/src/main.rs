use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{Parser, ValueEnum};
use thiserror::Error;
use uuid::Uuid;

use pi_agent_rust::{
    AgentError, AgentEvent, AgentLoop, CancellationToken, Config, ConfigError, RegistryError,
    RunOutcome, SessionError, SessionStore, SkillCatalog, SkillLoader, SkillLoaderError,
    ToolChoice, ToolContext, ToolRegistry,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ToolChoiceArg {
    /// Let the model decide whether to call an advertised tool.
    Auto,
    /// Advertise no tool call for this request.
    None,
}

impl From<ToolChoiceArg> for ToolChoice {
    fn from(value: ToolChoiceArg) -> Self {
        match value {
            ToolChoiceArg::Auto => Self::Auto,
            ToolChoiceArg::None => Self::None,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "pi-agent-rust",
    version,
    about = "A small, OpenAI-compatible Pi agent client"
)]
struct Args {
    /// Optional TOML configuration file. Explicit CLI values override it.
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// OpenAI-compatible API base URL.
    #[arg(long)]
    base_url: Option<String>,

    /// Model identifier sent to the provider.
    #[arg(long)]
    model: Option<String>,

    /// Environment variable containing an optional bearer token.
    #[arg(long)]
    api_key_env: Option<String>,

    /// Request timeout in seconds.
    #[arg(long)]
    timeout_secs: Option<u64>,

    /// Optional wall-clock timeout for the complete agent run.
    #[arg(long)]
    run_timeout_secs: Option<u64>,

    /// Sampling temperature forwarded to the provider.
    #[arg(long)]
    temperature: Option<f32>,

    /// Maximum number of model/tool rounds in one run.
    #[arg(long)]
    max_steps: Option<usize>,

    /// Print model and tool progress to stderr.
    #[arg(long)]
    verbose: bool,

    /// Tool selection policy sent to the OpenAI-compatible provider.
    #[arg(long, value_enum, default_value = "auto")]
    tool_choice: ToolChoiceArg,

    /// Workspace root exposed to built-in tools.
    #[arg(long)]
    workspace: Option<PathBuf>,

    /// Root directory containing SKILL.md files to expose to the model.
    #[arg(long, value_name = "DIR")]
    skills_dir: Option<PathBuf>,

    /// JSONL file used to persist the conversation transcript.
    #[arg(long)]
    session_file: Option<PathBuf>,

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
    fn thinking_override(&self) -> Option<bool> {
        if self.think {
            Some(true)
        } else if self.no_think {
            Some(false)
        } else {
            None
        }
    }
}

#[derive(Debug, Error)]
enum AppError {
    #[error(transparent)]
    Agent(#[from] AgentError),

    #[error(transparent)]
    Config(#[from] ConfigError),

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
    let prompt = args
        .prompt
        .clone()
        .or_else(|| (!args.prompt_words.is_empty()).then(|| args.prompt_words.join(" ")))
        .ok_or(AppError::MissingPrompt)?;
    let config = load_config(&args)?;
    let skills = load_skills(args.skills_dir.as_deref(), config.max_file_bytes)?;

    let session_id = resolve_session_id(args.session_id.as_deref())?;

    tokio::fs::create_dir_all(&config.workspace)
        .await
        .map_err(|source| AppError::Workspace {
            path: config.workspace.clone(),
            source,
        })?;
    let mut tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))?;
    if let Some(skills) = skills.as_ref() {
        tools.register_skill_reader(skills.clone(), config.max_tool_output_bytes)?;
    }
    let session = SessionStore::open(config.session_file.clone(), session_id).await?;
    let agent = AgentLoop::from_config(&config, tools)?
        .with_tool_choice(args.tool_choice.into())
        .with_session(session)
        .with_runtime_ledger_file(config.session_file.with_extension("runtime.jsonl"));
    let agent = if let Some(skills) = skills {
        agent.with_skills(skills)
    } else {
        agent
    };
    let agent = if args.verbose {
        agent.with_event_sink(print_event)
    } else {
        agent
    };
    let cancellation = CancellationToken::new();
    let run = agent.run_with_cancellation(prompt, cancellation.clone());
    tokio::pin!(run);
    let result = tokio::select! {
        result = &mut run => result?,
        signal = tokio::signal::ctrl_c() => {
            match signal {
                Ok(()) => {
                    eprintln!("[agent] Ctrl-C received; cancelling the run");
                    cancellation.cancel();
                    run.await?
                }
                Err(error) => {
                    cancellation.cancel();
                    let _ = run.await;
                    return Err(AppError::Signal(error));
                }
            }
        }
    };

    if let Some(content) = result.final_message.content {
        println!("{content}");
    } else {
        println!("{}", serde_json::to_string_pretty(&result.final_message)?);
    }

    Ok(())
}

fn load_config(args: &Args) -> Result<Config, AppError> {
    let thinking_override = args.thinking_override();
    let has_config_file = args.config.is_some();
    let mut config = match args.config.as_deref() {
        Some(path) => Config::load(path)?,
        None => Config::default(),
    };
    if !has_config_file {
        if args.base_url.is_none() {
            config.base_url = "http://127.0.0.1:6273/v1".to_string();
        }
        if args.model.is_none() {
            config.model = "qwen3.8".to_string();
        }
        if args.temperature.is_none() {
            config.temperature = 0.7;
        }
    }
    if let Some(value) = &args.base_url {
        config.base_url = value.clone();
    }
    if let Some(value) = &args.model {
        config.model = value.clone();
    }
    if let Some(value) = &args.api_key_env {
        config.api_key_env = (!value.is_empty()).then_some(value.clone());
    }
    if let Some(value) = &args.workspace {
        config.workspace = value.clone();
    }
    if let Some(value) = &args.session_file {
        config.session_file = value.clone();
    }
    if let Some(value) = args.max_steps {
        config.max_steps = value;
    }
    if let Some(value) = args.run_timeout_secs {
        config.run_timeout_secs = Some(value);
    }
    if let Some(value) = args.temperature {
        config.temperature = value;
    }
    if let Some(value) = args.timeout_secs {
        config.request_timeout_secs = value;
    }
    if let Some(value) = thinking_override {
        config.enable_thinking = Some(value);
    }
    Ok(config)
}

fn load_skills(
    root: Option<&Path>,
    max_file_bytes: usize,
) -> Result<Option<SkillCatalog>, AppError> {
    let Some(root) = root else {
        return Ok(None);
    };

    let report = SkillLoader::new(root)
        .with_max_file_bytes(max_file_bytes as u64)
        .load()?;
    for diagnostic in report.diagnostics() {
        eprintln!(
            "warning: skill {}: {}",
            diagnostic.path.display(),
            diagnostic.error
        );
    }

    Ok((!report.catalog().is_empty()).then_some(report.catalog))
}

fn print_event(event: AgentEvent) {
    match event {
        AgentEvent::ModelStarted { step, model, .. } => {
            eprintln!("[agent] model step {step}: {model}");
        }
        AgentEvent::ToolStarted { step, call, .. } => {
            eprintln!(
                "[agent] tool step {step}: {} ({})",
                call.function.name, call.id
            );
        }
        AgentEvent::ToolFinished {
            tool_name,
            is_error,
            ..
        } => {
            let status = if is_error { "failed" } else { "completed" };
            eprintln!("[agent] tool {tool_name}: {status}");
        }
        AgentEvent::RunFinished {
            steps,
            outcome,
            error,
            ..
        } => match (outcome, error) {
            (RunOutcome::Completed, _) => eprintln!("[agent] completed in {steps} step(s)"),
            (outcome, Some(error)) => {
                eprintln!("[agent] {outcome:?} after {steps} step(s): {error}")
            }
            (outcome, None) => eprintln!("[agent] {outcome:?} after {steps} step(s)"),
        },
        AgentEvent::RunStarted { .. }
        | AgentEvent::AssistantDelta { .. }
        | AgentEvent::AssistantMessage { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn thinking_is_enabled_by_default() {
        let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
            .expect("arguments should parse");

        assert_eq!(args.thinking_override(), None);
    }

    #[test]
    fn think_flag_can_be_enabled() {
        let args = Args::try_parse_from(["pi-agent-rust", "--think", "--prompt", "hello"])
            .expect("arguments should parse");

        assert_eq!(args.thinking_override(), Some(true));
    }

    #[test]
    fn think_mode_can_be_disabled() {
        let args = Args::try_parse_from(["pi-agent-rust", "--no-think", "--prompt", "hello"])
            .expect("arguments should parse");

        assert_eq!(args.thinking_override(), Some(false));
    }

    #[test]
    fn tool_choice_defaults_to_auto() {
        let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
            .expect("arguments should parse");

        assert!(matches!(args.tool_choice, ToolChoiceArg::Auto));
    }

    #[test]
    fn tool_choice_none_is_parsed_and_mapped() {
        let args = Args::try_parse_from([
            "pi-agent-rust",
            "--tool-choice",
            "none",
            "--prompt",
            "hello",
        ])
        .expect("arguments should parse");

        assert!(matches!(args.tool_choice, ToolChoiceArg::None));
        assert_eq!(ToolChoice::from(args.tool_choice), ToolChoice::None);
    }

    #[test]
    fn run_timeout_is_unset_by_default() {
        let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
            .expect("arguments should parse");

        assert_eq!(args.run_timeout_secs, None);
    }

    #[test]
    fn run_timeout_can_be_configured() {
        let args = Args::try_parse_from([
            "pi-agent-rust",
            "--run-timeout-secs",
            "300",
            "--prompt",
            "hello",
        ])
        .expect("arguments should parse");

        assert_eq!(args.run_timeout_secs, Some(300));
    }

    #[test]
    fn verbose_is_disabled_by_default() {
        let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
            .expect("arguments should parse");

        assert!(!args.verbose);
    }

    #[test]
    fn verbose_flag_can_be_enabled() {
        let args = Args::try_parse_from(["pi-agent-rust", "--verbose", "hello"])
            .expect("arguments should parse");

        assert!(args.verbose);
    }

    #[test]
    fn skills_directory_can_be_selected() {
        let args = Args::try_parse_from([
            "pi-agent-rust",
            "--skills-dir",
            "/tmp/skills",
            "--prompt",
            "hello",
        ])
        .expect("arguments should parse");

        assert_eq!(args.skills_dir, Some(PathBuf::from("/tmp/skills")));
    }

    #[test]
    fn skills_directory_loads_a_catalog_for_the_agent() {
        let directory = tempdir().expect("temporary directory should exist");
        let skill_directory = directory.path().join("research");
        fs::create_dir_all(&skill_directory).expect("skill directory should be created");
        fs::write(
            skill_directory.join("SKILL.md"),
            "---\nname: research\ndescription: Research tasks\n---\nUse primary sources.",
        )
        .expect("skill fixture should be written");

        let catalog = load_skills(Some(directory.path()), 1024)
            .expect("skill directory should load")
            .expect("valid skill should produce a catalog");

        assert_eq!(catalog.len(), 1);
        assert_eq!(
            catalog.get("research").unwrap().content,
            "Use primary sources."
        );
    }

    #[test]
    fn skills_directory_registers_the_read_skill_tool() {
        let directory = tempdir().expect("temporary directory should exist");
        let skill_directory = directory.path().join("research");
        fs::create_dir_all(&skill_directory).expect("skill directory should be created");
        fs::write(
            skill_directory.join("SKILL.md"),
            "---\nname: research\ndescription: Research tasks\n---\nUse primary sources.",
        )
        .expect("skill fixture should be written");
        let skills = load_skills(Some(directory.path()), 1024)
            .expect("skill directory should load")
            .expect("valid skill should produce a catalog");
        let mut tools = ToolRegistry::with_read_file(ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        })
        .expect("read_file should register");

        tools
            .register_skill_reader(skills, 1024)
            .expect("read_skill should register");

        assert!(tools
            .definitions()
            .iter()
            .any(|definition| definition.function.name == "read_skill"));
    }

    #[test]
    fn skills_directory_uses_the_configured_file_limit() {
        let directory = tempdir().expect("temporary directory should exist");
        let skill_directory = directory.path().join("research");
        fs::create_dir_all(&skill_directory).expect("skill directory should be created");
        fs::write(
            skill_directory.join("SKILL.md"),
            "---\nname: research\ndescription: Research tasks\n---\nThis body is larger than the test limit.",
        )
        .expect("skill fixture should be written");

        let catalog = load_skills(Some(directory.path()), 16).expect("skill directory should load");

        assert!(catalog.is_none());
    }

    #[test]
    fn config_file_values_are_used_when_cli_does_not_override_them() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("agent.toml");
        fs::write(
            &path,
            r#"
                base_url = "http://config.example/v1"
                model = "config-model"
                workspace = "config-workspace"
                run_timeout_secs = 45
            "#,
        )
        .expect("config fixture should be written");
        let args = Args::try_parse_from([
            "pi-agent-rust",
            "--config",
            path.to_str().expect("config path should be valid UTF-8"),
            "--prompt",
            "hello",
        ])
        .expect("arguments should parse");

        let config = load_config(&args).expect("config should load");

        assert_eq!(config.base_url, "http://config.example/v1");
        assert_eq!(config.model, "config-model");
        assert_eq!(config.workspace, directory.path().join("config-workspace"));
        assert_eq!(config.run_timeout_secs, Some(45));
    }

    #[test]
    fn no_config_file_preserves_cli_defaults() {
        let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
            .expect("arguments should parse");

        let config = load_config(&args).expect("default config should load");

        assert_eq!(config.base_url, "http://127.0.0.1:6273/v1");
        assert_eq!(config.model, "qwen3.8");
        assert_eq!(config.temperature, 0.7);
    }

    #[test]
    fn explicit_cli_values_override_config_file_values() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("agent.toml");
        fs::write(&path, "model = \"config-model\"\n").expect("config fixture should be written");
        let args = Args::try_parse_from([
            "pi-agent-rust",
            "--config",
            path.to_str().expect("config path should be valid UTF-8"),
            "--model",
            "cli-model",
            "--prompt",
            "hello",
        ])
        .expect("arguments should parse");

        let config = load_config(&args).expect("config should load");

        assert_eq!(config.model, "cli-model");
    }

    #[test]
    fn missing_config_file_preserves_typed_error() {
        let directory = tempdir().expect("temporary directory should exist");
        let path = directory.path().join("missing.toml");
        let args = Args::try_parse_from([
            "pi-agent-rust",
            "--config",
            path.to_str().expect("config path should be valid UTF-8"),
            "--prompt",
            "hello",
        ])
        .expect("arguments should parse");

        let error = load_config(&args).expect_err("missing config should fail");

        assert!(matches!(error, AppError::Config(ConfigError::Read { .. })));
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
