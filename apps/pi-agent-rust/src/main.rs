use std::{
    path::{Path, PathBuf},
    process::ExitCode,
    str::FromStr,
    time::Duration,
};

use clap::{Parser, ValueEnum};
use pi_agent_runtime::{DeltaBoxClient, DeltaBoxConfig, DeltaBoxFsMode, SandboxMode};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

use pi_agent_rust::deltabox::{
    run_deltabox_worker, DeltaBoxDisposition, DeltaBoxSandbox, DELTABOX_WORKER_FLAG,
};
use pi_agent_rust::{
    AgentError, AgentEvent, AgentLoop, CancellationToken, ChatOptions, Config, ConfigError,
    ModelClient, ModelError, RegistryError, RunOutcome, RuntimeReconcileOutcome, SessionError,
    SessionStore, ShellTool, SkillCatalog, SkillLoader, SkillLoaderError, SubagentTool, ToolChoice,
    ToolContext, ToolRegistry, WebSearchTool,
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolChoiceArg {
    /// Let the model decide whether to call an advertised tool.
    Auto,
    /// Advertise no tool call for this request.
    None,
    /// Require the model to call one of the advertised tools.
    Required,
    /// Require one named function, for example function:read_file.
    Function { name: String },
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum SandboxModeArg {
    Managed,
    Disabled,
    Deltabox,
}

impl From<SandboxModeArg> for SandboxMode {
    fn from(value: SandboxModeArg) -> Self {
        match value {
            SandboxModeArg::Managed => Self::Managed,
            SandboxModeArg::Disabled => Self::Disabled,
            SandboxModeArg::Deltabox => Self::DeltaBox,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum DeltaBoxFsModeArg {
    CbcowThrough,
    CbcowPrivate,
    CopyPrivate,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum DeltaBoxActionArg {
    Status,
    Diff,
    Commit,
    Abort,
    Stop,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum DeltaBoxSuccessArg {
    Review,
    Commit,
    Abort,
}

impl From<DeltaBoxSuccessArg> for DeltaBoxDisposition {
    fn from(value: DeltaBoxSuccessArg) -> Self {
        match value {
            DeltaBoxSuccessArg::Review => Self::Review,
            DeltaBoxSuccessArg::Commit => Self::Commit,
            DeltaBoxSuccessArg::Abort => Self::Abort,
        }
    }
}

impl From<DeltaBoxFsModeArg> for DeltaBoxFsMode {
    fn from(value: DeltaBoxFsModeArg) -> Self {
        match value {
            DeltaBoxFsModeArg::CbcowThrough => Self::CbCowThrough,
            DeltaBoxFsModeArg::CbcowPrivate => Self::CbCowPrivate,
            DeltaBoxFsModeArg::CopyPrivate => Self::CopyPrivate,
        }
    }
}

impl FromStr for ToolChoiceArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim() {
            "auto" => Ok(Self::Auto),
            "none" => Ok(Self::None),
            "required" => Ok(Self::Required),
            value => match value.strip_prefix("function:") {
                Some(name) if valid_function_name(name.trim()) => Ok(Self::Function {
                    name: name.trim().to_string(),
                }),
                Some(_) => Err(
                    "function tool choice requires a 1-64 character name using only letters, numbers, '_' or '-'"
                        .to_string(),
                ),
                None => Err(format!(
                    "invalid tool choice {value:?}; expected auto, none, required, or function:<name>"
                )),
            },
        }
    }
}

fn valid_function_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

impl From<ToolChoiceArg> for ToolChoice {
    fn from(value: ToolChoiceArg) -> Self {
        match value {
            ToolChoiceArg::Auto => Self::Auto,
            ToolChoiceArg::None => Self::None,
            ToolChoiceArg::Required => Self::Required,
            ToolChoiceArg::Function { name } => Self::Function { name },
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

    /// Stable system prompt inserted before the session transcript.
    #[arg(long)]
    system_prompt: Option<String>,

    /// Maximum number of model/tool rounds in one run.
    #[arg(long)]
    max_steps: Option<usize>,

    /// Print model and tool progress to stderr.
    #[arg(long)]
    verbose: bool,

    /// Allow irreversible tools such as bash to execute after runtime admission.
    #[arg(long)]
    allow_unsafe_shell: bool,

    /// Sandbox boundary for bash: managed (default), deltabox, or disabled.
    #[arg(long, value_enum, default_value = "managed")]
    sandbox_mode: SandboxModeArg,

    /// Optional DeltaBox executable path used with --sandbox-mode deltabox.
    #[arg(long, value_name = "FILE")]
    deltabox_bin: Option<PathBuf>,

    /// DeltaBox filesystem provider. Private mode hides changes until commit.
    #[arg(long, value_enum, default_value = "cbcow-private")]
    deltabox_fs_mode: DeltaBoxFsModeArg,

    /// Require the selected DeltaBox filesystem provider without fallback.
    #[arg(long)]
    deltabox_require_fs_mode: bool,

    /// Keep successful DeltaBox changes pending review unless commit is explicit.
    #[arg(long, value_enum, default_value = "review")]
    deltabox_on_success: DeltaBoxSuccessArg,

    /// Inspect or finish a durable DeltaBox session without querying a model.
    #[arg(long, value_enum, requires = "deltabox_session", conflicts_with_all = ["prompt", "prompt_words"])]
    deltabox_action: Option<DeltaBoxActionArg>,

    /// Durable session ID used by --deltabox-action.
    #[arg(long, requires = "deltabox_action")]
    deltabox_session: Option<String>,

    /// Tool selection policy: auto, none, required, or function:<name>.
    #[arg(long, value_name = "MODE", default_value = "auto")]
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

    /// Resume the latest runtime run represented by the runtime JSONL ledger.
    #[arg(long)]
    resume_runtime: bool,

    /// Reconcile one unknown runtime action without sending a model request.
    #[arg(long, conflicts_with_all = ["prompt", "prompt_words"])]
    reconcile_action_id: Option<Uuid>,

    /// Evidence recorded with a runtime reconciliation decision.
    #[arg(long)]
    reconcile_evidence: Option<String>,

    /// Mark the action as failed with this external error. Without this flag,
    /// the action is marked succeeded and --reconcile-output is used as output.
    #[arg(long)]
    reconcile_error: Option<String>,

    /// JSON output recorded when reconciling an action as succeeded.
    #[arg(long)]
    reconcile_output: Option<String>,

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

fn resolve_tool_choice(
    choice: &ToolChoiceArg,
    tools: &ToolRegistry,
) -> Result<ToolChoice, AppError> {
    if let ToolChoiceArg::Function { name } = choice {
        let available = tools
            .model_definitions()
            .into_iter()
            .any(|definition| definition.function.name == *name);
        if !available {
            return Err(AppError::UnknownToolChoiceFunction { name: name.clone() });
        }
    }

    Ok(match choice {
        ToolChoiceArg::Auto => ToolChoice::Auto,
        ToolChoiceArg::None => ToolChoice::None,
        ToolChoiceArg::Required => ToolChoice::Required,
        ToolChoiceArg::Function { name } => ToolChoice::Function { name: name.clone() },
    })
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

fn main() -> ExitCode {
    let worker = std::env::args().nth(1).as_deref() == Some(DELTABOX_WORKER_FLAG);
    let mut builder = if worker {
        tokio::runtime::Builder::new_current_thread()
    } else {
        tokio::runtime::Builder::new_multi_thread()
    };
    let runtime = match builder.enable_all().build() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("error: failed to start async runtime: {error}");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(async {
        let result = if worker {
            run_deltabox_worker(std::env::args().skip(2).collect())
                .await
                .map_err(|error| AppError::ShellSandbox {
                    message: error.to_string(),
                })
                .and_then(|result| {
                    println!("{}", serde_json::to_string(&result)?);
                    Ok(())
                })
        } else {
            run().await
        };
        match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        }
    })
}

async fn run() -> Result<(), AppError> {
    let args = Args::parse();
    let config = load_config(&args)?;
    if let Some(action_id) = args.reconcile_action_id {
        return reconcile_runtime_action(&args, &config, action_id);
    }
    if let Some(action) = args.deltabox_action {
        let client = DeltaBoxClient::attach(
            deltabox_config(&args, &config)?,
            args.deltabox_session
                .as_deref()
                .expect("clap requires a session ID"),
        )
        .map_err(|error| AppError::ShellSandbox {
            message: error.to_string(),
        })?;
        let receipt = match action {
            DeltaBoxActionArg::Status => client.status(),
            DeltaBoxActionArg::Diff => client.diff(),
            DeltaBoxActionArg::Commit => client.commit(),
            DeltaBoxActionArg::Abort => client.abort(),
            DeltaBoxActionArg::Stop => client.stop(),
        }
        .map_err(|error| AppError::ShellSandbox {
            message: error.to_string(),
        })?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        return Ok(());
    }
    let prompt = args
        .prompt
        .clone()
        .or_else(|| (!args.prompt_words.is_empty()).then(|| args.prompt_words.join(" ")))
        .ok_or(AppError::MissingPrompt)?;
    let skills = load_skills(args.skills_dir.as_deref(), config.max_file_bytes)?;

    tokio::fs::create_dir_all(&config.workspace)
        .await
        .map_err(|source| AppError::Workspace {
            path: config.workspace.clone(),
            source,
        })?;
    let sandbox = if args.sandbox_mode == SandboxModeArg::Deltabox {
        let sandbox = DeltaBoxSandbox::start(
            deltabox_config(&args, &config)?,
            Duration::from_secs(config.tool_timeout_secs),
        )
        .map_err(|error| AppError::ShellSandbox {
            message: error.to_string(),
        })?;
        eprintln!(
            "[deltabox] session {}: {:?}, visibility={}, provider={}",
            sandbox.client().session_id(),
            sandbox.client().info().resolved_fs_mode,
            sandbox.client().info().host_visibility,
            sandbox.client().info().isolation_provider
        );
        if let Some(reason) = &sandbox.client().info().fallback_reason {
            eprintln!("[deltabox] filesystem fallback: {reason}");
        }
        if !sandbox.client().info().degraded_protections.is_empty() {
            eprintln!(
                "[deltabox] unavailable protections: {:?}",
                sandbox.client().info().degraded_protections
            );
        }
        Some(sandbox)
    } else {
        None
    };
    let result = run_agent(&args, &config, prompt, skills, sandbox.as_ref()).await;
    if let Some(sandbox) = sandbox {
        let disposition = if result.is_ok() {
            args.deltabox_on_success.into()
        } else {
            DeltaBoxDisposition::Abort
        };
        let session_id = sandbox.client().session_id().to_string();
        let finish = tokio::task::spawn_blocking(move || sandbox.finish(disposition))
            .await
            .map_err(|error| AppError::ShellSandbox {
                message: error.to_string(),
            })?;
        let receipt = finish.map_err(|error| AppError::ShellSandbox {
            message: format!(
                "session {session_id} needs recovery: {error}; agent result: {:?}",
                result.as_ref().err()
            ),
        })?;
        eprintln!(
            "[deltabox] {disposition:?} session {session_id}: {}",
            serde_json::to_string(&receipt)?
        );
        if receipt
            .get("folded_publications")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            > 0
        {
            eprintln!(
                "[deltabox] sibling changes were included; re-run project checks after commit"
            );
        }
    }
    let result = result?;
    if let Some(content) = result.final_message.content {
        println!("{content}");
    } else {
        println!("{}", serde_json::to_string_pretty(&result.final_message)?);
    }
    Ok(())
}

fn deltabox_config(args: &Args, config: &Config) -> Result<DeltaBoxConfig, AppError> {
    let mut sandbox = match &args.deltabox_bin {
        Some(binary) => DeltaBoxConfig::from_binary(binary, &config.workspace),
        None => {
            DeltaBoxConfig::discover(&config.workspace).map_err(|error| AppError::ShellSandbox {
                message: error.to_string(),
            })?
        }
    };
    sandbox.fs_mode = args.deltabox_fs_mode.into();
    sandbox.require_fs_mode = args.deltabox_require_fs_mode;
    Ok(sandbox)
}

async fn run_agent(
    args: &Args,
    config: &Config,
    prompt: String,
    skills: Option<SkillCatalog>,
    sandbox: Option<&DeltaBoxSandbox>,
) -> Result<pi_agent_rust::RunResult, AppError> {
    let session_id = resolve_session_id(args.session_id.as_deref())?;
    let tool_context = ToolContext::from_config(config);
    let mut tools = if let Some(sandbox) = sandbox {
        sandbox.tools(tool_context.clone())?
    } else {
        let mut tools = ToolRegistry::with_read_file(tool_context.clone())?;
        tools.register_list_files(tool_context.clone())?;
        tools.register_search_files(tool_context.clone())?;
        tools.register_write_file(tool_context.clone())?;
        tools.register_edit_file(tool_context.clone())?;
        let shell = ShellTool::with_sandbox_mode(tool_context.clone(), args.sandbox_mode.into())
            .map_err(|error| AppError::ShellSandbox {
                message: error.to_string(),
            })?;
        tools.register(shell)?;
        tools
    };
    if let Some(web_search) = WebSearchTool::from_env(config.max_tool_output_bytes) {
        tools.register(web_search)?;
    }
    if let Some(skills) = skills.as_ref() {
        tools.register_skill_reader(skills.clone(), config.max_tool_output_bytes)?;
    }
    let child_skills = skills.clone();
    let child_tools = tools.clone();
    let subagent_client = ModelClient::new(
        &config.base_url,
        Duration::from_secs(config.request_timeout_secs),
    )?;
    let subagent = SubagentTool::new(
        subagent_client,
        config.model.clone(),
        config.api_key(),
        ChatOptions {
            temperature: config.temperature,
            enable_thinking: config.enable_thinking,
        },
        child_tools,
        config.run_policy(),
        config.tool_timeout_secs,
        config.max_tool_output_bytes,
    );
    let subagent = match child_skills {
        Some(skills) => subagent.with_skills(skills),
        None => subagent,
    };
    tools.register(subagent)?;
    let tool_choice = resolve_tool_choice(&args.tool_choice, &tools)?;
    let session = SessionStore::open(config.session_file.clone(), session_id).await?;
    let agent = AgentLoop::from_config(config, tools)?
        .with_tool_choice(tool_choice)
        .with_session(session)
        .with_runtime_ledger_file(config.session_file.with_extension("runtime.jsonl"));
    let agent = agent.with_allow_irreversible_tools(args.allow_unsafe_shell);
    let agent = if args.resume_runtime {
        agent.with_runtime_recovery()
    } else {
        agent
    };
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
    let run_result = tokio::select! {
        result = &mut run => result,
        signal = tokio::signal::ctrl_c() => {
            match signal {
                Ok(()) => {
                    eprintln!("[agent] Ctrl-C received; cancelling the run");
                    cancellation.cancel();
                    run.await
                }
                Err(error) => {
                    cancellation.cancel();
                    let _ = run.await;
                    return Err(AppError::Signal(error));
                }
            }
        }
    };

    Ok(run_result?)
}

fn reconcile_runtime_action(args: &Args, config: &Config, action_id: Uuid) -> Result<(), AppError> {
    let evidence = args
        .reconcile_evidence
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or(AppError::MissingReconcileEvidence)?
        .to_string();
    let outcome = if let Some(error) = &args.reconcile_error {
        RuntimeReconcileOutcome::Failed {
            error: error.clone(),
            evidence,
        }
    } else {
        let output = match &args.reconcile_output {
            Some(raw) => serde_json::from_str(raw).map_err(AppError::InvalidReconcileOutput)?,
            None => Value::Null,
        };
        RuntimeReconcileOutcome::Succeeded { output, evidence }
    };
    let agent = AgentLoop::from_config(config, ToolRegistry::new())?
        .with_runtime_ledger_file(config.session_file.with_extension("runtime.jsonl"));
    agent.reconcile_runtime_action(action_id, outcome)?;
    println!("reconciled runtime action {action_id}");
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
    if let Some(value) = &args.system_prompt {
        config.system_prompt = (!value.trim().is_empty()).then_some(value.clone());
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
        AgentEvent::ModelRetry {
            step,
            attempt,
            delay_ms,
            error,
            ..
        } => {
            eprintln!("[agent] model retry {attempt} after step {step} in {delay_ms}ms: {error}");
        }
        AgentEvent::ToolStarted { step, call, .. } => {
            eprintln!(
                "[agent] tool step {step}: {} ({})",
                call.function.name, call.id
            );
        }
        AgentEvent::ToolFinished {
            tool_name,
            status,
            duration_ms,
            ..
        } => {
            eprintln!("[agent] tool {tool_name}: {status} ({duration_ms}ms)");
        }
        AgentEvent::ToolUpdate {
            tool_name, content, ..
        } => {
            eprintln!("[agent] tool {tool_name}: {content}");
        }
        AgentEvent::RunFinished {
            steps,
            outcome,
            error,
            ..
        } => match (outcome, error) {
            (RunOutcome::Completed, _) => eprintln!("[agent] completed in {steps} step(s)"),
            (RunOutcome::Unknown, Some(error)) => {
                eprintln!("[agent] outcome unknown after {steps} step(s): {error}")
            }
            (outcome, Some(error)) => {
                eprintln!("[agent] {outcome:?} after {steps} step(s): {error}")
            }
            (outcome, None) => eprintln!("[agent] {outcome:?} after {steps} step(s)"),
        },
        AgentEvent::TurnStarted { step, .. } => {
            eprintln!("[agent] turn {step} started");
        }
        AgentEvent::TurnFinished { step, .. } => {
            eprintln!("[agent] turn {step} finished");
        }
        AgentEvent::MemoryDecision {
            step,
            decision_kind,
            injection_count,
            ..
        } => {
            eprintln!(
                "[agent] memory step {step}: {decision_kind} ({injection_count} injection(s))"
            );
        }
        AgentEvent::RunStarted { .. }
        | AgentEvent::AssistantDelta { .. }
        | AgentEvent::ReasoningDelta { .. }
        | AgentEvent::ToolCallDelta { .. }
        | AgentEvent::AssistantStreamReset { .. }
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

        assert!(matches!(&args.tool_choice, ToolChoiceArg::Auto));
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

        assert!(matches!(&args.tool_choice, ToolChoiceArg::None));
        assert_eq!(ToolChoice::from(args.tool_choice), ToolChoice::None);
    }

    #[test]
    fn tool_choice_required_is_parsed_and_mapped() {
        let args = Args::try_parse_from([
            "pi-agent-rust",
            "--tool-choice",
            "required",
            "--prompt",
            "hello",
        ])
        .expect("required tool choice should parse");

        assert!(matches!(&args.tool_choice, ToolChoiceArg::Required));
        assert_eq!(ToolChoice::from(args.tool_choice), ToolChoice::Required);
    }

    #[test]
    fn tool_choice_function_is_parsed_and_mapped() {
        let args = Args::try_parse_from([
            "pi-agent-rust",
            "--tool-choice",
            "function:read_file",
            "--prompt",
            "hello",
        ])
        .expect("named function tool choice should parse");

        assert_eq!(
            ToolChoice::from(args.tool_choice),
            ToolChoice::Function {
                name: "read_file".to_string()
            }
        );
    }

    #[test]
    fn tool_choice_function_rejects_an_empty_name() {
        let error = Args::try_parse_from([
            "pi-agent-rust",
            "--tool-choice",
            "function:",
            "--prompt",
            "hello",
        ])
        .expect_err("an empty function name should be rejected");

        assert!(error.to_string().contains("1-64 character name"));
    }

    #[test]
    fn tool_choice_function_rejects_invalid_name_characters() {
        let error = Args::try_parse_from([
            "pi-agent-rust",
            "--tool-choice",
            "function:read.file",
            "--prompt",
            "hello",
        ])
        .expect_err("invalid function name characters should be rejected");

        assert!(error.to_string().contains("1-64 character name"));
    }

    #[test]
    fn tool_choice_function_must_be_registered_and_model_visible() {
        let directory = tempdir().expect("temporary directory should exist");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let tools = ToolRegistry::with_read_file(context).expect("read tool should register");

        let known = resolve_tool_choice(
            &ToolChoiceArg::Function {
                name: "read_file".to_string(),
            },
            &tools,
        )
        .expect("registered tool should be selectable");
        assert_eq!(
            known,
            ToolChoice::Function {
                name: "read_file".to_string()
            }
        );

        let error = resolve_tool_choice(
            &ToolChoiceArg::Function {
                name: "missing_tool".to_string(),
            },
            &tools,
        )
        .expect_err("unknown tool should fail before the model request");
        assert!(matches!(
            error,
            AppError::UnknownToolChoiceFunction { name } if name == "missing_tool"
        ));
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
    fn system_prompt_can_be_configured() {
        let args = Args::try_parse_from([
            "pi-agent-rust",
            "--system-prompt",
            "Act as a careful coding agent.",
            "--prompt",
            "hello",
        ])
        .expect("system prompt should parse");

        assert_eq!(
            args.system_prompt.as_deref(),
            Some("Act as a careful coding agent.")
        );
    }

    #[test]
    fn verbose_is_disabled_by_default() {
        let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
            .expect("arguments should parse");

        assert!(!args.verbose);
    }

    #[test]
    fn unsafe_shell_requires_an_explicit_flag() {
        let default_args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
            .expect("arguments should parse");
        assert!(!default_args.allow_unsafe_shell);

        let enabled_args =
            Args::try_parse_from(["pi-agent-rust", "--allow-unsafe-shell", "--prompt", "hello"])
                .expect("unsafe shell flag should parse");
        assert!(enabled_args.allow_unsafe_shell);
    }

    #[test]
    fn sandbox_mode_defaults_to_managed_and_accepts_explicit_disabled_mode() {
        let default_args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
            .expect("arguments should parse");
        assert_eq!(default_args.sandbox_mode, SandboxModeArg::Managed);

        let disabled_args = Args::try_parse_from([
            "pi-agent-rust",
            "--sandbox-mode",
            "disabled",
            "--prompt",
            "hello",
        ])
        .expect("disabled sandbox mode should parse");
        assert_eq!(disabled_args.sandbox_mode, SandboxModeArg::Disabled);

        let deltabox_args = Args::try_parse_from([
            "pi-agent-rust",
            "--sandbox-mode",
            "deltabox",
            "--deltabox-fs-mode",
            "copy-private",
            "--prompt",
            "hello",
        ])
        .expect("deltabox sandbox mode should parse");
        assert_eq!(deltabox_args.sandbox_mode, SandboxModeArg::Deltabox);
        assert_eq!(
            deltabox_args.deltabox_fs_mode,
            DeltaBoxFsModeArg::CopyPrivate
        );
    }

    #[test]
    fn verbose_flag_can_be_enabled() {
        let args = Args::try_parse_from(["pi-agent-rust", "--verbose", "hello"])
            .expect("arguments should parse");

        assert!(args.verbose);
    }

    #[test]
    fn runtime_reconciliation_arguments_parse_without_a_prompt() {
        let action_id = Uuid::new_v4();
        let args = Args::try_parse_from([
            "pi-agent-rust",
            "--reconcile-action-id",
            &action_id.to_string(),
            "--reconcile-evidence",
            "provider-check-1",
            "--reconcile-error",
            "provider confirmed no execution",
        ])
        .expect("reconciliation arguments should parse");

        assert_eq!(args.reconcile_action_id, Some(action_id));
        assert_eq!(args.reconcile_evidence.as_deref(), Some("provider-check-1"));
        assert_eq!(
            args.reconcile_error.as_deref(),
            Some("provider confirmed no execution")
        );
    }

    #[test]
    fn runtime_reconciliation_rejects_a_prompt() {
        let error = Args::try_parse_from([
            "pi-agent-rust",
            "--reconcile-action-id",
            &Uuid::new_v4().to_string(),
            "--prompt",
            "hello",
        ])
        .expect_err("reconciliation must be a standalone operation");

        assert!(error.to_string().contains("cannot be used with"));
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
    fn skills_directory_registers_the_default_filesystem_tools() {
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
        let tool_context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let mut tools =
            ToolRegistry::with_read_file(tool_context.clone()).expect("read_file should register");
        tools
            .register_write_file(tool_context.clone())
            .expect("write_file should register");
        tools
            .register_list_files(tool_context.clone())
            .expect("list_files should register");
        tools
            .register_search_files(tool_context.clone())
            .expect("search_files should register");
        tools
            .register_edit_file(tool_context)
            .expect("edit_file should register");

        tools
            .register_skill_reader(skills, 1024)
            .expect("read_skill should register");

        assert!(tools
            .definitions()
            .iter()
            .any(|definition| definition.function.name == "read_skill"));
        assert!(tools
            .definitions()
            .iter()
            .any(|definition| definition.function.name == "write_file"));
        assert!(tools
            .definitions()
            .iter()
            .any(|definition| definition.function.name == "edit_file"));
        assert!(tools
            .definitions()
            .iter()
            .any(|definition| definition.function.name == "list_files"));
        assert!(tools
            .definitions()
            .iter()
            .any(|definition| definition.function.name == "search_files"));
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
