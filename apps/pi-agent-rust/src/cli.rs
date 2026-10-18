use std::{path::PathBuf, str::FromStr};

use clap::{Parser, ValueEnum};
use pi_agent_runtime::{DeltaBoxFsMode, SandboxMode};
use pi_agent_rust::deltabox::DeltaBoxDisposition;
use pi_agent_rust::ToolChoice;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ToolChoiceArg {
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
pub(super) enum SandboxModeArg {
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
pub(super) enum DeltaBoxFsModeArg {
    CbcowThrough,
    CbcowPrivate,
    CopyPrivate,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(super) enum DeltaBoxActionArg {
    Status,
    Diff,
    Commit,
    Abort,
    Stop,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(super) enum DeltaBoxSuccessArg {
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
pub(super) struct Args {
    /// Optional TOML configuration file. Explicit CLI values override it.
    #[arg(long, value_name = "FILE")]
    pub(super) config: Option<PathBuf>,

    /// OpenAI-compatible API base URL.
    #[arg(long)]
    pub(super) base_url: Option<String>,

    /// Model identifier sent to the provider.
    #[arg(long)]
    pub(super) model: Option<String>,

    /// Environment variable containing an optional bearer token.
    #[arg(long)]
    pub(super) api_key_env: Option<String>,

    /// Request timeout in seconds.
    #[arg(long)]
    pub(super) timeout_secs: Option<u64>,

    /// Optional wall-clock timeout for the complete agent run.
    #[arg(long)]
    pub(super) run_timeout_secs: Option<u64>,

    /// Sampling temperature forwarded to the provider.
    #[arg(long)]
    pub(super) temperature: Option<f32>,

    /// Stable system prompt inserted before the session transcript.
    #[arg(long)]
    pub(super) system_prompt: Option<String>,

    /// Maximum number of model/tool rounds in one run.
    #[arg(long)]
    pub(super) max_steps: Option<usize>,

    /// Print model and tool progress to stderr.
    #[arg(long)]
    pub(super) verbose: bool,

    /// Allow irreversible tools such as bash to execute after runtime admission.
    #[arg(long)]
    pub(super) allow_unsafe_shell: bool,

    /// Sandbox boundary for bash: managed (default), deltabox, or disabled.
    #[arg(long, value_enum, default_value = "managed")]
    pub(super) sandbox_mode: SandboxModeArg,

    /// Optional DeltaBox executable path used with --sandbox-mode deltabox.
    #[arg(long, value_name = "FILE")]
    pub(super) deltabox_bin: Option<PathBuf>,

    /// DeltaBox filesystem provider. Private mode hides changes until commit.
    #[arg(long, value_enum, default_value = "cbcow-private")]
    pub(super) deltabox_fs_mode: DeltaBoxFsModeArg,

    /// Require the selected DeltaBox filesystem provider without fallback.
    #[arg(long)]
    pub(super) deltabox_require_fs_mode: bool,

    /// Keep successful DeltaBox changes pending review unless commit is explicit.
    #[arg(long, value_enum, default_value = "review")]
    pub(super) deltabox_on_success: DeltaBoxSuccessArg,

    /// Inspect or finish a durable DeltaBox session without querying a model.
    #[arg(long, value_enum, requires = "deltabox_session", conflicts_with_all = ["prompt", "prompt_words"])]
    pub(super) deltabox_action: Option<DeltaBoxActionArg>,

    /// Durable session ID used by --deltabox-action.
    #[arg(long, requires = "deltabox_action")]
    pub(super) deltabox_session: Option<String>,

    /// Tool selection policy: auto, none, required, or function:<name>.
    #[arg(long, value_name = "MODE", default_value = "auto")]
    pub(super) tool_choice: ToolChoiceArg,

    /// Workspace root exposed to built-in tools.
    #[arg(long)]
    pub(super) workspace: Option<PathBuf>,

    /// Root directory containing SKILL.md files to expose to the model.
    #[arg(long, value_name = "DIR")]
    pub(super) skills_dir: Option<PathBuf>,

    /// JSONL file used to persist the conversation transcript.
    #[arg(long)]
    pub(super) session_file: Option<PathBuf>,

    /// Existing session UUID used to group records in the transcript.
    #[arg(long)]
    pub(super) session_id: Option<String>,

    /// Resume the latest runtime run represented by the runtime JSONL ledger.
    #[arg(long)]
    pub(super) resume_runtime: bool,

    /// Reconcile one unknown runtime action without sending a model request.
    #[arg(long, conflicts_with_all = ["prompt", "prompt_words"])]
    pub(super) reconcile_action_id: Option<Uuid>,

    /// Evidence recorded with a runtime reconciliation decision.
    #[arg(long)]
    pub(super) reconcile_evidence: Option<String>,

    /// Mark the action as failed with this external error. Without this flag,
    /// the action is marked succeeded and --reconcile-output is used as output.
    #[arg(long)]
    pub(super) reconcile_error: Option<String>,

    /// JSON output recorded when reconciling an action as succeeded.
    #[arg(long)]
    pub(super) reconcile_output: Option<String>,

    /// Explicitly enable thinking; this is already the default.
    #[arg(long, conflicts_with = "no_think")]
    pub(super) think: bool,

    /// Disable thinking for providers that support this option.
    #[arg(long = "no-think", conflicts_with = "think")]
    pub(super) no_think: bool,

    /// A prompt supplied as one argument.
    #[arg(long, value_name = "PROMPT", conflicts_with = "prompt_words")]
    pub(super) prompt: Option<String>,

    /// Prompt words. Multiple words are joined with a single space.
    #[arg(value_name = "PROMPT", num_args = 0..)]
    pub(super) prompt_words: Vec<String>,
}

impl Args {
    pub(super) fn thinking_override(&self) -> Option<bool> {
        if self.think {
            Some(true)
        } else if self.no_think {
            Some(false)
        } else {
            None
        }
    }
}
