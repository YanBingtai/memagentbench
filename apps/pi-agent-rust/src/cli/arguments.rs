use std::path::PathBuf;

use clap::Parser;
use uuid::Uuid;

use super::{
    DeltaBoxActionArg, DeltaBoxFsModeArg, DeltaBoxSuccessArg, SandboxModeArg, ToolChoiceArg,
};

#[derive(Debug, Parser)]
#[command(
    name = "pi-agent-rust",
    version,
    about = "A small, OpenAI-compatible Pi agent client"
)]
pub(crate) struct Args {
    /// Optional TOML configuration file. Explicit CLI values override it.
    #[arg(long, value_name = "FILE")]
    pub(crate) config: Option<PathBuf>,

    /// OpenAI-compatible API base URL.
    #[arg(long)]
    pub(crate) base_url: Option<String>,

    /// Model identifier sent to the provider.
    #[arg(long)]
    pub(crate) model: Option<String>,

    /// Environment variable containing an optional bearer token.
    #[arg(long)]
    pub(crate) api_key_env: Option<String>,

    /// Request timeout in seconds.
    #[arg(long)]
    pub(crate) timeout_secs: Option<u64>,

    /// Optional wall-clock timeout for the complete agent run.
    #[arg(long)]
    pub(crate) run_timeout_secs: Option<u64>,

    /// Sampling temperature forwarded to the provider.
    #[arg(long)]
    pub(crate) temperature: Option<f32>,

    /// Stable system prompt inserted before the session transcript.
    #[arg(long)]
    pub(crate) system_prompt: Option<String>,

    /// Maximum number of model/tool rounds in one run.
    #[arg(long)]
    pub(crate) max_steps: Option<usize>,

    /// Print model and tool progress to stderr.
    #[arg(long)]
    pub(crate) verbose: bool,

    /// Allow irreversible tools such as bash to execute after runtime admission.
    #[arg(long)]
    pub(crate) allow_unsafe_shell: bool,

    /// Sandbox boundary for bash: managed (default), deltabox, or disabled.
    #[arg(long, value_enum, default_value = "managed")]
    pub(crate) sandbox_mode: SandboxModeArg,

    /// Optional DeltaBox executable path used with --sandbox-mode deltabox.
    #[arg(long, value_name = "FILE")]
    pub(crate) deltabox_bin: Option<PathBuf>,

    /// DeltaBox filesystem provider. Private mode hides changes until commit.
    #[arg(long, value_enum, default_value = "cbcow-private")]
    pub(crate) deltabox_fs_mode: DeltaBoxFsModeArg,

    /// Require the selected DeltaBox filesystem provider without fallback.
    #[arg(long)]
    pub(crate) deltabox_require_fs_mode: bool,

    /// Keep successful DeltaBox changes pending review unless commit is explicit.
    #[arg(long, value_enum, default_value = "review")]
    pub(crate) deltabox_on_success: DeltaBoxSuccessArg,

    /// Inspect or finish a durable DeltaBox session without querying a model.
    #[arg(
        long,
        value_enum,
        requires = "deltabox_session",
        conflicts_with_all = ["prompt", "prompt_words"]
    )]
    pub(crate) deltabox_action: Option<DeltaBoxActionArg>,

    /// Durable session ID used by --deltabox-action.
    #[arg(long, requires = "deltabox_action")]
    pub(crate) deltabox_session: Option<String>,

    /// Tool selection policy: auto, none, required, or function:<name>.
    #[arg(long, value_name = "MODE", default_value = "auto")]
    pub(crate) tool_choice: ToolChoiceArg,

    /// Workspace root exposed to built-in tools.
    #[arg(long)]
    pub(crate) workspace: Option<PathBuf>,

    /// Root directory containing SKILL.md files to expose to the model.
    #[arg(long, value_name = "DIR")]
    pub(crate) skills_dir: Option<PathBuf>,

    /// JSONL file used to persist the conversation transcript.
    #[arg(long)]
    pub(crate) session_file: Option<PathBuf>,

    /// Existing session UUID used to group records in the transcript.
    #[arg(long)]
    pub(crate) session_id: Option<String>,

    /// Resume the latest runtime run represented by the runtime JSONL ledger.
    #[arg(long)]
    pub(crate) resume_runtime: bool,

    /// Reconcile one unknown runtime action without sending a model request.
    #[arg(long, conflicts_with_all = ["prompt", "prompt_words"])]
    pub(crate) reconcile_action_id: Option<Uuid>,

    /// Evidence recorded with a runtime reconciliation decision.
    #[arg(long)]
    pub(crate) reconcile_evidence: Option<String>,

    /// Mark the action as failed with this external error. Without this flag,
    /// the action is marked succeeded and --reconcile-output is used as output.
    #[arg(long)]
    pub(crate) reconcile_error: Option<String>,

    /// JSON output recorded when reconciling an action as succeeded.
    #[arg(long)]
    pub(crate) reconcile_output: Option<String>,

    /// Explicitly enable thinking; this is already the default.
    #[arg(long, conflicts_with = "no_think")]
    pub(crate) think: bool,

    /// Disable thinking for providers that support this option.
    #[arg(long = "no-think", conflicts_with = "think")]
    pub(crate) no_think: bool,

    /// A prompt supplied as one argument.
    #[arg(long, value_name = "PROMPT", conflicts_with = "prompt_words")]
    pub(crate) prompt: Option<String>,

    /// Prompt words. Multiple words are joined with a single space.
    #[arg(value_name = "PROMPT", num_args = 0..)]
    pub(crate) prompt_words: Vec<String>,
}

impl Args {
    pub(crate) fn thinking_override(&self) -> Option<bool> {
        if self.think {
            Some(true)
        } else if self.no_think {
            Some(false)
        } else {
            None
        }
    }
}
