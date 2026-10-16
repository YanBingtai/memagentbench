use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex as StdMutex,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use pi_agent_runtime::{
    canonical_json_digest, ActionId, ActionProposal, AdmissionPolicy, AdmissionRequest, Budget,
    CommitGrant, EventLedger, EventSequence, InMemoryLedger, InjectionDecision, JsonlLedger,
    MemoryHookContext, MemoryInjection, MemoryProposal, MemoryRevision, PolicyRevision,
    ReconciliationOutcome, ResourceScope, RiskClass, RunId, RuntimeCoordinator,
};
use serde_json::{json, Value};
use thiserror::Error;
use tokio::sync::Notify;
use uuid::Uuid;

const MAX_QUEUED_PROMPTS: usize = 32;
const MAX_QUEUED_PROMPT_BYTES: usize = 64 * 1024;

use crate::{
    config::{Config, RunPolicy},
    events::{AgentEvent, RunContext, RunOutcome},
    message::{Message, ToolCall, ToolDefinition, ToolResultMetadata},
    model::{
        ChatModel, ChatOptions, ChatRequest, ModelClient, ModelError, ModelUsage, ToolCallDelta,
        ToolChoice,
    },
    session::{SessionError, SessionStore},
    skills::SkillCatalog,
    tools::{
        PreparedToolCall, ToolConcurrency, ToolExecutionContext, ToolGateway, ToolRegistry,
        ToolReplayPolicy, ToolResult,
    },
};

/// Errors that stop an agent run before it produces a final assistant message.
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("prompt must not be empty")]
    EmptyPrompt,

    #[error("cannot start another run while this agent is already active")]
    RunBusy,

    #[error("{kind} prompt must not be empty")]
    EmptyQueuedPrompt { kind: &'static str },

    #[error("{kind} queue is full (limit {limit})")]
    QueueFull { kind: &'static str, limit: usize },

    #[error("{kind} prompt exceeds the {limit}-byte limit")]
    QueuedPromptTooLarge { kind: &'static str, limit: usize },

    #[error("{kind} prompts can only be queued before the first run or during an active run")]
    ControlQueueRequiresActiveRun { kind: &'static str },

    #[error("max_steps must be greater than zero")]
    InvalidMaxSteps,

    #[error("run policy field {field} must be greater than zero")]
    InvalidRunPolicy { field: &'static str },

    #[error("agent reached its maximum of {max_steps} model steps")]
    MaxStepsExceeded { max_steps: usize },

    #[error("model response ended with finish_reason={reason}")]
    ModelTerminated { reason: String },

    #[error("agent run timed out after {timeout_ms} milliseconds")]
    RunTimedOut { timeout_ms: u64 },

    #[error("agent run was cancelled by the caller")]
    RunCancelled,

    #[error(
        "model context requires {required_bytes} serialized bytes, which exceeds the {limit_bytes}-byte limit"
    )]
    ContextTooLarge {
        required_bytes: usize,
        limit_bytes: usize,
    },

    #[error("context compaction failed: {0}")]
    ContextCompaction(String),

    #[error("model context has invalid message structure: {reason}")]
    InvalidContext { reason: String },

    #[error("runtime ledger failed: {0}")]
    Runtime(String),

    #[error(
        "supplied session identity {provided:?} does not match the attached session {expected}"
    )]
    SessionContextMismatch {
        expected: Uuid,
        provided: Option<Uuid>,
    },

    #[error("session/runtime mismatch for action {action_id}: {reason}")]
    RuntimeSessionMismatch { action_id: Uuid, reason: String },

    #[error(
        "session contains tool result {call_id:?} with outcome {status:?} but no runtime identity"
    )]
    UnidentifiedToolOutcome { call_id: String, status: String },

    #[error("runtime action {action_id} was not found among pending actions")]
    RuntimeActionNotFound { action_id: Uuid },

    #[error("runtime action {action_id} for tool {tool_name} has unknown outcome: {reason}")]
    RuntimeActionUnknown {
        action_id: Uuid,
        tool_name: String,
        reason: String,
    },

    #[error("tool execution was terminated by policy: {reason}")]
    ToolTerminated { reason: String },

    #[error("memory hook failed: {0}")]
    MemoryHook(String),

    #[error("after-run hook failed: {0}")]
    AfterRunHook(String),

    #[error("model requested {count} tool calls in step {step}; limit is {limit}")]
    TooManyToolCalls {
        step: usize,
        count: usize,
        limit: usize,
    },

    #[error("invalid tool call in step {step} at index {index}: {reason}")]
    InvalidToolCall {
        step: usize,
        index: usize,
        reason: String,
    },

    #[error(transparent)]
    Model(#[from] ModelError),

    #[error(transparent)]
    Session(#[from] SessionError),
}

pub type ToolHookFuture<T> = Pin<Box<dyn Future<Output = Result<T, String>> + Send>>;

#[derive(Debug, Clone)]
pub struct ToolHookContext {
    pub run: RunContext,
    pub step: usize,
    pub call: ToolCall,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeforeToolDecision {
    Allow,
    Block {
        reason: String,
    },
    /// Stop the current tool batch and end the run before admission.
    Terminate {
        reason: String,
    },
}

#[derive(Debug, Clone)]
pub struct ToolResultContext {
    pub run: RunContext,
    pub step: usize,
    pub call: ToolCall,
    pub arguments: Value,
    pub content: String,
    pub is_error: bool,
    pub details: Value,
    pub terminate: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ToolResultOverride {
    pub content: Option<String>,
    pub is_error: Option<bool>,
    pub details: Option<Value>,
    pub terminate: Option<bool>,
}

type BeforeToolHook =
    Arc<dyn Fn(ToolHookContext) -> ToolHookFuture<BeforeToolDecision> + Send + Sync>;
type AfterToolHook =
    Arc<dyn Fn(ToolResultContext) -> ToolHookFuture<ToolResultOverride> + Send + Sync>;

enum ToolHookWait<T> {
    Completed(Result<T, String>),
    TimedOut(Duration),
    Cancelled,
}

#[derive(Debug)]
enum ParallelExecutionOutcome {
    Completed(Result<ToolResult, crate::tools::ToolError>),
    ToolTimedOut,
    RunTimedOut(Duration),
    Cancelled,
}

struct ParallelBatchOutcome {
    all_tools_terminate: bool,
    last_tool_message: Option<Message>,
    terminal_error: Option<AgentError>,
}

/// A clonable cancellation signal shared by an agent run and its caller.
///
/// Cancellation is cooperative: the loop stops waiting for the current model
/// or tool future, then persists structural tool results before returning.
/// Clones all observe the same request, so a CLI, TUI, or service handler can
/// retain one clone while the agent owns another.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    state: Arc<CancellationState>,
}

#[derive(Debug, Default)]
struct CancellationState {
    requested: AtomicBool,
    notify: Notify,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Calling this more than once is harmless.
    pub fn cancel(&self) {
        if !self.state.requested.swap(true, Ordering::Release) {
            self.state.notify.notify_waiters();
            // Keep one permit for a waiter that starts between the atomic
            // check and registration with Notify.
            self.state.notify.notify_one();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.requested.load(Ordering::Acquire)
    }

    /// Wait until cancellation is requested.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let notified = self.state.notify.notified();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}

/// The completed transcript and final answer from one agent run.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub messages: Vec<Message>,
    pub final_message: Message,
    pub steps: usize,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct RunStats {
    steps: usize,
    model_retries: usize,
    has_unknown_action: bool,
    model_requests: usize,
    input_usage_complete: bool,
    output_usage_complete: bool,
    total_usage_complete: bool,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

impl Default for RunStats {
    fn default() -> Self {
        Self {
            steps: 0,
            model_retries: 0,
            has_unknown_action: false,
            model_requests: 0,
            input_usage_complete: true,
            output_usage_complete: true,
            total_usage_complete: true,
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
        }
    }
}

impl RunStats {
    fn record_model_usage(&mut self, usage: Option<ModelUsage>) {
        self.model_requests = self.model_requests.saturating_add(1);
        let Some(usage) = usage else {
            self.input_usage_complete = false;
            self.output_usage_complete = false;
            self.total_usage_complete = false;
            return;
        };

        record_token(
            &mut self.input_tokens,
            &mut self.input_usage_complete,
            usage.input_tokens,
        );
        record_token(
            &mut self.output_tokens,
            &mut self.output_usage_complete,
            usage.output_tokens,
        );
        record_token(
            &mut self.total_tokens,
            &mut self.total_usage_complete,
            usage.total_tokens,
        );
    }

    fn model_usage(&self) -> (Option<u64>, Option<u64>, Option<u64>) {
        if self.model_requests == 0 {
            return (None, None, None);
        }
        (
            self.input_tokens.filter(|_| self.input_usage_complete),
            self.output_tokens.filter(|_| self.output_usage_complete),
            self.total_tokens.filter(|_| self.total_usage_complete),
        )
    }
}

fn record_token(slot: &mut Option<u64>, complete: &mut bool, value: Option<u64>) {
    let Some(value) = value else {
        *complete = false;
        return;
    };
    *slot = Some(slot.unwrap_or_default().saturating_add(value));
}

/// A runtime action that stopped with an unknown external outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingRuntimeAction {
    pub action_id: Uuid,
    pub grant_id: Uuid,
    pub run_id: Uuid,
    pub tool_name: String,
    pub idempotency_key: String,
    pub argument_digest: String,
    pub risk: RiskClass,
    pub issued_at_ms: u64,
    pub expires_at_ms: Option<u64>,
    /// Whether the registered tool permits an automatic replay policy.
    ///
    /// This is advisory metadata for a reconciler. The runtime still owns the
    /// final decision and unknown actions remain blocked until reconciliation.
    pub replay_policy: ToolReplayPolicy,
}

/// Evidence supplied by an external reconciler for an unknown action.
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeReconcileOutcome {
    Succeeded { output: Value, evidence: String },
    Failed { error: String, evidence: String },
}

/// Input supplied to a memory hook before one model request.
///
/// The hook receives an owned, already-truncated context so it can perform
/// asynchronous retrieval without borrowing the agent loop across an await.
#[derive(Debug, Clone)]
pub struct MemoryHookRequest {
    pub context: MemoryHookContext,
    pub messages: Vec<Message>,
}

pub type MemoryHookFuture =
    Pin<Box<dyn Future<Output = Result<Option<MemoryProposal>, String>> + Send>>;

type MemoryProposalHook = Arc<dyn Fn(MemoryHookRequest) -> MemoryHookFuture + Send + Sync>;

/// Snapshot supplied to an optional hook after one agent run.
///
/// `transcript` is the canonical runtime transcript accumulated before any
/// ephemeral memory or skill context is added to a model request. It remains
/// available when the run fails after producing assistant or tool messages.
/// The core loop keeps this contract independent from any memory backend.
#[derive(Debug, Clone)]
pub struct AfterRunRequest {
    pub context: RunContext,
    pub transcript: Vec<Message>,
    pub steps: usize,
    pub model_retries: usize,
    pub outcome: RunOutcome,
    pub result: Option<RunResult>,
    pub error: Option<String>,
}

pub type AfterRunFuture = Pin<Box<dyn Future<Output = Result<(), String>> + Send>>;

type AfterRunHook = Arc<dyn Fn(AfterRunRequest) -> AfterRunFuture + Send + Sync>;

/// Why the agent asked a caller-provided compactor to transform context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextCompactionReason {
    ByteBudget,
}

/// Input supplied to the optional context compaction hook.
///
/// The returned messages are request-only context. They are not written to
/// the session automatically; the durable transcript remains the source of
/// truth for recovery and auditing.
#[derive(Debug, Clone)]
pub struct ContextCompactionRequest {
    pub context: RunContext,
    pub step: usize,
    pub reason: ContextCompactionReason,
    pub messages: Vec<Message>,
    pub max_context_messages: usize,
    pub max_context_bytes: usize,
}

pub type ContextCompactionFuture =
    Pin<Box<dyn Future<Output = Result<Vec<Message>, String>> + Send>>;

type ContextCompactionHook =
    Arc<dyn Fn(ContextCompactionRequest) -> ContextCompactionFuture + Send + Sync>;

/// Static limits and revisions used to build each runtime memory context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryHookConfig {
    pub namespace: String,
    pub memory_revision: MemoryRevision,
    pub policy_revision: PolicyRevision,
    pub max_tokens: u32,
    pub max_candidates: u32,
    pub max_projection_bytes: u32,
}

impl Default for MemoryHookConfig {
    fn default() -> Self {
        Self {
            namespace: "default".to_owned(),
            memory_revision: MemoryRevision::INITIAL,
            policy_revision: PolicyRevision::INITIAL,
            max_tokens: 1_024,
            max_candidates: 16,
            max_projection_bytes: 32 * 1024,
        }
    }
}

#[derive(Debug, Default)]
struct ControlState {
    steering: VecDeque<String>,
    follow_up: VecDeque<String>,
    active: bool,
    has_run: bool,
}

struct RunGuard {
    control: Arc<StdMutex<ControlState>>,
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        let mut control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        control.steering.clear();
        control.follow_up.clear();
        control.active = false;
    }
}

/// Receives ephemeral runtime events from one agent run.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: AgentEvent);
}

impl<F> EventSink for F
where
    F: Fn(AgentEvent) + Send + Sync,
{
    fn emit(&self, event: AgentEvent) {
        self(event);
    }
}

/// Coordinates model calls, tool execution, and transcript updates.
pub struct AgentLoop<M = ModelClient> {
    client: M,
    tools: ToolGateway,
    model: String,
    api_key: Option<String>,
    options: ChatOptions,
    policy: RunPolicy,
    allow_irreversible_tools: bool,
    run_timeout: Option<Duration>,
    session: Option<SessionStore>,
    runtime_ledger_file: Option<std::path::PathBuf>,
    runtime_recovery: bool,
    system_prompt: Option<String>,
    skills: Option<SkillCatalog>,
    event_sink: Option<Arc<dyn EventSink>>,
    before_tool_hook: Option<BeforeToolHook>,
    after_tool_hook: Option<AfterToolHook>,
    memory_hook: Option<(MemoryHookConfig, MemoryProposalHook)>,
    context_compaction_hook: Option<ContextCompactionHook>,
    after_run_hook: Option<AfterRunHook>,
    control: Arc<StdMutex<ControlState>>,
    delegation_depth: u16,
}

impl AgentLoop<ModelClient> {
    /// Build an agent from the application's configuration and tool registry.
    pub fn from_config(config: &Config, tools: ToolRegistry) -> Result<Self, AgentError> {
        let client = ModelClient::new(
            &config.base_url,
            Duration::from_secs(config.request_timeout_secs),
        )?;
        let tools = tools.with_output_limit(config.max_tool_output_bytes);
        let agent = AgentLoop::new_with_policy(
            client,
            tools,
            config.model.clone(),
            config.api_key(),
            ChatOptions {
                temperature: config.temperature,
                enable_thinking: config.enable_thinking,
            },
            config.run_policy(),
        )?;
        Ok(match &config.system_prompt {
            Some(prompt) => agent.with_system_prompt(prompt.clone()),
            None => agent,
        })
    }

    /// Configure how the OpenAI-compatible provider selects advertised tools.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.client = self.client.with_tool_choice(tool_choice);
        self
    }
}

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub fn new(
        client: M,
        tools: ToolRegistry,
        model: impl Into<String>,
        api_key: Option<String>,
        options: ChatOptions,
        max_steps: usize,
    ) -> Result<Self, AgentError> {
        Self::new_with_policy(
            client,
            tools,
            model,
            api_key,
            options,
            RunPolicy {
                max_steps,
                ..RunPolicy::default()
            },
        )
    }

    /// Build an agent with explicit runtime limits and deadlines.
    pub fn new_with_policy(
        client: M,
        tools: ToolRegistry,
        model: impl Into<String>,
        api_key: Option<String>,
        options: ChatOptions,
        policy: RunPolicy,
    ) -> Result<Self, AgentError> {
        validate_policy(policy)?;

        Ok(Self {
            client,
            tools: tools.gateway(),
            model: model.into(),
            api_key,
            options,
            policy,
            allow_irreversible_tools: false,
            run_timeout: policy.run_timeout_secs.map(Duration::from_secs),
            session: None,
            runtime_ledger_file: None,
            runtime_recovery: false,
            system_prompt: None,
            skills: None,
            event_sink: None,
            before_tool_hook: None,
            after_tool_hook: None,
            memory_hook: None,
            context_compaction_hook: None,
            after_run_hook: None,
            control: Arc::new(StdMutex::new(ControlState::default())),
            delegation_depth: 0,
        })
    }

    /// Attach durable transcript storage to this agent.
    pub fn with_session(mut self, session: SessionStore) -> Self {
        self.session = Some(session);
        self
    }

    /// Persist runtime action events in the supplied JSONL ledger.
    pub fn with_runtime_ledger_file(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.runtime_ledger_file = Some(path.into());
        self
    }

    /// Allow tools classified as irreversible to pass runtime admission.
    ///
    /// This is intentionally opt-in. A shell can mutate files, start
    /// processes, or affect external systems, so callers should enable this
    /// only when the surrounding environment supplies the required approval
    /// or sandbox boundary.
    pub fn with_allow_irreversible_tools(mut self, allow: bool) -> Self {
        self.allow_irreversible_tools = allow;
        self
    }

    /// Resume the latest runtime run represented by the configured ledger.
    ///
    /// The default starts a fresh runtime run for each invocation. Recovery is
    /// opt-in because an unresolved external action must be reconciled before
    /// a new tool call is allowed to proceed.
    pub fn with_runtime_recovery(mut self) -> Self {
        self.runtime_recovery = true;
        self
    }

    /// Attach a stable, ephemeral system prompt to every model request.
    ///
    /// The prompt is inserted before session messages and is never appended to
    /// the durable transcript. Empty prompts are ignored so callers can pass
    /// optional configuration without creating blank system messages.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();
        self.system_prompt = (!prompt.trim().is_empty()).then_some(prompt);
        self
    }

    /// Attach skills whose metadata should be visible to the model.
    ///
    /// The catalog index is added as an ephemeral system message for model
    /// requests. It is not written to the durable session transcript, so a
    /// later run can rebuild it from the current skill roots.
    pub fn with_skills(mut self, skills: SkillCatalog) -> Self {
        if !skills.is_empty() {
            let max_output_bytes = self.tools.output_limit();
            self.tools
                .ensure_skill_reader(skills.clone(), max_output_bytes)
                .expect("the built-in read_skill tool must remain valid");
            self.skills = Some(skills);
        } else {
            self.skills = None;
        }
        self
    }

    /// Set the current delegation depth used in child-agent lineage metadata.
    pub fn with_delegation_depth(mut self, depth: u16) -> Self {
        self.delegation_depth = depth;
        self
    }

    /// Attach a sink for ephemeral runtime events.
    pub fn with_event_sink<S>(mut self, sink: S) -> Self
    where
        S: EventSink + 'static,
    {
        self.event_sink = Some(Arc::new(sink));
        self
    }

    /// Install an asynchronous hook that can block a tool before admission.
    pub fn with_before_tool_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(ToolHookContext) -> ToolHookFuture<BeforeToolDecision> + Send + Sync + 'static,
    {
        self.before_tool_hook = Some(Arc::new(hook));
        self
    }

    /// Install an asynchronous hook that can audit or replace a tool result.
    pub fn with_after_tool_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(ToolResultContext) -> ToolHookFuture<ToolResultOverride> + Send + Sync + 'static,
    {
        self.after_tool_hook = Some(Arc::new(hook));
        self
    }

    /// Install an asynchronous provider-agnostic memory proposal hook.
    ///
    /// The hook prepares a proposal only. The runtime validates its run,
    /// revision, expiry, candidate count, projection size, and token budget
    /// immediately before the model request. Accepted projections are added
    /// as ephemeral system context and are not written to the session.
    pub fn with_memory_hook<F>(mut self, config: MemoryHookConfig, hook: F) -> Self
    where
        F: Fn(MemoryHookRequest) -> MemoryHookFuture + Send + Sync + 'static,
    {
        self.memory_hook = Some((config, Arc::new(hook)));
        self
    }

    /// Install an asynchronous context compactor used only after a context
    /// budget is exceeded. The transformed messages stay ephemeral and are
    /// checked against the same message and byte limits before the request.
    pub fn with_context_compactor<F>(mut self, hook: F) -> Self
    where
        F: Fn(ContextCompactionRequest) -> ContextCompactionFuture + Send + Sync + 'static,
    {
        self.context_compaction_hook = Some(Arc::new(hook));
        self
    }

    /// Install an asynchronous hook that observes the completed run.
    ///
    /// The hook runs before the terminal `RunFinished` event is emitted. A
    /// hook error is returned to the caller as [`AgentError::AfterRunHook`],
    /// so persistence failures cannot disappear behind a successful model
    /// response.
    pub fn with_after_run_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(AfterRunRequest) -> AfterRunFuture + Send + Sync + 'static,
    {
        self.after_run_hook = Some(Arc::new(hook));
        self
    }

    /// Inspect unresolved runtime actions without starting an agent run.
    pub fn pending_runtime_actions(&self) -> Result<Vec<PendingRuntimeAction>, AgentError> {
        self.ensure_idle()?;
        let Some(path) = &self.runtime_ledger_file else {
            return Ok(Vec::new());
        };
        let ledger = JsonlLedger::open(path)
            .map_err(|error| AgentError::Runtime(format!("{}: {error}", path.display())))?;
        let ledger = Box::new(ledger) as Box<dyn EventLedger + Send>;
        let Some(runtime) = RuntimeCoordinator::recover_latest(
            runtime_policy(
                &self.tools.definitions(),
                self.policy.tool_timeout_secs,
                self.allow_irreversible_tools,
            ),
            ledger,
        )
        .map_err(|error| AgentError::Runtime(error.to_string()))?
        else {
            return Ok(Vec::new());
        };
        Ok(runtime
            .pending_actions()
            .iter()
            .map(|grant| pending_runtime_action(grant, &self.tools))
            .collect())
    }

    /// Close one unknown runtime action using caller-provided external evidence.
    pub fn reconcile_runtime_action(
        &self,
        action_id: Uuid,
        outcome: RuntimeReconcileOutcome,
    ) -> Result<(), AgentError> {
        self.ensure_idle()?;
        let Some(path) = &self.runtime_ledger_file else {
            return Err(AgentError::Runtime(
                "no runtime ledger is configured".to_string(),
            ));
        };
        let ledger = JsonlLedger::open(path)
            .map_err(|error| AgentError::Runtime(format!("{}: {error}", path.display())))?;
        let ledger = Box::new(ledger) as Box<dyn EventLedger + Send>;
        let mut runtime = RuntimeCoordinator::recover_latest(
            runtime_policy(
                &self.tools.definitions(),
                self.policy.tool_timeout_secs,
                self.allow_irreversible_tools,
            ),
            ledger,
        )
        .map_err(|error| AgentError::Runtime(error.to_string()))?
        .ok_or_else(|| AgentError::RuntimeActionNotFound { action_id })?;
        let grant = runtime
            .pending_actions()
            .iter()
            .find(|grant| grant.action_id.as_uuid() == action_id)
            .cloned()
            .ok_or(AgentError::RuntimeActionNotFound { action_id })?;
        let outcome = match outcome {
            RuntimeReconcileOutcome::Succeeded { output, evidence } => {
                ReconciliationOutcome::Succeeded { output, evidence }
            }
            RuntimeReconcileOutcome::Failed { error, evidence } => {
                ReconciliationOutcome::Failed { error, evidence }
            }
        };
        runtime
            .reconcile(grant, outcome, unix_timestamp_ms())
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
        Ok(())
    }

    /// Queue a user message for the next safe model boundary in this run.
    ///
    /// A message may be preloaded before the first run or queued while a run
    /// is active. It is persisted only when injected after the current tool
    /// batch reaches a terminal result; queued messages are drained FIFO.
    pub fn steer(&self, prompt: impl Into<String>) -> Result<(), AgentError> {
        self.enqueue_prompt("steering", prompt.into(), |control| &mut control.steering)
    }

    /// Queue one user message to run after the current assistant turn finishes.
    /// Follow-ups may be preloaded before the first run or queued while a run
    /// is active. They are consumed one at a time so a long queue cannot skip
    /// the model-step budget in a single boundary.
    pub fn follow_up(&self, prompt: impl Into<String>) -> Result<(), AgentError> {
        self.enqueue_prompt("follow-up", prompt.into(), |control| &mut control.follow_up)
    }

    fn enqueue_prompt(
        &self,
        kind: &'static str,
        prompt: String,
        queue: impl FnOnce(&mut ControlState) -> &mut VecDeque<String>,
    ) -> Result<(), AgentError> {
        if prompt.trim().is_empty() {
            return Err(AgentError::EmptyQueuedPrompt { kind });
        }
        if prompt.len() > MAX_QUEUED_PROMPT_BYTES {
            return Err(AgentError::QueuedPromptTooLarge {
                kind,
                limit: MAX_QUEUED_PROMPT_BYTES,
            });
        }
        let mut control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !control.active && control.has_run {
            return Err(AgentError::ControlQueueRequiresActiveRun { kind });
        }
        let queue = queue(&mut control);
        if queue.len() >= MAX_QUEUED_PROMPTS {
            return Err(AgentError::QueueFull {
                kind,
                limit: MAX_QUEUED_PROMPTS,
            });
        }
        queue.push_back(prompt);
        Ok(())
    }

    fn peek_steering(&self) -> Option<String> {
        self.control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .steering
            .front()
            .cloned()
    }

    fn ack_steering(&self) {
        let _ = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .steering
            .pop_front();
    }

    fn peek_follow_up(&self) -> Option<String> {
        self.control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .follow_up
            .front()
            .cloned()
    }

    fn ack_follow_up(&self) {
        let _ = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .follow_up
            .pop_front();
    }

    fn start_run(&self) -> Result<RunGuard, AgentError> {
        let mut control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if control.active {
            return Err(AgentError::RunBusy);
        }
        control.active = true;
        Ok(RunGuard {
            control: Arc::clone(&self.control),
        })
    }

    fn mark_run_started(&self) {
        self.control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .has_run = true;
    }

    fn ensure_idle(&self) -> Result<(), AgentError> {
        let control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if control.active {
            return Err(AgentError::RunBusy);
        }
        Ok(())
    }

    /// Run one prompt until the model returns an assistant message without tools.
    pub async fn run(&self, prompt: impl Into<String>) -> Result<RunResult, AgentError> {
        let cancellation = CancellationToken::new();
        self.run_with_deadline_and_context(prompt.into(), self.run_timeout, &cancellation, None)
            .await
    }

    /// Run one prompt using a caller-supplied execution identity.
    ///
    /// This is intended for orchestration and benchmark adapters that need to
    /// correlate model steps with an external operation. The supplied
    /// `run_id` and `turn_id` are preserved in lifecycle events and the fresh
    /// runtime coordinator. Runtime recovery is intentionally bypassed for
    /// this path: recovering an older ledger run would make the requested
    /// identity ambiguous. Configure reconciliation before starting a linked
    /// run when the ledger contains unresolved actions.
    pub async fn run_with_context(
        &self,
        prompt: impl Into<String>,
        context: RunContext,
    ) -> Result<RunResult, AgentError> {
        let cancellation = CancellationToken::new();
        self.run_with_deadline_and_context(
            prompt.into(),
            self.run_timeout,
            &cancellation,
            Some(context),
        )
        .await
    }

    /// Run one prompt with a deadline for model and tool execution.
    ///
    /// Session writes are deliberately outside the cancellable futures. If a
    /// tool is interrupted, synthetic tool results are persisted for all
    /// outstanding calls so a resumed transcript remains structurally valid.
    pub async fn run_with_timeout(
        &self,
        prompt: impl Into<String>,
        timeout: Duration,
    ) -> Result<RunResult, AgentError> {
        let cancellation = CancellationToken::new();
        self.run_with_deadline_and_context(prompt.into(), Some(timeout), &cancellation, None)
            .await
    }

    /// Run one prompt until completion, timeout, or an external cancellation.
    pub async fn run_with_cancellation(
        &self,
        prompt: impl Into<String>,
        cancellation: CancellationToken,
    ) -> Result<RunResult, AgentError> {
        self.run_with_deadline_and_context(prompt.into(), self.run_timeout, &cancellation, None)
            .await
    }

    async fn run_with_deadline_and_context(
        &self,
        prompt: String,
        timeout: Option<Duration>,
        cancellation: &CancellationToken,
        supplied_context: Option<RunContext>,
    ) -> Result<RunResult, AgentError> {
        let _run_guard = self.start_run()?;
        let deadline = timeout.map(|timeout| (tokio::time::Instant::now() + timeout, timeout));
        let externally_supplied_context = supplied_context.is_some();
        let mut context = supplied_context.unwrap_or_else(|| {
            RunContext::new(self.session.as_ref().map(SessionStore::session_id))
        });
        let session_mismatch = self.session.as_ref().and_then(|session| {
            (context.session_id != Some(session.session_id())).then_some(
                AgentError::SessionContextMismatch {
                    expected: session.session_id(),
                    provided: context.session_id,
                },
            )
        });
        let runtime = if prompt.trim().is_empty() {
            Err(AgentError::EmptyPrompt)
        } else if cancellation.is_cancelled() {
            Err(AgentError::RunCancelled)
        } else if let Some(error) = session_mismatch {
            Err(error)
        } else {
            self.mark_run_started();
            self.prepare_runtime(
                RunId::from_uuid(context.run_id),
                !externally_supplied_context,
            )
        };
        if !externally_supplied_context {
            if let Ok(runtime) = &runtime {
                context.run_id = runtime.run_id().as_uuid();
            }
        }
        self.emit(AgentEvent::RunStarted { context });

        let mut stats = RunStats::default();
        let mut transcript = Vec::new();
        let mut result = match runtime {
            Ok(runtime) => {
                self.run_inner(
                    prompt,
                    context,
                    &mut stats,
                    &mut transcript,
                    deadline,
                    cancellation,
                    runtime,
                )
                .await
            }
            Err(error) => Err(error),
        };
        let mut after_run_hook_error = None;
        if let Some(hook) = &self.after_run_hook {
            let outcome = match &result {
                Ok(_) => RunOutcome::Completed,
                Err(error) => run_outcome_for_error_with_runtime(error, stats.has_unknown_action),
            };
            let request = AfterRunRequest {
                context,
                transcript,
                steps: stats.steps,
                model_retries: stats.model_retries,
                outcome,
                result: result.as_ref().ok().cloned(),
                error: result.as_ref().err().map(ToString::to_string),
            };
            if let Err(error) = hook(request).await {
                if result.is_ok() {
                    result = Err(AgentError::AfterRunHook(error));
                } else {
                    after_run_hook_error = Some(error);
                }
            }
        }
        match &result {
            Ok(result) => self.emit(AgentEvent::RunFinished {
                context,
                steps: result.steps,
                outcome: RunOutcome::Completed,
                final_message: Some(result.final_message.clone()),
                error: None,
            }),
            Err(error) => self.emit(AgentEvent::RunFinished {
                context,
                steps: stats.steps,
                outcome: run_outcome_for_error_with_runtime(error, stats.has_unknown_action),
                final_message: None,
                error: Some(match after_run_hook_error {
                    Some(hook_error) => {
                        format!("{error}; after-run hook failed: {hook_error}")
                    }
                    None => error.to_string(),
                }),
            }),
        }
        result
    }

    fn prepare_runtime(
        &self,
        fresh_runtime_run_id: RunId,
        allow_recovery: bool,
    ) -> Result<RuntimeCoordinator<Box<dyn EventLedger + Send>>, AgentError> {
        let runtime_definitions = self.tools.definitions();
        let runtime_policy = runtime_policy(
            &runtime_definitions,
            self.policy.tool_timeout_secs,
            self.allow_irreversible_tools,
        );
        let (_, runtime): (RunId, RuntimeCoordinator<Box<dyn EventLedger + Send>>) = if let Some(
            path,
        ) =
            &self.runtime_ledger_file
        {
            let ledger = JsonlLedger::open(path)
                .map_err(|error| AgentError::Runtime(format!("{}: {error}", path.display())))?;
            let ledger = Box::new(ledger) as Box<dyn EventLedger + Send>;
            if allow_recovery && self.runtime_recovery {
                match RuntimeCoordinator::recover_latest(runtime_policy.clone(), ledger)
                    .map_err(|error| AgentError::Runtime(error.to_string()))?
                {
                    Some(runtime) if !runtime.pending_actions().is_empty() => {
                        return Err(AgentError::Runtime(
                            "runtime recovery found unresolved actions; reconcile them before resuming the agent"
                                .to_string(),
                        ));
                    }
                    Some(runtime) => (runtime.run_id(), runtime),
                    None => {
                        let ledger = JsonlLedger::open(path).map_err(|error| {
                            AgentError::Runtime(format!("{}: {error}", path.display()))
                        })?;
                        (
                            fresh_runtime_run_id,
                            RuntimeCoordinator::new(
                                fresh_runtime_run_id,
                                runtime_policy,
                                Box::new(ledger) as Box<dyn EventLedger + Send>,
                            ),
                        )
                    }
                }
            } else {
                if let Some(recovered) =
                    RuntimeCoordinator::recover_latest(runtime_policy.clone(), ledger)
                        .map_err(|error| AgentError::Runtime(error.to_string()))?
                {
                    if !recovered.pending_actions().is_empty() {
                        return Err(AgentError::Runtime(
                            "runtime ledger contains unresolved actions; use explicit runtime recovery or reconcile them before starting a new run"
                                .to_string(),
                        ));
                    }
                }
                let ledger = JsonlLedger::open(path)
                    .map_err(|error| AgentError::Runtime(format!("{}: {error}", path.display())))?;
                (
                    fresh_runtime_run_id,
                    RuntimeCoordinator::new(
                        fresh_runtime_run_id,
                        runtime_policy,
                        Box::new(ledger) as Box<dyn EventLedger + Send>,
                    ),
                )
            }
        } else {
            (
                fresh_runtime_run_id,
                RuntimeCoordinator::new(
                    fresh_runtime_run_id,
                    runtime_policy,
                    Box::new(InMemoryLedger::default()) as Box<dyn EventLedger + Send>,
                ),
            )
        };
        Ok(runtime)
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_inner(
        &self,
        prompt: String,
        context: RunContext,
        stats: &mut RunStats,
        transcript: &mut Vec<Message>,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
        mut runtime: RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    ) -> Result<RunResult, AgentError> {
        let definitions = self.tools.model_definitions();
        let runtime_run_id = runtime.run_id();
        let mut messages = match &self.session {
            Some(session) => session
                .load_repaired()
                .await?
                .into_iter()
                .map(|record| record.message)
                .collect(),
            None => Vec::new(),
        };
        if self.runtime_ledger_file.is_some() {
            validate_session_runtime_consistency(&messages, &runtime)?;
            apply_reconciled_tool_results(&mut messages, &runtime, self.tools.output_limit())?;
        }
        if let Some(system_prompt) = &self.system_prompt {
            messages.insert(0, Message::system(system_prompt.clone()));
        }
        if let Some(skills) = &self.skills {
            let skill_index = skills.format_for_system_prompt();
            if !skill_index.is_empty() {
                let system_end = messages
                    .iter()
                    .take_while(|message| message.role == "system")
                    .count();
                messages.insert(system_end, Message::system(skill_index));
            }
        }
        if cancellation.is_cancelled() {
            return Err(AgentError::RunCancelled);
        }
        let user_message = Message::user(prompt);
        messages.push(user_message.clone());
        transcript.push(user_message.clone());
        if let Some(session) = &self.session {
            session.append(&user_message).await?;
        }

        for step in 1..=self.policy.max_steps {
            self.inject_steering_messages(&mut messages, transcript)
                .await?;
            stats.steps = step;
            self.emit(AgentEvent::TurnStarted { context, step });
            self.emit(AgentEvent::ModelStarted {
                context,
                step,
                model: self.model.clone(),
            });
            let mut request_messages = self
                .fit_context(context, step, messages.clone(), deadline, cancellation)
                .await?;
            self.prepare_memory_context(
                context,
                step,
                runtime_run_id,
                &mut request_messages,
                &mut runtime,
                deadline,
                cancellation,
            )
            .await?;
            request_messages = self
                .fit_context(context, step, request_messages, deadline, cancellation)
                .await?;
            let mut model_retry_count: usize = 0;
            let mut assistant = loop {
                let attempt_emitted = Arc::new(AtomicBool::new(false));
                let delta_emitted = Arc::clone(&attempt_emitted);
                let mut emit_attempt_delta = |content: &str| {
                    delta_emitted.store(true, Ordering::Release);
                    self.emit(AgentEvent::AssistantDelta {
                        context,
                        step,
                        content: content.to_string(),
                    });
                };
                let reasoning_emitted = Arc::clone(&attempt_emitted);
                let mut emit_attempt_reasoning = |content: &str| {
                    reasoning_emitted.store(true, Ordering::Release);
                    self.emit(AgentEvent::ReasoningDelta {
                        context,
                        step,
                        content: content.to_string(),
                    });
                };
                let tool_call_emitted = Arc::clone(&attempt_emitted);
                let mut emit_attempt_tool_call = |delta: &ToolCallDelta| {
                    tool_call_emitted.store(true, Ordering::Release);
                    self.emit(AgentEvent::ToolCallDelta {
                        context,
                        step,
                        index: delta.index,
                        id: delta.id.clone(),
                        call_type: delta.call_type.clone(),
                        name: delta.name.clone(),
                        arguments: delta.arguments.clone(),
                    });
                };
                let model_request = self.client.stream_with_reasoning_and_tools(
                    ChatRequest {
                        model: &self.model,
                        api_key: self.api_key.as_deref(),
                        messages: &request_messages,
                        tools: &definitions,
                        options: self.options,
                    },
                    &mut emit_attempt_delta,
                    &mut emit_attempt_reasoning,
                    &mut emit_attempt_tool_call,
                );
                let result = match deadline {
                    Some((deadline, timeout)) => tokio::select! {
                        result = tokio::time::timeout_at(deadline, model_request) => {
                            result
                                .map_err(|_| timeout_error(timeout))
                                .and_then(|result| result.map_err(AgentError::Model))
                        }
                        _ = cancellation.cancelled() => Err(AgentError::RunCancelled),
                    },
                    None => tokio::select! {
                        result = model_request => result.map_err(AgentError::Model),
                        _ = cancellation.cancelled() => Err(AgentError::RunCancelled),
                    },
                };
                let attempt_number = model_retry_count.saturating_add(1);
                let attempt_had_output = attempt_emitted.load(Ordering::Acquire);
                match result {
                    Ok(assistant) => {
                        stats.record_model_usage(self.client.take_last_usage());
                        break assistant;
                    }
                    Err(AgentError::Model(error))
                        if error.is_retryable()
                            && model_retry_count < self.policy.max_model_retries =>
                    {
                        if attempt_had_output {
                            self.emit(AgentEvent::AssistantStreamReset {
                                context,
                                step,
                                attempt: attempt_number,
                                reason: error.to_string(),
                            });
                        }
                        model_retry_count += 1;
                        stats.model_retries = stats.model_retries.saturating_add(1);
                        let delay_ms = model_retry_delay_ms(
                            self.policy.model_retry_backoff_ms,
                            model_retry_count,
                        );
                        self.emit(AgentEvent::ModelRetry {
                            context,
                            step,
                            attempt: model_retry_count,
                            delay_ms,
                            error: error.to_string(),
                        });
                        let delay = Duration::from_millis(delay_ms);
                        match deadline {
                            Some((deadline, timeout)) => tokio::select! {
                                result = tokio::time::timeout_at(deadline, tokio::time::sleep(delay)) => {
                                    result.map_err(|_| timeout_error(timeout))?;
                                }
                                _ = cancellation.cancelled() => return Err(AgentError::RunCancelled),
                            },
                            None => tokio::select! {
                                _ = tokio::time::sleep(delay) => {}
                                _ = cancellation.cancelled() => return Err(AgentError::RunCancelled),
                            },
                        }
                    }
                    Err(error) => {
                        if attempt_had_output {
                            self.emit(AgentEvent::AssistantStreamReset {
                                context,
                                step,
                                attempt: attempt_number,
                                reason: error.to_string(),
                            });
                        }
                        return Err(error);
                    }
                }
            };
            let has_tool_calls = assistant
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty());
            let assistant_message = assistant.clone();
            if has_tool_calls {
                let tool_calls = assistant_message
                    .tool_calls
                    .as_deref()
                    .expect("non-empty tool calls must be present");
                let validation_error = if tool_calls.len() > self.policy.max_tool_calls_per_step {
                    Some(AgentError::TooManyToolCalls {
                        step,
                        count: tool_calls.len(),
                        limit: self.policy.max_tool_calls_per_step,
                    })
                } else {
                    validate_tool_call_batch(step, tool_calls).err()
                };
                if let Some(error) = validation_error {
                    // A malformed assistant batch is an ephemeral protocol error.
                    // Do not append it to the durable transcript: session loading
                    // requires every assistant tool call to have a valid, ordered
                    // result, and this batch has not reached execution yet.
                    self.emit(AgentEvent::AssistantMessage {
                        context,
                        step,
                        message: assistant_message.clone(),
                    });
                    self.emit(AgentEvent::TurnFinished {
                        context,
                        step,
                        message: assistant_message,
                        tool_results: Vec::new(),
                    });
                    return Err(error);
                }
            }
            messages.push(assistant_message.clone());
            transcript.push(assistant_message.clone());
            if let Some(session) = &self.session {
                session.append(&assistant_message).await?;
            }
            self.emit(AgentEvent::AssistantMessage {
                context,
                step,
                message: assistant_message.clone(),
            });
            let mut turn_tool_results = Vec::new();

            if assistant_message.stop_reason.as_deref() == Some("content_filter") {
                if has_tool_calls {
                    turn_tool_results = self
                        .append_unexecuted_tool_results(
                            context,
                            step,
                            assistant_message
                                .tool_calls
                                .as_deref()
                                .expect("tool calls should be present when the batch is non-empty"),
                            &mut messages,
                            transcript,
                            "model output was blocked by content filter before the tool call could execute",
                            "rejected",
                        )
                        .await?;
                }
                self.emit(AgentEvent::TurnFinished {
                    context,
                    step,
                    message: assistant_message,
                    tool_results: turn_tool_results,
                });
                return Err(AgentError::ModelTerminated {
                    reason: "content_filter".to_string(),
                });
            }

            if matches!(
                assistant_message.stop_reason.as_deref(),
                Some("error" | "aborted")
            ) {
                let reason = assistant_message
                    .stop_reason
                    .clone()
                    .expect("matched terminal model reason must be present");
                if has_tool_calls {
                    turn_tool_results = self
                        .append_unexecuted_tool_results(
                            context,
                            step,
                            assistant_message
                                .tool_calls
                                .as_deref()
                                .expect("tool calls should be present when the batch is non-empty"),
                            &mut messages,
                            transcript,
                            &format!("model terminated before tool execution: {reason}"),
                            &reason,
                        )
                        .await?;
                }
                self.emit(AgentEvent::TurnFinished {
                    context,
                    step,
                    message: assistant_message,
                    tool_results: turn_tool_results,
                });
                return Err(AgentError::ModelTerminated { reason });
            }

            if !has_tool_calls {
                if assistant_message.stop_reason.as_deref() == Some("length") {
                    self.emit(AgentEvent::TurnFinished {
                        context,
                        step,
                        message: assistant_message,
                        tool_results: turn_tool_results,
                    });
                    return Err(AgentError::ModelTerminated {
                        reason: "length".to_string(),
                    });
                }
                if step < self.policy.max_steps {
                    if self
                        .inject_steering_messages(&mut messages, transcript)
                        .await?
                    {
                        self.emit(AgentEvent::TurnFinished {
                            context,
                            step,
                            message: assistant_message.clone(),
                            tool_results: turn_tool_results,
                        });
                        continue;
                    }
                    if self.peek_steering().is_some() {
                        self.emit(AgentEvent::TurnFinished {
                            context,
                            step,
                            message: assistant_message.clone(),
                            tool_results: turn_tool_results,
                        });
                        continue;
                    }
                    if self
                        .inject_follow_up_message(&mut messages, transcript)
                        .await?
                    {
                        self.emit(AgentEvent::TurnFinished {
                            context,
                            step,
                            message: assistant_message.clone(),
                            tool_results: turn_tool_results,
                        });
                        continue;
                    }
                }
                self.emit(AgentEvent::TurnFinished {
                    context,
                    step,
                    message: assistant_message.clone(),
                    tool_results: turn_tool_results,
                });
                let (input_tokens, output_tokens, total_tokens) = stats.model_usage();
                return Ok(RunResult {
                    messages,
                    final_message: assistant,
                    steps: step,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                });
            }

            let tool_calls = assistant.tool_calls.take().unwrap_or_default();

            if assistant_message.stop_reason.as_deref() == Some("length") {
                let truncated_results = self
                    .append_unexecuted_tool_results(
                        context,
                        step,
                        &tool_calls,
                        &mut messages,
                        transcript,
                        "model output was truncated before the tool call could execute",
                        "truncated",
                    )
                    .await?;
                self.emit(AgentEvent::TurnFinished {
                    context,
                    step,
                    message: assistant_message,
                    tool_results: truncated_results,
                });
                if step == self.policy.max_steps {
                    return Err(AgentError::MaxStepsExceeded {
                        max_steps: self.policy.max_steps,
                    });
                }
                continue;
            }

            if step == self.policy.max_steps {
                let pending_results = self
                    .append_unexecuted_tool_results(
                        context,
                        step,
                        &tool_calls,
                        &mut messages,
                        transcript,
                        "agent reached its maximum step count before executing the tool call",
                        "max_steps",
                    )
                    .await?;
                self.emit(AgentEvent::TurnFinished {
                    context,
                    step,
                    message: assistant_message,
                    tool_results: pending_results,
                });
                return Err(AgentError::MaxStepsExceeded {
                    max_steps: self.policy.max_steps,
                });
            }

            if let Some(outcome) = self
                .try_run_parallel_tool_batch(
                    context,
                    step,
                    &tool_calls,
                    &mut messages,
                    transcript,
                    &mut turn_tool_results,
                    stats,
                    deadline,
                    cancellation,
                    runtime_run_id,
                    &mut runtime,
                )
                .await?
            {
                self.emit(AgentEvent::TurnFinished {
                    context,
                    step,
                    message: assistant_message,
                    tool_results: turn_tool_results,
                });
                if let Some(error) = outcome.terminal_error {
                    return Err(error);
                }
                if outcome.all_tools_terminate {
                    let (input_tokens, output_tokens, total_tokens) = stats.model_usage();
                    return Ok(RunResult {
                        messages,
                        final_message: outcome
                            .last_tool_message
                            .expect("a non-empty parallel batch produces a tool message"),
                        steps: step,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                    });
                }
                continue;
            }

            let mut preflight = tool_calls
                .iter()
                .map(|tool_call| preflight_model_tool_call(&self.tools, tool_call))
                .collect::<Vec<_>>()
                .into_iter();
            let mut all_tools_terminate = true;
            let mut last_tool_message = None;
            for (tool_index, tool_call) in tool_calls.iter().enumerate() {
                let (arguments, mut prepared_call, validation_error) = preflight
                    .next()
                    .expect("one preflight result should exist for every tool call");
                let tool_started_at = Instant::now();
                self.emit(AgentEvent::ToolStarted {
                    context,
                    step,
                    call: tool_call.clone(),
                });
                let tool_timeout = Duration::from_secs(self.policy.tool_timeout_secs);
                let now_ms = unix_timestamp_ms();
                let mut stop_after_tool: Option<(AgentError, String)> = None;
                let mut runtime_action: Option<(Uuid, Uuid)> = None;
                // Keep a known terminal action open until the final result
                // has passed the after-tool hook and output normalization.
                // Unknown outcomes are committed immediately above because
                // they must remain pending for external reconciliation.
                let mut pending_runtime_terminal: Option<CommitGrant> = None;
                let mut tool_result = if let Some(reason) = validation_error {
                    ToolResult::error(reason)
                } else {
                    let before_decision = match &self.before_tool_hook {
                        Some(hook) => {
                            let hook_result = await_tool_hook(
                                hook(ToolHookContext {
                                    run: context,
                                    step,
                                    call: tool_call.clone(),
                                    arguments: arguments.clone(),
                                }),
                                deadline,
                                cancellation,
                            )
                            .await;
                            match hook_result {
                                ToolHookWait::Completed(result) => {
                                    result.unwrap_or_else(|error| BeforeToolDecision::Block {
                                        reason: format!("before_tool_call hook failed: {error}"),
                                    })
                                }
                                ToolHookWait::TimedOut(run_timeout) => {
                                    let reason = format!(
                                        "before_tool_call hook timed out after {} milliseconds",
                                        timeout_millis(run_timeout)
                                    );
                                    self.append_cancelled_tool_results(
                                        context,
                                        step,
                                        &tool_calls,
                                        tool_index,
                                        &mut messages,
                                        transcript,
                                        None,
                                        &reason,
                                    )
                                    .await?;
                                    return Err(timeout_error(run_timeout));
                                }
                                ToolHookWait::Cancelled => {
                                    let reason = "before_tool_call hook cancelled by the caller";
                                    self.append_cancelled_tool_results(
                                        context,
                                        step,
                                        &tool_calls,
                                        tool_index,
                                        &mut messages,
                                        transcript,
                                        None,
                                        reason,
                                    )
                                    .await?;
                                    return Err(AgentError::RunCancelled);
                                }
                            }
                        }
                        None => BeforeToolDecision::Allow,
                    };
                    if cancellation.is_cancelled() {
                        let reason = "Tool execution cancelled by the caller";
                        self.append_cancelled_tool_results(
                            context,
                            step,
                            &tool_calls,
                            tool_index,
                            &mut messages,
                            transcript,
                            None,
                            reason,
                        )
                        .await?;
                        return Err(AgentError::RunCancelled);
                    }
                    match before_decision {
                        BeforeToolDecision::Block { reason } => ToolResult::error(format!(
                            "Tool blocked: {reason}"
                        ))
                        .with_details(serde_json::json!({
                            "status": "rejected",
                            "executed": false,
                        })),
                        BeforeToolDecision::Terminate { reason } => {
                            let reason = format!("Tool execution terminated by policy: {reason}");
                            stop_after_tool = Some((
                                AgentError::ToolTerminated {
                                    reason: reason.clone(),
                                },
                                reason.clone(),
                            ));
                            ToolResult::error(reason)
                                .with_details(serde_json::json!({
                                    "status": "rejected",
                                    "executed": false,
                                }))
                                .terminating()
                        }
                        BeforeToolDecision::Allow => {
                            let request = AdmissionRequest {
                                proposal: ActionProposal {
                                    run_id: runtime_run_id,
                                    action_id: ActionId::new(),
                                    base_revision: runtime.current_revision(),
                                    tool_name: tool_call.function.name.clone(),
                                    arguments: arguments.clone(),
                                    arguments_digest: canonical_json_digest(&arguments),
                                    risk: self
                                        .tools
                                        .risk_class(&tool_call.function.name)
                                        .unwrap_or(RiskClass::Unknown),
                                    idempotency_key: format!(
                                        "{}:{}:{}:{}:{}",
                                        runtime_run_id.as_uuid(),
                                        context.turn_id,
                                        step,
                                        tool_index,
                                        tool_call.id
                                    ),
                                    created_at_ms: now_ms,
                                    expires_at_ms: Some(
                                        now_ms.saturating_add(timeout_millis(tool_timeout)),
                                    ),
                                },
                                resource_scope: ResourceScope::new([format!(
                                    "tool:{}",
                                    tool_call.function.name
                                )]),
                                budget: Budget {
                                    wall_time_ms: Some(timeout_millis(tool_timeout)),
                                    tokens: None,
                                    tool_calls: Some(1),
                                },
                            };
                            match runtime.admit(request, now_ms) {
                                Err(error) => {
                                    ToolResult::error(format!("Tool admission failed: {error}"))
                                        .with_details(json!({
                                            "status": "rejected",
                                            "executed": false,
                                        }))
                                }
                                Ok(grant) => {
                                    runtime_action =
                                        Some((grant.action_id.as_uuid(), grant.grant_id));
                                    let prepared = prepared_call
                                        .take()
                                        .expect("a call reaching execution must be prepared");
                                    let tool_context =
                                        ToolExecutionContext::new(cancellation.clone())
                                            .with_output_limit(self.tools.output_limit())
                                            .with_parent_identity(
                                                context.run_id,
                                                context.turn_id,
                                                step,
                                                tool_call.id.clone(),
                                            )
                                            .with_delegation_depth(self.delegation_depth)
                                            .with_update_sink({
                                                let event_sink = self.event_sink.clone();
                                                let call_id = tool_call.id.clone();
                                                let tool_name = tool_call.function.name.clone();
                                                move |update| {
                                                    if let Some(sink) = &event_sink {
                                                        sink.emit(AgentEvent::ToolUpdate {
                                                            context,
                                                            step,
                                                            call_id: call_id.clone(),
                                                            tool_name: tool_name.clone(),
                                                            content: update.content,
                                                            details: update.details,
                                                        });
                                                    }
                                                }
                                            });
                                    let execution = match deadline {
                                        Some((deadline, run_timeout)) => tokio::select! {
                                            result = tokio::time::timeout_at(
                                                deadline,
                                                tokio::time::timeout(
                                                    tool_timeout,
                                                    self.tools
                                                        .execute_prepared_result_with_context(
                                                            prepared,
                                                            tool_context.clone(),
                                                        ),
                                                ),
                                            ) => match result {
                                                Ok(result) => result,
                                                Err(_) => {
                                                    let reason = format!(
                                                        "Tool execution cancelled because the agent run timed out after {} milliseconds",
                                                        timeout_millis(run_timeout)
                                                    );
                                                    let runtime_identity =
                                                        (grant.action_id.as_uuid(), grant.grant_id);
                                                    runtime
                                                        .mark_unknown(grant, reason.clone(), unix_timestamp_ms())
                                                        .map_err(|error| AgentError::Runtime(error.to_string()))?;
                                                    stats.has_unknown_action = true;
                                                    self.append_cancelled_tool_results(
                                                        context,
                                                        step,
                                                        &tool_calls,
                                                        tool_index,
                                                        &mut messages,
                                                        transcript,
                                                        Some(runtime_identity),
                                                        &reason,
                                                    )
                                                    .await?;
                                                    return Err(timeout_error(run_timeout));
                                                }
                                            },
                                            _ = cancellation.cancelled() => {
                                                let reason = "Tool execution cancelled by the caller";
                                                let runtime_identity =
                                                    (grant.action_id.as_uuid(), grant.grant_id);
                                                runtime
                                                    .mark_unknown(grant, reason, unix_timestamp_ms())
                                                    .map_err(|error| AgentError::Runtime(error.to_string()))?;
                                                stats.has_unknown_action = true;
                                                self.append_cancelled_tool_results(
                                                    context,
                                                    step,
                                                    &tool_calls,
                                                    tool_index,
                                                    &mut messages,
                                                    transcript,
                                                    Some(runtime_identity),
                                                    reason,
                                                )
                                                .await?;
                                                return Err(AgentError::RunCancelled);
                                            }
                                        },
                                        None => tokio::select! {
                                            result = tokio::time::timeout(
                                                tool_timeout,
                                                self.tools
                                                    .execute_prepared_result_with_context(
                                                        prepared,
                                                        tool_context,
                                                    ),
                                            ) => result,
                                            _ = cancellation.cancelled() => {
                                                let reason = "Tool execution cancelled by the caller";
                                                let runtime_identity =
                                                    (grant.action_id.as_uuid(), grant.grant_id);
                                                runtime
                                                    .mark_unknown(grant, reason, unix_timestamp_ms())
                                                    .map_err(|error| AgentError::Runtime(error.to_string()))?;
                                                stats.has_unknown_action = true;
                                                self.append_cancelled_tool_results(
                                                    context,
                                                    step,
                                                    &tool_calls,
                                                    tool_index,
                                                    &mut messages,
                                                    transcript,
                                                    Some(runtime_identity),
                                                    reason,
                                                )
                                                .await?;
                                                return Err(AgentError::RunCancelled);
                                            }
                                        },
                                    };

                                    match execution {
                                        Ok(Ok(output)) => {
                                            if tool_result_has_unknown_outcome(&output) {
                                                let reason = format!(
                                                    "Tool execution reported an uncertain outcome: {}",
                                                    output.content
                                                );
                                                let action_id = grant.action_id.as_uuid();
                                                let tool_name = grant.tool_name.clone();
                                                let mut details = output.details.clone();
                                                if let Value::Object(fields) = &mut details {
                                                    fields.insert(
                                                        "status".to_string(),
                                                        Value::String("unknown".to_string()),
                                                    );
                                                    fields.insert(
                                                        "executed".to_string(),
                                                        Value::Bool(true),
                                                    );
                                                } else {
                                                    details = serde_json::json!({
                                                        "status": "unknown",
                                                        "executed": true,
                                                    });
                                                }
                                                runtime
                                                    .mark_unknown(
                                                        grant,
                                                        reason.clone(),
                                                        unix_timestamp_ms(),
                                                    )
                                                    .map_err(|error| {
                                                        AgentError::Runtime(error.to_string())
                                                    })?;
                                                stats.has_unknown_action = true;
                                                stop_after_tool = Some((
                                                    AgentError::RuntimeActionUnknown {
                                                        action_id,
                                                        tool_name,
                                                        reason: reason.clone(),
                                                    },
                                                    reason.clone(),
                                                ));
                                                ToolResult {
                                                    content: reason,
                                                    details,
                                                    is_error: true,
                                                    terminate: false,
                                                }
                                            } else {
                                                pending_runtime_terminal = Some(grant);
                                                output
                                            }
                                        }
                                        Ok(Err(error)) => {
                                            pending_runtime_terminal = Some(grant);
                                            ToolResult::error(format!(
                                                "Tool execution failed: {error}"
                                            ))
                                        }
                                        Err(_) => {
                                            let reason = format!(
                                                "Tool execution timed out after {} seconds; outcome is unknown",
                                                tool_timeout.as_secs()
                                            );
                                            let action_id = grant.action_id.as_uuid();
                                            let tool_name = grant.tool_name.clone();
                                            runtime
                                                .mark_unknown(
                                                    grant,
                                                    reason.clone(),
                                                    unix_timestamp_ms(),
                                                )
                                                .map_err(|error| {
                                                    AgentError::Runtime(error.to_string())
                                                })?;
                                            stats.has_unknown_action = true;
                                            stop_after_tool = Some((
                                                AgentError::RuntimeActionUnknown {
                                                    action_id,
                                                    tool_name,
                                                    reason: reason.clone(),
                                                },
                                                reason.clone(),
                                            ));
                                            ToolResult {
                                                content: reason,
                                                details: serde_json::json!({
                                                    "status": "unknown",
                                                    "executed": true,
                                                }),
                                                is_error: true,
                                                terminate: false,
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                };
                let original_content = tool_result.content.clone();
                let mut stop_after_hook: Option<(AgentError, String)> = None;
                let (content, is_error) = if stop_after_tool.is_some() {
                    (tool_result.content.clone(), tool_result.is_error)
                } else {
                    match &self.after_tool_hook {
                        Some(hook) => {
                            let hook_result = await_tool_hook(
                                hook(ToolResultContext {
                                    run: context,
                                    step,
                                    call: tool_call.clone(),
                                    arguments: arguments.clone(),
                                    content: tool_result.content.clone(),
                                    is_error: tool_result.is_error,
                                    details: tool_result.details.clone(),
                                    terminate: tool_result.terminate,
                                }),
                                deadline,
                                cancellation,
                            )
                            .await;
                            match hook_result {
                                ToolHookWait::Completed(Ok(override_result)) => {
                                    if let Some(details) = override_result.details {
                                        tool_result.details = details;
                                    }
                                    if let Some(terminate) = override_result.terminate {
                                        tool_result.terminate = terminate;
                                    }
                                    (
                                        override_result.content.unwrap_or(original_content),
                                        override_result.is_error.unwrap_or(tool_result.is_error),
                                    )
                                }
                                ToolHookWait::Completed(Err(error)) => {
                                    (format!("after_tool_call hook failed: {error}"), true)
                                }
                                ToolHookWait::TimedOut(run_timeout) => {
                                    let reason = format!(
                                        "after_tool_call hook timed out after {} milliseconds",
                                        timeout_millis(run_timeout)
                                    );
                                    stop_after_hook =
                                        Some((timeout_error(run_timeout), reason.clone()));
                                    (reason, true)
                                }
                                ToolHookWait::Cancelled => {
                                    let reason =
                                        "after_tool_call hook cancelled by the caller".to_string();
                                    stop_after_hook =
                                        Some((AgentError::RunCancelled, reason.clone()));
                                    (reason, true)
                                }
                            }
                        }
                        None => (tool_result.content.clone(), tool_result.is_error),
                    }
                };
                tool_result.content = content;
                tool_result.is_error = is_error;
                tool_result = self.tools.normalize_result(tool_result);
                let content = tool_result.content.clone();
                if let Some(grant) = pending_runtime_terminal.take() {
                    if tool_result.is_error {
                        runtime
                            .fail(grant, content.clone(), unix_timestamp_ms())
                            .map_err(|error| AgentError::Runtime(error.to_string()))?;
                    } else {
                        let output = serde_json::to_value(&tool_result).map_err(|error| {
                            AgentError::Runtime(format!(
                                "failed to serialize normalized tool result: {error}"
                            ))
                        })?;
                        runtime
                            .complete(grant, output, unix_timestamp_ms())
                            .map_err(|error| AgentError::Runtime(error.to_string()))?;
                    }
                }
                self.emit(AgentEvent::ToolFinished {
                    context,
                    step,
                    call_id: tool_call.id.clone(),
                    tool_name: tool_call.function.name.clone(),
                    content: content.clone(),
                    is_error,
                    details: tool_result.details.clone(),
                    terminate: tool_result.terminate,
                    duration_ms: elapsed_millis(tool_started_at),
                    status: tool_result_status(&tool_result),
                    action_id: runtime_action.map(|(action_id, _)| action_id),
                    grant_id: runtime_action.map(|(_, grant_id)| grant_id),
                });
                let metadata = runtime_action
                    .map(|(action_id, grant_id)| {
                        ToolResultMetadata::new(
                            tool_result.details.clone(),
                            tool_result.is_error,
                            tool_result.terminate,
                        )
                        .with_runtime_identity(action_id, grant_id)
                    })
                    .unwrap_or_else(|| {
                        ToolResultMetadata::new(
                            tool_result.details.clone(),
                            tool_result.is_error,
                            tool_result.terminate,
                        )
                    });
                let tool_message =
                    Message::tool_with_result(tool_call.id.clone(), content, metadata);
                messages.push(tool_message.clone());
                transcript.push(tool_message.clone());
                turn_tool_results.push(tool_message.clone());
                if let Some(session) = &self.session {
                    session.append(&tool_message).await?;
                }
                if !tool_result.terminate {
                    all_tools_terminate = false;
                }
                last_tool_message = Some(tool_message);
                if let Some((error, reason)) = stop_after_tool.or(stop_after_hook) {
                    self.append_cancelled_tool_results(
                        context,
                        step,
                        &tool_calls,
                        tool_index + 1,
                        &mut messages,
                        transcript,
                        None,
                        &reason,
                    )
                    .await?;
                    self.emit(AgentEvent::TurnFinished {
                        context,
                        step,
                        message: assistant_message,
                        tool_results: turn_tool_results,
                    });
                    return Err(error);
                }
            }
            if all_tools_terminate {
                self.emit(AgentEvent::TurnFinished {
                    context,
                    step,
                    message: assistant_message,
                    tool_results: turn_tool_results,
                });
                let (input_tokens, output_tokens, total_tokens) = stats.model_usage();
                return Ok(RunResult {
                    messages,
                    final_message: last_tool_message
                        .expect("a non-empty tool batch produces a tool message"),
                    steps: step,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                });
            }
            self.emit(AgentEvent::TurnFinished {
                context,
                step,
                message: assistant_message,
                tool_results: turn_tool_results,
            });
        }

        Err(AgentError::MaxStepsExceeded {
            max_steps: self.policy.max_steps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_run_parallel_tool_batch(
        &self,
        context: RunContext,
        step: usize,
        tool_calls: &[ToolCall],
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
        turn_tool_results: &mut Vec<Message>,
        stats: &mut RunStats,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
        runtime_run_id: RunId,
        runtime: &mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
    ) -> Result<Option<ParallelBatchOutcome>, AgentError> {
        if tool_calls.len() < 2
            || self.policy.max_parallel_tool_calls < 2
            || self.before_tool_hook.is_some()
            || self.after_tool_hook.is_some()
        {
            return Ok(None);
        }

        let mut prepared_calls = Vec::with_capacity(tool_calls.len());
        for tool_call in tool_calls {
            let (arguments, prepared, _validation_error) =
                preflight_model_tool_call(&self.tools, tool_call);
            let Some(prepared) = prepared else {
                return Ok(None);
            };
            if self.tools.concurrency(&tool_call.function.name) != Some(ToolConcurrency::Parallel)
                || self.tools.replay_policy(&tool_call.function.name)
                    != Some(ToolReplayPolicy::Safe)
                || self.tools.risk_class(&tool_call.function.name) != Some(RiskClass::ReadOnly)
            {
                return Ok(None);
            }
            prepared_calls.push(Some((arguments, prepared)));
        }

        let tool_timeout = Duration::from_secs(self.policy.tool_timeout_secs);
        let now_ms = unix_timestamp_ms();
        let requests = prepared_calls
            .iter()
            .enumerate()
            .map(|(tool_index, prepared)| {
                let (arguments, _) = prepared
                    .as_ref()
                    .expect("parallel preflight entries remain available");
                AdmissionRequest {
                    proposal: ActionProposal {
                        run_id: runtime_run_id,
                        action_id: ActionId::new(),
                        base_revision: runtime.current_revision(),
                        tool_name: tool_calls[tool_index].function.name.clone(),
                        arguments: arguments.clone(),
                        arguments_digest: canonical_json_digest(arguments),
                        risk: RiskClass::ReadOnly,
                        idempotency_key: format!(
                            "{}:{}:{}:{}:{}",
                            runtime_run_id.as_uuid(),
                            context.turn_id,
                            step,
                            tool_index,
                            tool_calls[tool_index].id
                        ),
                        created_at_ms: now_ms,
                        expires_at_ms: Some(now_ms.saturating_add(timeout_millis(tool_timeout))),
                    },
                    resource_scope: ResourceScope::new([format!(
                        "tool:{}",
                        tool_calls[tool_index].function.name
                    )]),
                    budget: Budget {
                        wall_time_ms: Some(timeout_millis(tool_timeout)),
                        tokens: None,
                        tool_calls: Some(1),
                    },
                }
            })
            .collect::<Vec<_>>();
        let receipt = match runtime.admit_batch(requests, now_ms) {
            Ok(receipt) => receipt,
            Err(_) => return Ok(None),
        };

        let mut all_tools_terminate = true;
        let mut last_tool_message = None;
        let mut terminal_error = None;
        let max_parallel = self.policy.max_parallel_tool_calls;
        let mut chunk_start = 0;

        while chunk_start < tool_calls.len() {
            let chunk_end = (chunk_start + max_parallel).min(tool_calls.len());
            let mut tasks = tokio::task::JoinSet::new();
            for index in chunk_start..chunk_end {
                let call = &tool_calls[index];
                self.emit(AgentEvent::ToolStarted {
                    context,
                    step,
                    call: call.clone(),
                });
                let (_, prepared) = prepared_calls[index]
                    .take()
                    .expect("parallel tool preparation should be consumed once");
                let event_sink = self.event_sink.clone();
                let call_id = call.id.clone();
                let tool_name = call.function.name.clone();
                let tool_context = ToolExecutionContext::new(cancellation.clone())
                    .with_output_limit(self.tools.output_limit())
                    .with_parent_identity(context.run_id, context.turn_id, step, call.id.clone())
                    .with_delegation_depth(self.delegation_depth)
                    .with_update_sink(move |update| {
                        if let Some(sink) = &event_sink {
                            sink.emit(AgentEvent::ToolUpdate {
                                context,
                                step,
                                call_id: call_id.clone(),
                                tool_name: tool_name.clone(),
                                content: update.content,
                                details: update.details,
                            });
                        }
                    });
                let tools = self.tools.clone();
                let grant = receipt.grants[index].clone();
                let cancellation = cancellation.clone();
                let started_at = Instant::now();
                tasks.spawn(async move {
                    let execution = execute_parallel_tool(
                        tools,
                        prepared,
                        tool_context,
                        tool_timeout,
                        deadline,
                        cancellation,
                    )
                    .await;
                    (index, grant, started_at, execution)
                });
            }

            let mut completed = Vec::with_capacity(chunk_end - chunk_start);
            while let Some(joined) = tasks.join_next().await {
                completed.push(joined.map_err(|error| {
                    AgentError::Runtime(format!("parallel tool worker failed: {error}"))
                })?);
            }
            completed.sort_by_key(|(index, _, _, _)| *index);

            for (index, grant, started_at, execution) in completed {
                let runtime_action = Some((grant.action_id.as_uuid(), grant.grant_id));
                let mut terminal_runtime = true;
                let mut unknown_reason = None;
                let mut stop_error = None;
                let mut tool_result = match execution {
                    ParallelExecutionOutcome::Completed(Ok(output)) => {
                        if tool_result_has_unknown_outcome(&output) {
                            let reason = format!(
                                "Tool execution reported an uncertain outcome: {}",
                                output.content
                            );
                            unknown_reason = Some(reason.clone());
                            stop_error = Some(AgentError::RuntimeActionUnknown {
                                action_id: grant.action_id.as_uuid(),
                                tool_name: grant.tool_name.clone(),
                                reason: reason.clone(),
                            });
                            let mut details = output.details.clone();
                            if let Value::Object(fields) = &mut details {
                                fields.insert(
                                    "status".to_string(),
                                    Value::String("unknown".to_string()),
                                );
                                fields.insert("executed".to_string(), Value::Bool(true));
                            } else {
                                details = json!({"status": "unknown", "executed": true});
                            }
                            ToolResult {
                                content: reason,
                                details,
                                is_error: true,
                                terminate: false,
                            }
                        } else {
                            output
                        }
                    }
                    ParallelExecutionOutcome::Completed(Err(error)) => {
                        ToolResult::error(format!("Tool execution failed: {error}"))
                    }
                    ParallelExecutionOutcome::ToolTimedOut => {
                        let reason = format!(
                            "Tool execution timed out after {} seconds; outcome is unknown",
                            tool_timeout.as_secs()
                        );
                        unknown_reason = Some(reason.clone());
                        stop_error = Some(AgentError::RuntimeActionUnknown {
                            action_id: grant.action_id.as_uuid(),
                            tool_name: grant.tool_name.clone(),
                            reason: reason.clone(),
                        });
                        ToolResult::error(reason).with_details(json!({
                            "status": "unknown",
                            "executed": true,
                        }))
                    }
                    ParallelExecutionOutcome::RunTimedOut(run_timeout) => {
                        let reason = format!(
                            "Tool execution cancelled because the agent run timed out after {} milliseconds",
                            timeout_millis(run_timeout)
                        );
                        unknown_reason = Some(reason.clone());
                        stop_error = Some(timeout_error(run_timeout));
                        ToolResult::error(reason).with_details(json!({
                            "status": "unknown",
                            "executed": true,
                        }))
                    }
                    ParallelExecutionOutcome::Cancelled => {
                        let reason = "Tool execution cancelled by the caller".to_string();
                        unknown_reason = Some(reason.clone());
                        stop_error = Some(AgentError::RunCancelled);
                        ToolResult::error(reason).with_details(json!({
                            "status": "unknown",
                            "executed": true,
                        }))
                    }
                };

                if let Some(reason) = unknown_reason {
                    runtime
                        .mark_unknown(grant.clone(), reason, unix_timestamp_ms())
                        .map_err(|error| AgentError::Runtime(error.to_string()))?;
                    stats.has_unknown_action = true;
                    terminal_runtime = false;
                }
                tool_result = self.tools.normalize_result(tool_result);
                if terminal_runtime {
                    let result = if tool_result.is_error {
                        runtime.fail(
                            grant.clone(),
                            tool_result.content.clone(),
                            unix_timestamp_ms(),
                        )
                    } else {
                        let output = serde_json::to_value(&tool_result).map_err(|error| {
                            AgentError::Runtime(format!(
                                "failed to serialize normalized tool result: {error}"
                            ))
                        })?;
                        runtime.complete(grant.clone(), output, unix_timestamp_ms())
                    };
                    result.map_err(|error| AgentError::Runtime(error.to_string()))?;
                }

                let content = tool_result.content.clone();
                self.emit(AgentEvent::ToolFinished {
                    context,
                    step,
                    call_id: tool_calls[index].id.clone(),
                    tool_name: tool_calls[index].function.name.clone(),
                    content: content.clone(),
                    is_error: tool_result.is_error,
                    details: tool_result.details.clone(),
                    terminate: tool_result.terminate,
                    duration_ms: elapsed_millis(started_at),
                    status: tool_result_status(&tool_result),
                    action_id: runtime_action.map(|(action_id, _)| action_id),
                    grant_id: runtime_action.map(|(_, grant_id)| grant_id),
                });
                let metadata = ToolResultMetadata::new(
                    tool_result.details.clone(),
                    tool_result.is_error,
                    tool_result.terminate,
                )
                .with_runtime_identity(grant.action_id.as_uuid(), grant.grant_id);
                let tool_message =
                    Message::tool_with_result(tool_calls[index].id.clone(), content, metadata);
                messages.push(tool_message.clone());
                transcript.push(tool_message.clone());
                if let Some(session) = &self.session {
                    session.append(&tool_message).await?;
                }
                turn_tool_results.push(tool_message.clone());
                last_tool_message = Some(tool_message);
                if !tool_result.terminate {
                    all_tools_terminate = false;
                }
                if terminal_error.is_none() {
                    terminal_error = stop_error;
                }
            }

            chunk_start = chunk_end;
            if terminal_error.is_some() {
                if chunk_start < tool_calls.len() {
                    let reason = "parallel tool batch stopped before remaining calls executed";
                    for grant in receipt.grants.iter().skip(chunk_start) {
                        runtime
                            .fail(grant.clone(), reason, unix_timestamp_ms())
                            .map_err(|error| AgentError::Runtime(error.to_string()))?;
                    }
                    let pending = self
                        .append_unexecuted_tool_results(
                            context,
                            step,
                            &tool_calls[chunk_start..],
                            messages,
                            transcript,
                            reason,
                            "cancelled",
                        )
                        .await?;
                    turn_tool_results.extend(pending);
                }
                break;
            }
        }

        Ok(Some(ParallelBatchOutcome {
            all_tools_terminate,
            last_tool_message,
            terminal_error,
        }))
    }

    async fn fit_context(
        &self,
        context: RunContext,
        step: usize,
        messages: Vec<Message>,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
    ) -> Result<Vec<Message>, AgentError> {
        match truncate_context(
            &messages,
            self.policy.max_context_messages,
            self.policy.max_context_bytes,
        ) {
            Ok(messages) => {
                validate_context_structure(&messages)?;
                Ok(messages)
            }
            Err(error @ AgentError::ContextTooLarge { .. }) => {
                let Some(hook) = &self.context_compaction_hook else {
                    return Err(error);
                };
                let compacted = match await_tool_hook(
                    hook(ContextCompactionRequest {
                        context,
                        step,
                        reason: ContextCompactionReason::ByteBudget,
                        messages,
                        max_context_messages: self.policy.max_context_messages,
                        max_context_bytes: self.policy.max_context_bytes,
                    }),
                    deadline,
                    cancellation,
                )
                .await
                {
                    ToolHookWait::Completed(Ok(messages)) => messages,
                    ToolHookWait::Completed(Err(error)) => {
                        return Err(AgentError::ContextCompaction(error));
                    }
                    ToolHookWait::TimedOut(timeout) => return Err(timeout_error(timeout)),
                    ToolHookWait::Cancelled => return Err(AgentError::RunCancelled),
                };
                let compacted = truncate_context(
                    &compacted,
                    self.policy.max_context_messages,
                    self.policy.max_context_bytes,
                )?;
                validate_context_structure(&compacted)?;
                Ok(compacted)
            }
            Err(error) => Err(error),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn prepare_memory_context(
        &self,
        run: RunContext,
        step: usize,
        runtime_run_id: RunId,
        messages: &mut Vec<Message>,
        runtime: &mut RuntimeCoordinator<Box<dyn EventLedger + Send>>,
        deadline: Option<(tokio::time::Instant, Duration)>,
        cancellation: &CancellationToken,
    ) -> Result<(), AgentError> {
        let Some((config, hook)) = &self.memory_hook else {
            return Ok(());
        };

        let request_now_ms = unix_timestamp_ms();
        let hook_context = MemoryHookContext {
            run_id: runtime_run_id,
            turn_id: Some(run.turn_id),
            namespace: config.namespace.clone(),
            current_revision: runtime.current_revision(),
            memory_revision: config.memory_revision,
            policy_revision: config.policy_revision,
            now_ms: request_now_ms,
            max_tokens: config.max_tokens,
            max_candidates: config.max_candidates,
            max_projection_bytes: config.max_projection_bytes,
        };
        let hook_request = hook(MemoryHookRequest {
            context: hook_context.clone(),
            messages: messages.clone(),
        });
        let proposal = match deadline {
            Some((deadline, timeout)) => tokio::select! {
                result = tokio::time::timeout_at(deadline, hook_request) => {
                    result
                        .map_err(|_| timeout_error(timeout))?
                        .map_err(AgentError::MemoryHook)?
                }
                _ = cancellation.cancelled() => return Err(AgentError::RunCancelled),
            },
            None => tokio::select! {
                result = hook_request => result.map_err(AgentError::MemoryHook)?,
                _ = cancellation.cancelled() => return Err(AgentError::RunCancelled),
            },
        };
        let Some(proposal) = proposal else {
            return Ok(());
        };

        let evaluation_now_ms = unix_timestamp_ms();
        let mut evaluation_context = hook_context;
        evaluation_context.now_ms = evaluation_now_ms;
        let preview = proposal
            .evaluate(&evaluation_context)
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
        if let InjectionDecision::Inject { injections, .. } = &preview {
            let content = format_memory_context(injections);
            if !content.is_empty() {
                let system_index = messages
                    .iter()
                    .take_while(|message| message.role == "system")
                    .count();
                let mut preview_messages = messages.clone();
                preview_messages.insert(system_index, Message::system(content));
                if self.context_compaction_hook.is_none() {
                    truncate_context(
                        &preview_messages,
                        self.policy.max_context_messages,
                        self.policy.max_context_bytes,
                    )?;
                }
            }
        }
        let receipt = runtime
            .evaluate_memory_proposal_with_receipt(
                &proposal,
                &evaluation_context,
                evaluation_now_ms,
            )
            .map_err(|error| AgentError::Runtime(error.to_string()))?;
        let decision_event = match &receipt.decision {
            InjectionDecision::Inject {
                proposal_id,
                injections,
                total_token_cost,
            } => AgentEvent::MemoryDecision {
                context: run,
                step,
                proposal_id: *proposal_id,
                event_sequence: receipt.event_sequence.value(),
                decision_kind: "injected".to_string(),
                injection_count: injections.len(),
                total_token_cost: *total_token_cost,
                skip_reason: None,
            },
            InjectionDecision::Skip {
                proposal_id,
                reason,
            } => AgentEvent::MemoryDecision {
                context: run,
                step,
                proposal_id: *proposal_id,
                event_sequence: receipt.event_sequence.value(),
                decision_kind: "skipped".to_string(),
                injection_count: 0,
                total_token_cost: 0,
                skip_reason: Some(memory_skip_reason_name(*reason).to_string()),
            },
        };
        self.emit(decision_event);
        if let InjectionDecision::Inject { injections, .. } = receipt.decision {
            let content = format_memory_context(&injections);
            if !content.is_empty() {
                let system_index = messages
                    .iter()
                    .take_while(|message| message.role == "system")
                    .count();
                messages.insert(system_index, Message::system(content));
            }
        }
        Ok(())
    }

    async fn inject_steering_messages(
        &self,
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
    ) -> Result<bool, AgentError> {
        let mut injected = false;
        while let Some(prompt) = self.peek_steering() {
            let message = Message::user(prompt);
            if let Some(session) = &self.session {
                session.append(&message).await?;
            }
            messages.push(message.clone());
            transcript.push(message);
            self.ack_steering();
            injected = true;
        }
        Ok(injected)
    }

    async fn inject_follow_up_message(
        &self,
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
    ) -> Result<bool, AgentError> {
        let Some(prompt) = self.peek_follow_up() else {
            return Ok(false);
        };
        let message = Message::user(prompt);
        if let Some(session) = &self.session {
            session.append(&message).await?;
        }
        messages.push(message.clone());
        transcript.push(message);
        self.ack_follow_up();
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    async fn append_cancelled_tool_results(
        &self,
        context: RunContext,
        step: usize,
        tool_calls: &[ToolCall],
        first_pending: usize,
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
        runtime_identity: Option<(Uuid, Uuid)>,
        reason: &str,
    ) -> Result<(), AgentError> {
        for (index, tool_call) in tool_calls.iter().enumerate().skip(first_pending) {
            if index > first_pending {
                self.emit(AgentEvent::ToolStarted {
                    context,
                    step,
                    call: tool_call.clone(),
                });
            }
            let admitted_unknown = index == first_pending && runtime_identity.is_some();
            let details = serde_json::json!({
                "status": if admitted_unknown { "unknown" } else { "cancelled" },
                "executed": admitted_unknown,
            });
            let (action_id, grant_id) = if index == first_pending {
                runtime_identity
                    .map(|(action_id, grant_id)| (Some(action_id), Some(grant_id)))
                    .unwrap_or((None, None))
            } else {
                (None, None)
            };
            self.emit(AgentEvent::ToolFinished {
                context,
                step,
                call_id: tool_call.id.clone(),
                tool_name: tool_call.function.name.clone(),
                content: reason.to_string(),
                is_error: true,
                details: details.clone(),
                terminate: false,
                duration_ms: 0,
                status: if admitted_unknown {
                    "unknown".to_string()
                } else {
                    "cancelled".to_string()
                },
                action_id,
                grant_id,
            });
            let tool_message = Message::tool_with_result(
                tool_call.id.clone(),
                reason,
                match (action_id, grant_id) {
                    (Some(action_id), Some(grant_id)) => {
                        ToolResultMetadata::new(details, true, false)
                            .with_runtime_identity(action_id, grant_id)
                    }
                    _ => ToolResultMetadata::new(details, true, false),
                },
            );
            messages.push(tool_message.clone());
            transcript.push(tool_message.clone());
            if let Some(session) = &self.session {
                session.append(&tool_message).await?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn append_unexecuted_tool_results(
        &self,
        context: RunContext,
        step: usize,
        tool_calls: &[ToolCall],
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
        reason: &str,
        status: &str,
    ) -> Result<Vec<Message>, AgentError> {
        let mut results = Vec::with_capacity(tool_calls.len());
        for tool_call in tool_calls {
            self.emit(AgentEvent::ToolStarted {
                context,
                step,
                call: tool_call.clone(),
            });
            let details = serde_json::json!({
                "status": status,
                "executed": false,
            });
            self.emit(AgentEvent::ToolFinished {
                context,
                step,
                call_id: tool_call.id.clone(),
                tool_name: tool_call.function.name.clone(),
                content: reason.to_string(),
                is_error: true,
                details: details.clone(),
                terminate: false,
                duration_ms: 0,
                status: status.to_string(),
                action_id: None,
                grant_id: None,
            });
            let tool_message = Message::tool_with_result(
                tool_call.id.clone(),
                reason,
                ToolResultMetadata::new(details, true, false),
            );
            messages.push(tool_message.clone());
            transcript.push(tool_message.clone());
            if let Some(session) = &self.session {
                session.append(&tool_message).await?;
            }
            results.push(tool_message);
        }
        Ok(results)
    }

    fn emit(&self, event: AgentEvent) {
        if let Some(sink) = &self.event_sink {
            sink.emit(event);
        }
    }
}

async fn await_tool_hook<T>(
    hook: ToolHookFuture<T>,
    deadline: Option<(tokio::time::Instant, Duration)>,
    cancellation: &CancellationToken,
) -> ToolHookWait<T> {
    match deadline {
        Some((deadline, run_timeout)) => tokio::select! {
            result = tokio::time::timeout_at(deadline, hook) => match result {
                Ok(_) if cancellation.is_cancelled() => ToolHookWait::Cancelled,
                Ok(_) if tokio::time::Instant::now() >= deadline => {
                    ToolHookWait::TimedOut(run_timeout)
                }
                Ok(result) => ToolHookWait::Completed(result),
                Err(_) => ToolHookWait::TimedOut(run_timeout),
            },
            _ = cancellation.cancelled() => ToolHookWait::Cancelled,
        },
        None => tokio::select! {
            result = hook => {
                if cancellation.is_cancelled() {
                    ToolHookWait::Cancelled
                } else {
                    ToolHookWait::Completed(result)
                }
            },
            _ = cancellation.cancelled() => ToolHookWait::Cancelled,
        },
    }
}

async fn execute_parallel_tool(
    tools: ToolGateway,
    prepared: PreparedToolCall,
    tool_context: ToolExecutionContext,
    tool_timeout: Duration,
    deadline: Option<(tokio::time::Instant, Duration)>,
    cancellation: CancellationToken,
) -> ParallelExecutionOutcome {
    match deadline {
        Some((deadline, run_timeout)) => {
            tokio::select! {
                result = tokio::time::timeout_at(
                    deadline,
                    tokio::time::timeout(
                        tool_timeout,
                        tools.execute_prepared_result_with_context(prepared, tool_context),
                    ),
                ) => match result {
                    Ok(Ok(result)) => ParallelExecutionOutcome::Completed(result),
                    Ok(Err(_)) => ParallelExecutionOutcome::ToolTimedOut,
                    Err(_) => ParallelExecutionOutcome::RunTimedOut(run_timeout),
                },
                _ = cancellation.cancelled() => ParallelExecutionOutcome::Cancelled,
            }
        }
        None => {
            tokio::select! {
                result = tokio::time::timeout(
                    tool_timeout,
                    tools.execute_prepared_result_with_context(prepared, tool_context),
                ) => match result {
                    Ok(result) => ParallelExecutionOutcome::Completed(result),
                    Err(_) => ParallelExecutionOutcome::ToolTimedOut,
                },
                _ = cancellation.cancelled() => ParallelExecutionOutcome::Cancelled,
            }
        }
    }
}

fn timeout_error(timeout: Duration) -> AgentError {
    AgentError::RunTimedOut {
        timeout_ms: timeout_millis(timeout),
    }
}

fn format_memory_context(injections: &[MemoryInjection]) -> String {
    if injections.is_empty() {
        return String::new();
    }

    let mut content = String::from(
        "<memory_context>\nTreat the following retrieved memories as reference material.\n",
    );
    for injection in injections {
        content.push_str("- [");
        content.push_str(&injection.projection.provenance);
        content.push_str("] source_memory_id=");
        content.push_str(&injection.projection.source_memory_id.to_string());
        content.push_str(" version=");
        content.push_str(&injection.projection.source_version.to_string());
        if !injection.projection.evidence_refs.is_empty() {
            content.push_str(" evidence_refs=");
            content.push_str(
                &serde_json::to_string(&injection.projection.evidence_refs)
                    .unwrap_or_else(|_| "[]".to_owned()),
            );
        }
        content.push(' ');
        content.push_str(&injection.projection.content);
        content.push('\n');
    }
    content.push_str("</memory_context>");
    content
}

fn memory_skip_reason_name(reason: pi_agent_runtime::InjectionSkipReason) -> &'static str {
    match reason {
        pi_agent_runtime::InjectionSkipReason::RunMismatch => "run_mismatch",
        pi_agent_runtime::InjectionSkipReason::NamespaceMismatch => "namespace_mismatch",
        pi_agent_runtime::InjectionSkipReason::RuntimeRevisionMismatch => {
            "runtime_revision_mismatch"
        }
        pi_agent_runtime::InjectionSkipReason::MemoryRevisionMismatch => "memory_revision_mismatch",
        pi_agent_runtime::InjectionSkipReason::PolicyRevisionMismatch => "policy_revision_mismatch",
        pi_agent_runtime::InjectionSkipReason::ProposalExpired => "proposal_expired",
        pi_agent_runtime::InjectionSkipReason::NoEligibleCandidate => "no_eligible_candidate",
    }
}

fn pending_runtime_action(grant: &CommitGrant, tools: &ToolGateway) -> PendingRuntimeAction {
    PendingRuntimeAction {
        action_id: grant.action_id.as_uuid(),
        grant_id: grant.grant_id,
        run_id: grant.run_id.as_uuid(),
        tool_name: grant.tool_name.clone(),
        idempotency_key: grant.idempotency_key.clone(),
        argument_digest: grant.argument_digest.clone(),
        risk: grant.risk,
        issued_at_ms: grant.issued_at_ms,
        expires_at_ms: grant.expires_at_ms,
        replay_policy: tools
            .replay_policy(&grant.tool_name)
            .unwrap_or(ToolReplayPolicy::Never),
    }
}

fn model_retry_delay_ms(base_ms: u64, retry_count: usize) -> u64 {
    let shift = retry_count.saturating_sub(1).min(6) as u32;
    base_ms.saturating_mul(1_u64 << shift).min(30_000)
}

fn runtime_policy(
    definitions: &[ToolDefinition],
    tool_timeout_secs: u64,
    allow_irreversible_tools: bool,
) -> AdmissionPolicy {
    let mut policy = AdmissionPolicy::deny_all().with_budget_limit(Budget {
        wall_time_ms: Some(tool_timeout_secs.saturating_mul(1_000)),
        tokens: None,
        tool_calls: Some(1),
    });
    if allow_irreversible_tools {
        policy = policy.allow_irreversible();
    }
    for definition in definitions {
        let name = definition.function.name.clone();
        policy = policy
            .allow_tool(name.clone())
            .allow_resources(ResourceScope::new([format!("tool:{name}")]));
    }
    policy
}

fn validate_session_runtime_consistency<L: EventLedger>(
    messages: &[Message],
    runtime: &RuntimeCoordinator<L>,
) -> Result<(), AgentError> {
    let events = runtime
        .ledger()
        .scan_after(EventSequence::INITIAL)
        .map_err(|error| AgentError::Runtime(error.to_string()))?;
    let mut admitted = HashMap::new();
    let mut terminal = HashSet::new();
    for event in events {
        let Some(action_id) = event.draft.payload.get("action_id").and_then(Value::as_str) else {
            continue;
        };
        if event.draft.kind.as_str() == pi_agent_runtime::ACTION_ADMITTED_EVENT {
            if let Some(grant_id) = event.draft.payload.get("grant_id").and_then(Value::as_str) {
                admitted.insert(action_id.to_string(), grant_id.to_string());
            }
        }
        if matches!(
            event.draft.kind.as_str(),
            pi_agent_runtime::ACTION_SUCCEEDED_EVENT
                | pi_agent_runtime::ACTION_FAILED_EVENT
                | pi_agent_runtime::ACTION_UNKNOWN_EVENT
                | pi_agent_runtime::ACTION_RECONCILED_EVENT
        ) {
            terminal.insert(action_id.to_string());
        }
    }

    for message in messages {
        let Some(metadata) = message.tool_result.as_ref() else {
            continue;
        };
        let Some(action_id) = metadata.action_id else {
            if let Some(status) = metadata.details.get("status").and_then(Value::as_str) {
                let anonymous_outcome_is_uncertain = match status {
                    "unknown" | "timed_out" => true,
                    // Cancellation before runtime admission cannot have
                    // caused an external side effect. If the execution bit
                    // is absent or true, the result still needs reconciliation.
                    "cancelled" => metadata.details.get("executed") != Some(&Value::Bool(false)),
                    _ => false,
                };
                if anonymous_outcome_is_uncertain {
                    return Err(AgentError::UnidentifiedToolOutcome {
                        call_id: message.tool_call_id.clone().unwrap_or_default(),
                        status: status.to_string(),
                    });
                }
            }
            continue;
        };
        let action_id_text = action_id.to_string();
        let Some(expected_grant_id) = admitted.get(&action_id_text) else {
            return Err(AgentError::RuntimeSessionMismatch {
                action_id,
                reason: "durable admission event is missing".to_string(),
            });
        };
        if !terminal.contains(&action_id_text) {
            return Err(AgentError::RuntimeSessionMismatch {
                action_id,
                reason: "durable terminal event is missing".to_string(),
            });
        }
        let Some(grant_id) = metadata.grant_id else {
            return Err(AgentError::RuntimeSessionMismatch {
                action_id,
                reason: "session result is missing its grant identity".to_string(),
            });
        };
        if grant_id.to_string() != *expected_grant_id {
            return Err(AgentError::RuntimeSessionMismatch {
                action_id,
                reason: format!(
                    "grant identity {grant_id} does not match durable grant {expected_grant_id}"
                ),
            });
        }
    }
    Ok(())
}

fn apply_reconciled_tool_results<L: EventLedger>(
    messages: &mut [Message],
    runtime: &RuntimeCoordinator<L>,
    max_output_bytes: usize,
) -> Result<(), AgentError> {
    let events = runtime
        .ledger()
        .scan_after(EventSequence::INITIAL)
        .map_err(|error| AgentError::Runtime(error.to_string()))?;
    let mut reconciled = HashMap::new();
    for event in events {
        if event.draft.kind.as_str() != pi_agent_runtime::ACTION_RECONCILED_EVENT {
            continue;
        }
        let Some(action_id) = event.draft.payload.get("action_id").and_then(Value::as_str) else {
            continue;
        };
        reconciled.insert(action_id.to_string(), event.draft.payload);
    }

    for message in messages {
        let Some(metadata) = message.tool_result.as_mut() else {
            continue;
        };
        let Some(action_id) = metadata.action_id else {
            continue;
        };
        let Some(event) = reconciled.get(&action_id.to_string()) else {
            continue;
        };
        let outcome = event
            .get("outcome")
            .and_then(Value::as_str)
            .unwrap_or("failed");
        let is_error = outcome == "failed";
        let content = if is_error {
            event
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("runtime action failed during reconciliation")
                .to_string()
        } else {
            match event.get("output").cloned().unwrap_or(Value::Null) {
                Value::String(content) => content,
                output => output.to_string(),
            }
        };
        let bounded = ToolResult {
            content,
            details: json!({
                "status": "reconciled",
                "outcome": outcome,
                "evidence": event.get("evidence").cloned().unwrap_or(Value::Null),
                "executed": true,
            }),
            is_error,
            terminate: metadata.terminate,
        }
        .bounded(max_output_bytes);
        message.content = Some(bounded.content);
        metadata.details = bounded.details;
        metadata.is_error = bounded.is_error;
    }
    Ok(())
}

fn unix_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64)
        .unwrap_or(0)
}

fn timeout_millis(timeout: Duration) -> u64 {
    timeout.as_millis().min(u64::MAX as u128) as u64
}

fn elapsed_millis(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u64::MAX as u128) as u64
}

fn tool_result_status(result: &ToolResult) -> String {
    let detail_status = result.details.get("status").and_then(Value::as_str);
    if detail_status.is_some_and(|status| {
        matches!(
            status,
            "cancelled"
                | "timed_out"
                | "unknown"
                | "rejected"
                | "truncated"
                | "max_steps"
                | "reconciled"
        )
    }) {
        return detail_status.expect("status was checked above").to_string();
    }
    if result.is_error {
        "failed".to_string()
    } else {
        "succeeded".to_string()
    }
}

fn run_outcome_for_error(error: &AgentError) -> RunOutcome {
    match error {
        AgentError::RuntimeActionUnknown { .. } => RunOutcome::Unknown,
        AgentError::ModelTerminated { reason } if reason == "aborted" => RunOutcome::Cancelled,
        AgentError::RunTimedOut { .. } | AgentError::RunCancelled => RunOutcome::Cancelled,
        _ => RunOutcome::Failed,
    }
}

fn run_outcome_for_error_with_runtime(error: &AgentError, has_unknown_action: bool) -> RunOutcome {
    if has_unknown_action {
        RunOutcome::Unknown
    } else {
        run_outcome_for_error(error)
    }
}

fn tool_result_has_unknown_outcome(result: &ToolResult) -> bool {
    result
        .details
        .get("status")
        .and_then(Value::as_str)
        .is_some_and(|status| matches!(status, "unknown" | "cancelled" | "timed_out"))
}

fn validate_tool_call_batch(step: usize, tool_calls: &[ToolCall]) -> Result<(), AgentError> {
    let mut call_ids = BTreeSet::new();
    for (index, call) in tool_calls.iter().enumerate() {
        if call.id.trim().is_empty() {
            return Err(AgentError::InvalidToolCall {
                step,
                index,
                reason: "call id must not be empty".to_string(),
            });
        }
        if !call_ids.insert(call.id.as_str()) {
            return Err(AgentError::InvalidToolCall {
                step,
                index,
                reason: format!("duplicate call id {:?}", call.id),
            });
        }
        if call.call_type != "function" {
            return Err(AgentError::InvalidToolCall {
                step,
                index,
                reason: format!("unsupported call type {:?}", call.call_type),
            });
        }
        if call.function.name.trim().is_empty() {
            return Err(AgentError::InvalidToolCall {
                step,
                index,
                reason: "function name must not be empty".to_string(),
            });
        }
    }
    Ok(())
}

fn preflight_model_tool_call(
    tools: &ToolGateway,
    tool_call: &ToolCall,
) -> (Value, Option<PreparedToolCall>, Option<String>) {
    let arguments = match serde_json::from_str::<Value>(&tool_call.function.arguments) {
        Ok(value) if value.is_object() => value,
        Ok(_) => {
            return (
                Value::Null,
                None,
                Some("tool arguments must be a JSON object".to_string()),
            );
        }
        Err(_) => {
            return (
                Value::Null,
                None,
                Some("tool arguments must be valid JSON".to_string()),
            );
        }
    };

    match tools.prepare_model_call(tool_call, arguments.clone()) {
        Ok(prepared) => {
            let arguments = prepared.arguments().clone();
            (arguments, Some(prepared), None)
        }
        Err(error) => (
            arguments,
            None,
            Some(format!("invalid tool arguments: {error}")),
        ),
    }
}

fn validate_policy(policy: RunPolicy) -> Result<(), AgentError> {
    if policy.max_steps == 0 {
        return Err(AgentError::InvalidMaxSteps);
    }
    if policy.max_tool_calls_per_step == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "max_tool_calls_per_step",
        });
    }
    if policy.max_parallel_tool_calls == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "max_parallel_tool_calls",
        });
    }
    if policy.tool_timeout_secs == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "tool_timeout_secs",
        });
    }
    if policy.run_timeout_secs == Some(0) {
        return Err(AgentError::InvalidRunPolicy {
            field: "run_timeout_secs",
        });
    }
    if policy.max_context_messages == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "max_context_messages",
        });
    }
    if policy.max_context_bytes == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "max_context_bytes",
        });
    }
    Ok(())
}

/// Keep the newest complete user turns within the context budget.
///
/// System messages are retained when they fit the byte budget. A turn starts at a user message and
/// includes every following assistant and tool message until the next user
/// message. If the newest turn is larger than the budget, it is kept whole so
/// an assistant tool call is never separated from its tool results; if it
/// cannot fit, the run fails with a typed context-limit error instead of
/// sending an oversized request. The byte budget is an approximate
/// serialized-message limit; providers should still apply their own
/// token-aware request limit when one is available.
fn truncate_context(
    messages: &[Message],
    max_context_messages: usize,
    max_context_bytes: usize,
) -> Result<Vec<Message>, AgentError> {
    let mut system_messages = Vec::new();
    let mut turns = Vec::new();
    let mut current_turn = Vec::new();

    for message in messages {
        if message.role == "system" {
            system_messages.push(message.clone());
            continue;
        }
        if message.role == "user" && !current_turn.is_empty() {
            turns.push(current_turn);
            current_turn = Vec::new();
        }
        current_turn.push(message.clone());
    }
    if !current_turn.is_empty() {
        turns.push(current_turn);
    }

    let mut selected_turns = Vec::new();
    let mut used_messages = 0;
    let system_bytes = system_messages.iter().map(message_size).sum::<usize>();
    if system_bytes > max_context_bytes {
        return Err(AgentError::ContextTooLarge {
            required_bytes: system_bytes,
            limit_bytes: max_context_bytes,
        });
    }
    let turn_budget = max_context_bytes.saturating_sub(system_bytes);
    let mut used_bytes: usize = 0;
    for turn in turns.into_iter().rev() {
        let turn_bytes = turn.iter().map(message_size).sum::<usize>();
        if selected_turns.is_empty() && turn_bytes > turn_budget {
            return Err(AgentError::ContextTooLarge {
                required_bytes: system_bytes.saturating_add(turn_bytes),
                limit_bytes: max_context_bytes,
            });
        }
        if selected_turns.is_empty()
            || (used_messages + turn.len() <= max_context_messages
                && used_bytes.saturating_add(turn_bytes) <= turn_budget)
        {
            used_messages += turn.len();
            used_bytes = used_bytes.saturating_add(turn_bytes);
            selected_turns.push(turn);
        } else {
            break;
        }
    }
    selected_turns.reverse();

    let mut result = system_messages;
    result.extend(selected_turns.into_iter().flatten());
    Ok(result)
}

/// Validate the message structure sent to a model provider.
///
/// Context compaction is allowed to rewrite messages, but it must preserve the
/// same tool-call protocol as a durable session: system messages stay at the
/// front, every tool result belongs to one preceding assistant call, and no
/// assistant call is left without all of its results.
fn validate_context_structure(messages: &[Message]) -> Result<(), AgentError> {
    let mut saw_non_system = false;
    let mut saw_user = false;
    let mut declared_calls = BTreeSet::new();
    let mut pending_calls = BTreeSet::new();

    for message in messages {
        match message.role.as_str() {
            "system" => {
                if saw_non_system {
                    return Err(AgentError::InvalidContext {
                        reason: "system messages must precede conversation messages".to_string(),
                    });
                }
            }
            "user" => {
                if !pending_calls.is_empty() {
                    return Err(AgentError::InvalidContext {
                        reason: "a user message appears before all tool results".to_string(),
                    });
                }
                saw_non_system = true;
                saw_user = true;
            }
            "assistant" => {
                if !pending_calls.is_empty() {
                    return Err(AgentError::InvalidContext {
                        reason: "an assistant message appears before all tool results".to_string(),
                    });
                }
                saw_non_system = true;
                if let Some(tool_calls) = &message.tool_calls {
                    for call in tool_calls {
                        if call.id.trim().is_empty() || !declared_calls.insert(call.id.as_str()) {
                            return Err(AgentError::InvalidContext {
                                reason: format!(
                                    "assistant tool call id {:?} is invalid or duplicated",
                                    call.id
                                ),
                            });
                        }
                        pending_calls.insert(call.id.clone());
                    }
                }
            }
            "tool" => {
                saw_non_system = true;
                let Some(call_id) = message.tool_call_id.as_deref() else {
                    return Err(AgentError::InvalidContext {
                        reason: "tool result is missing its tool_call_id".to_string(),
                    });
                };
                if !pending_calls.remove(call_id) {
                    return Err(AgentError::InvalidContext {
                        reason: format!(
                            "tool result refers to unknown or completed call {call_id:?}"
                        ),
                    });
                }
            }
            role => {
                return Err(AgentError::InvalidContext {
                    reason: format!("unsupported message role {role:?}"),
                });
            }
        }
    }

    if !saw_user {
        return Err(AgentError::InvalidContext {
            reason: "context must contain at least one user message".to_string(),
        });
    }
    if !pending_calls.is_empty() {
        return Err(AgentError::InvalidContext {
            reason: format!("tool calls are missing results: {pending_calls:?}"),
        });
    }
    Ok(())
}

fn message_size(message: &Message) -> usize {
    serde_json::to_vec(message)
        .map(|encoded| encoded.len())
        .unwrap_or_else(|_| message.content.as_deref().unwrap_or_default().len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        message::{FunctionCall, ToolCall, ToolDefinition},
        model::{ChatFuture, DeltaCallback, ReasoningDeltaCallback},
        session::SessionStore,
        skills::Skill,
        tools::{
            Tool, ToolConcurrency, ToolContext, ToolError, ToolExecutionContext, ToolExposure,
            ToolFuture, ToolReplayPolicy, ToolResult, ToolResultFuture, ToolUpdate,
        },
    };
    use pi_agent_runtime::{EventSequence, JsonlLedger, MemoryCandidate, MemoryProjection};
    use serde_json::json;
    use std::{
        collections::VecDeque,
        fs,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
        time::Duration,
    };
    use tempfile::tempdir;

    #[test]
    fn memory_context_keeps_source_identity_and_evidence_references() {
        let context = format_memory_context(&[MemoryInjection {
            candidate_id: Uuid::from_u128(1),
            projection: MemoryProjection {
                projection_id: Uuid::from_u128(2),
                source_memory_id: Uuid::from_u128(3),
                source_version: 4,
                content: "视频中的红色方块".to_owned(),
                evidence_refs: vec![json!({
                    "artifact_id": "artifact-7",
                    "uri": "cas://episode-7.mp4",
                    "frame_start": 360,
                    "step_start": 30
                })],
                provenance: "video-memory:episode-7".to_owned(),
                token_cost: 12,
            },
        }]);

        assert!(context.contains("source_memory_id=00000000-0000-0000-0000-000000000003"));
        assert!(context.contains("version=4"));
        assert!(context.contains("artifact-7"));
        assert!(context.contains("frame_start"));
        assert!(context.contains("step_start"));
    }

    #[test]
    fn unknown_runtime_action_has_a_distinct_run_outcome() {
        let error = AgentError::RuntimeActionUnknown {
            action_id: Uuid::new_v4(),
            tool_name: "write_file".to_string(),
            reason: "executor disconnected".to_string(),
        };

        assert_eq!(run_outcome_for_error(&error), RunOutcome::Unknown);
        assert_eq!(
            run_outcome_for_error(&AgentError::RunCancelled),
            RunOutcome::Cancelled
        );
        assert_eq!(
            run_outcome_for_error(&AgentError::EmptyPrompt),
            RunOutcome::Failed
        );
    }

    #[test]
    fn reconciled_tool_results_keep_their_terminal_status() {
        let result = ToolResult::text("confirmed").with_details(json!({
            "status": "reconciled",
            "outcome": "succeeded",
        }));

        assert_eq!(tool_result_status(&result), "reconciled");
    }

    use tokio::sync::Notify;
    use uuid::Uuid;
    use wiremock::{
        matchers::{body_string_contains, method},
        Mock, MockServer, ResponseTemplate,
    };

    fn test_config(base_url: String) -> Config {
        Config {
            base_url,
            model: "test-model".to_string(),
            max_steps: 2,
            request_timeout_secs: 5,
            ..Config::default()
        }
    }

    fn streaming_response(message: Message) -> ResponseTemplate {
        streaming_response_with_optional_usage(message, None)
    }

    fn streaming_response_with_usage(message: Message, usage: Value) -> ResponseTemplate {
        streaming_response_with_optional_usage(message, Some(usage))
    }

    fn streaming_response_with_optional_usage(
        message: Message,
        usage: Option<Value>,
    ) -> ResponseTemplate {
        let mut delta = serde_json::to_value(message.clone()).expect("message should serialize");
        if let Some(tool_calls) = delta
            .get_mut("tool_calls")
            .and_then(|value| value.as_array_mut())
        {
            for (index, tool_call) in tool_calls.iter_mut().enumerate() {
                tool_call["index"] = json!(index);
            }
        }
        let finish_reason = if message
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
        {
            "tool_calls"
        } else {
            "stop"
        };
        let mut event = json!({
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        });
        if let Some(usage) = usage {
            event["usage"] = usage;
        }
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(format!(
                "data: {}\n\ndata: [DONE]\n\n",
                serde_json::to_string(&event).expect("stream event should serialize")
            ))
    }

    #[test]
    fn truncates_old_turns_without_splitting_tool_results() {
        let messages = vec![
            Message::system("system rule"),
            Message::user("old question"),
            Message::assistant(Some("old answer".to_string()), None),
            Message::user("current question"),
            Message::assistant(
                None,
                Some(vec![ToolCall {
                    id: "call-1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read_file".to_string(),
                        arguments: r#"{"path":"Cargo.toml"}"#.to_string(),
                    },
                }]),
            ),
            Message::tool("call-1", "file content"),
            Message::assistant(Some("current answer".to_string()), None),
        ];

        let truncated =
            truncate_context(&messages, 2, usize::MAX).expect("latest complete turn should fit");

        assert_eq!(truncated.len(), 5);
        assert_eq!(truncated[0].role, "system");
        assert_eq!(truncated[1].content.as_deref(), Some("current question"));
        assert!(truncated[2].tool_calls.is_some());
        assert_eq!(truncated[3].role, "tool");
        assert_eq!(truncated[4].content.as_deref(), Some("current answer"));
        assert!(!truncated
            .iter()
            .any(|message| message.content.as_deref() == Some("old question")));
    }

    #[test]
    fn context_byte_budget_rejects_an_oversized_latest_tool_turn() {
        let messages = vec![
            Message::system("system rule"),
            Message::user("old question"),
            Message::assistant(Some("old answer".to_string()), None),
            Message::user("current question"),
            Message::assistant(None, Some(vec![tool_call("read_file", "call-1")])),
            Message::tool("call-1", "current tool result"),
            Message::assistant(Some("current answer".to_string()), None),
        ];
        let latest_turn_bytes = messages[3..].iter().map(message_size).sum::<usize>();

        let error = truncate_context(&messages, 128, latest_turn_bytes - 1)
            .expect_err("an oversized latest turn must not exceed the byte budget");

        assert!(matches!(
            error,
            AgentError::ContextTooLarge {
                limit_bytes,
                required_bytes
            } if limit_bytes == latest_turn_bytes - 1 && required_bytes > limit_bytes
        ));
    }

    #[test]
    fn context_byte_budget_rejects_oversized_system_messages() {
        let messages = vec![Message::system("system instruction that is too large")];

        let error = truncate_context(&messages, 128, message_size(&messages[0]) - 1)
            .expect_err("system context must respect the byte budget");

        assert!(matches!(error, AgentError::ContextTooLarge { .. }));
    }

    #[tokio::test]
    async fn context_compactor_recovers_an_oversized_request() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let observed_invocations = Arc::clone(&invocations);
        let agent = AgentLoop::new_with_policy(
            FakeChatModel {
                response: Message::assistant(Some("compacted answer".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                max_steps: 1,
                max_context_bytes: 128,
                ..RunPolicy::default()
            },
        )
        .expect("compaction agent should be valid")
        .with_context_compactor(move |request| {
            observed_invocations.fetch_add(1, Ordering::SeqCst);
            assert_eq!(request.reason, ContextCompactionReason::ByteBudget);
            assert_eq!(request.step, 1);
            Box::pin(async { Ok(vec![Message::user("compacted context")]) })
        });

        let result = agent
            .run("x".repeat(1_024))
            .await
            .expect("the compactor should produce a request within budget");

        assert_eq!(
            result.final_message.content.as_deref(),
            Some("compacted answer")
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn context_compactor_rejects_unpaired_tool_calls() {
        let agent = AgentLoop::new_with_policy(
            FakeChatModel {
                response: Message::assistant(Some("must not run".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                max_steps: 1,
                max_context_bytes: 512,
                ..RunPolicy::default()
            },
        )
        .expect("compaction agent should be valid")
        .with_context_compactor(|_request| {
            Box::pin(async {
                Ok(vec![
                    Message::user("compacted"),
                    Message::assistant(None, Some(vec![tool_call("read_file", "call-1")])),
                ])
            })
        });

        let error = agent
            .run("x".repeat(1_024))
            .await
            .expect_err("compaction must preserve tool-call/result pairing");

        assert!(matches!(
            error,
            AgentError::InvalidContext { reason }
                if reason.contains("missing results")
        ));
    }

    #[test]
    fn rejects_duplicate_tool_call_ids_before_runtime_admission() {
        let calls = vec![
            tool_call("read_file", "call-1"),
            tool_call("read_file", "call-1"),
        ];

        let error =
            validate_tool_call_batch(2, &calls).expect_err("duplicate call IDs should be rejected");

        assert!(matches!(
            error,
            AgentError::InvalidToolCall {
                step: 2,
                index: 1,
                reason,
            } if reason.contains("duplicate call id")
        ));
    }

    #[tokio::test]
    async fn returns_non_object_tool_arguments_as_a_tool_result() {
        let mut call = tool_call("echo_tool", "call-1");
        call.function.arguments = "[]".to_string();

        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![call])),
            Message::assistant(Some("recovered".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new();
        tools
            .register(ImmediateTool)
            .expect("echo tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("invalid argument agent should be valid");

        let result = agent
            .run("recover from invalid arguments")
            .await
            .expect("invalid arguments should be reported to the model");

        assert_eq!(result.final_message.content.as_deref(), Some("recovered"));
        let tool_message = result
            .messages
            .iter()
            .find(|message| message.role == "tool")
            .expect("invalid arguments should produce a tool message");
        assert_eq!(tool_message.tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(
            tool_message.content.as_deref(),
            Some("tool arguments must be a JSON object")
        );
    }

    #[tokio::test]
    async fn dispatches_parallel_read_tools_concurrently_and_preserves_order() {
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(
                None,
                Some(vec![
                    tool_call("parallel_sleep", "call-1"),
                    tool_call("parallel_sleep", "call-2"),
                ]),
            ),
            Message::assistant(Some("parallel complete".to_string()), None),
        ])));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let mut tools = ToolRegistry::new();
        tools
            .register(ParallelSleepTool {
                active: active.clone(),
                max_active: max_active.clone(),
            })
            .expect("parallel tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("parallel agent should be valid");

        let result = agent
            .run("run both read tools")
            .await
            .expect("parallel tools should complete");

        assert_eq!(max_active.load(Ordering::SeqCst), 2);
        let tool_messages = result
            .messages
            .iter()
            .filter(|message| message.role == "tool")
            .collect::<Vec<_>>();
        assert_eq!(tool_messages.len(), 2);
        assert_eq!(tool_messages[0].tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(tool_messages[1].tool_call_id.as_deref(), Some("call-2"));
    }

    #[derive(Debug)]
    struct FakeChatModel {
        response: Message,
    }

    impl ChatModel for FakeChatModel {
        fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
            let response = self.response.clone();
            Box::pin(async move { Ok(response) })
        }
    }

    #[derive(Debug)]
    struct FailingChatModel;

    impl ChatModel for FailingChatModel {
        fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
            Box::pin(async { Err(ModelError::EmptyResponse) })
        }
    }

    #[derive(Debug)]
    struct FailingAfterToolChatModel {
        calls: Arc<AtomicUsize>,
    }

    impl ChatModel for FailingAfterToolChatModel {
        fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                Box::pin(async {
                    Ok(Message::assistant(
                        None,
                        Some(vec![tool_call("echo_tool", "call-1")]),
                    ))
                })
            } else {
                Box::pin(async { Err(ModelError::EmptyResponse) })
            }
        }
    }

    #[derive(Debug)]
    struct ReasoningChatModel;

    impl ChatModel for ReasoningChatModel {
        fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
            Box::pin(async { Ok(Message::assistant(Some("answer".to_string()), None)) })
        }

        fn stream_with_reasoning<'a>(
            &'a self,
            _request: ChatRequest<'a>,
            on_delta: DeltaCallback<'a>,
            on_reasoning: ReasoningDeltaCallback<'a>,
        ) -> ChatFuture<'a> {
            on_reasoning("thinking");
            on_delta("answer");
            Box::pin(async {
                let mut message = Message::assistant(Some("answer".to_string()), None);
                message.reasoning_content = Some("thinking".to_string());
                Ok(message)
            })
        }
    }

    #[derive(Debug)]
    #[allow(dead_code)]
    struct RetryChatModel {
        calls: Arc<AtomicUsize>,
        response: Message,
    }

    impl ChatModel for RetryChatModel {
        fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
            let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
            let response = self.response.clone();
            Box::pin(async move {
                if attempt == 0 {
                    Err(ModelError::Api {
                        status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
                        message: "temporary provider outage".to_string(),
                    })
                } else {
                    Ok(response)
                }
            })
        }
    }

    #[derive(Debug)]
    struct PartialRetryChatModel {
        calls: Arc<AtomicUsize>,
    }

    impl ChatModel for PartialRetryChatModel {
        fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
            Box::pin(async { Err(ModelError::StreamIncomplete) })
        }

        fn stream_with_reasoning<'a>(
            &'a self,
            _request: ChatRequest<'a>,
            on_delta: DeltaCallback<'a>,
            on_reasoning: ReasoningDeltaCallback<'a>,
        ) -> ChatFuture<'a> {
            let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                on_reasoning("draft");
                on_delta("Hel");
                Box::pin(async { Err(ModelError::StreamIncomplete) })
            } else {
                on_delta("Hello");
                Box::pin(async { Ok(Message::assistant(Some("Hello".to_string()), None)) })
            }
        }
    }

    #[derive(Debug)]
    struct GatedStreamingChatModel {
        started: Arc<Notify>,
        release: Arc<Notify>,
    }

    impl ChatModel for GatedStreamingChatModel {
        fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
            Box::pin(async { Ok(Message::assistant(Some("partial".to_string()), None)) })
        }

        fn stream_with_reasoning<'a>(
            &'a self,
            _request: ChatRequest<'a>,
            on_delta: DeltaCallback<'a>,
            _on_reasoning: ReasoningDeltaCallback<'a>,
        ) -> ChatFuture<'a> {
            on_delta("partial");
            self.started.notify_one();
            let release = Arc::clone(&self.release);
            Box::pin(async move {
                release.notified().await;
                Ok(Message::assistant(Some("partial".to_string()), None))
            })
        }
    }

    #[derive(Debug)]
    struct SequenceChatModel {
        responses: Arc<Mutex<VecDeque<Message>>>,
    }

    impl ChatModel for SequenceChatModel {
        fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
            let response = self
                .responses
                .lock()
                .expect("response queue should not be poisoned")
                .pop_front()
                .expect("test response queue should contain a response");
            Box::pin(async move { Ok(response) })
        }
    }

    #[derive(Debug)]
    struct RecordingChatModel {
        response: Message,
        requests: Arc<Mutex<Vec<Vec<Message>>>>,
    }

    impl ChatModel for RecordingChatModel {
        fn complete<'a>(&'a self, request: ChatRequest<'a>) -> ChatFuture<'a> {
            self.requests
                .lock()
                .expect("request mutex should not be poisoned")
                .push(request.messages.to_vec());
            let response = self.response.clone();
            Box::pin(async move { Ok(response) })
        }
    }

    #[derive(Debug)]
    struct SlowChatModel {
        delay: Duration,
        response: Message,
    }

    #[derive(Debug)]
    struct BlockingChatModel {
        started: Arc<Notify>,
        release: Arc<Notify>,
        response: Message,
    }

    impl ChatModel for BlockingChatModel {
        fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
            let started = Arc::clone(&self.started);
            let release = Arc::clone(&self.release);
            let response = self.response.clone();
            Box::pin(async move {
                started.notify_one();
                release.notified().await;
                Ok(response)
            })
        }
    }

    impl ChatModel for SlowChatModel {
        fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
            let delay = self.delay;
            let response = self.response.clone();
            Box::pin(async move {
                tokio::time::sleep(delay).await;
                Ok(response)
            })
        }
    }

    #[derive(Debug)]
    struct SlowTool;

    impl Tool for SlowTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "slow_tool",
                "A test tool that completes after a delay.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            Box::pin(async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok::<String, ToolError>("done".to_string())
            })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    #[derive(Debug)]
    struct ParallelSleepTool {
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    impl Tool for ParallelSleepTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "parallel_sleep",
                "A read-only tool used to verify parallel dispatch.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            Box::pin(async { Ok::<String, ToolError>("done".to_string()) })
        }

        fn execute_result_with_context<'a>(
            &'a self,
            _arguments: serde_json::Value,
            _context: ToolExecutionContext,
        ) -> ToolResultFuture<'a> {
            let active = Arc::clone(&self.active);
            let max_active = Arc::clone(&self.max_active);
            Box::pin(async move {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_active.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(100)).await;
                active.fetch_sub(1, Ordering::SeqCst);
                Ok(ToolResult::text("done"))
            })
        }

        fn concurrency(&self) -> ToolConcurrency {
            ToolConcurrency::Parallel
        }

        fn replay_policy(&self) -> ToolReplayPolicy {
            ToolReplayPolicy::Safe
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    #[derive(Debug)]
    struct CancelInsideTool;

    impl Tool for CancelInsideTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "cancel_inside_tool",
                "Cancel the caller while the tool is running.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            Box::pin(async { Ok::<String, ToolError>("unused".to_string()) })
        }

        fn execute_result_with_context<'a>(
            &'a self,
            _arguments: serde_json::Value,
            context: ToolExecutionContext,
        ) -> ToolResultFuture<'a> {
            Box::pin(async move {
                context.cancellation_token().cancel();
                std::future::pending::<Result<ToolResult, ToolError>>().await
            })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    #[derive(Debug)]
    struct ToolTimeoutTool;

    impl Tool for ToolTimeoutTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "tool_timeout_tool",
                "A test tool that exceeds the per-tool timeout.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            Box::pin(async {
                tokio::time::sleep(Duration::from_millis(1_100)).await;
                Ok::<String, ToolError>("too late".to_string())
            })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    #[derive(Debug)]
    struct ImmediateTool;

    impl Tool for ImmediateTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "echo_tool",
                "Return a stable test result.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            Box::pin(async { Ok::<String, ToolError>("runtime-ok".to_string()) })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    #[derive(Debug)]
    struct UnknownRiskTool;

    impl Tool for UnknownRiskTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "unknown_risk_tool",
                "A tool whose risk class is intentionally unspecified.",
                json!({"type": "object", "properties": {}, "additionalProperties": false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            Box::pin(async { Ok::<String, ToolError>("must not execute".to_string()) })
        }
    }

    #[derive(Debug)]
    struct ErrorResultTool;

    impl Tool for ErrorResultTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "error_tool",
                "Return a structured tool error.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            Box::pin(async { Ok::<String, ToolError>("legacy error".to_string()) })
        }

        fn execute_result<'a>(&'a self, _arguments: serde_json::Value) -> ToolResultFuture<'a> {
            Box::pin(async {
                Ok(ToolResult::error("known failure").with_details(json!({"kind": "test"})))
            })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    #[derive(Debug)]
    struct UncertainResultTool;

    impl Tool for UncertainResultTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "uncertain_tool",
                "Return a result whose external outcome is uncertain.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            Box::pin(async { Ok::<String, ToolError>("uncertain".to_string()) })
        }

        fn execute_result<'a>(&'a self, _arguments: serde_json::Value) -> ToolResultFuture<'a> {
            Box::pin(async {
                Ok(ToolResult::error("command timed out")
                    .with_details(json!({"status": "timed_out"})))
            })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::Reversible
        }
    }

    #[derive(Debug)]
    struct CountingValidationTool {
        validations: Arc<AtomicUsize>,
    }

    impl Tool for CountingValidationTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "counting_validation_tool",
                "A test tool that records argument validation calls.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn validate_arguments(&self, _arguments: &serde_json::Value) -> Result<(), ToolError> {
            self.validations.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            Box::pin(async { Ok::<String, ToolError>("validated".to_string()) })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    #[derive(Debug)]
    struct HiddenModelTool {
        executions: Arc<AtomicUsize>,
    }

    impl Tool for HiddenModelTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "hidden_model_tool",
                "A trusted caller-only test tool.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn exposure(&self) -> ToolExposure {
            ToolExposure::Hidden
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            self.executions.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok::<String, ToolError>("should not execute".to_string()) })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    #[derive(Debug)]
    struct ProgressTool;

    impl Tool for ProgressTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "progress_tool",
                "Emit one progress update before completing.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            Box::pin(async { Ok::<String, ToolError>("legacy".to_string()) })
        }

        fn execute_result_with_context<'a>(
            &'a self,
            _arguments: serde_json::Value,
            context: ToolExecutionContext,
        ) -> ToolResultFuture<'a> {
            Box::pin(async move {
                context.update(ToolUpdate::text("halfway").with_details(json!({"progress": 0.5})));
                Ok(ToolResult::text("done"))
            })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    #[derive(Debug)]
    struct StructuredTool;

    impl Tool for StructuredTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "structured_tool",
                "Return structured output and terminate the run.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            Box::pin(async { Ok::<String, ToolError>("legacy".to_string()) })
        }

        fn execute_result<'a>(&'a self, _arguments: serde_json::Value) -> ToolResultFuture<'a> {
            Box::pin(async {
                Ok(ToolResult::text("structured")
                    .with_details(json!({"source": "test"}))
                    .terminating())
            })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    fn tool_call(name: &str, id: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    fn recovery_memory_hook(request: MemoryHookRequest) -> MemoryHookFuture {
        Box::pin(async move {
            Ok(Some(MemoryProposal {
                proposal_id: Uuid::new_v4(),
                run_id: request.context.run_id,
                namespace: request.context.namespace.clone(),
                base_revision: request.context.current_revision,
                memory_revision: request.context.memory_revision,
                policy_revision: request.context.policy_revision,
                candidates: vec![MemoryCandidate {
                    candidate_id: Uuid::new_v4(),
                    projection: MemoryProjection {
                        projection_id: Uuid::new_v4(),
                        source_memory_id: Uuid::new_v4(),
                        source_version: 1,
                        content: "recovered memory".to_owned(),
                        evidence_refs: Vec::new(),
                        provenance: "recovery-test".to_owned(),
                        token_cost: 4,
                    },
                    score: 1.0,
                    confidence: 1.0,
                    expires_at_ms: Some(request.context.now_ms.saturating_add(1_000)),
                }],
                created_at_ms: request.context.now_ms,
                expires_at_ms: Some(request.context.now_ms.saturating_add(1_000)),
                retrieval_summary: None,
            }))
        })
    }

    #[tokio::test]
    async fn before_tool_hook_can_block_a_call_without_runtime_admission() {
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
            Message::assistant(Some("blocked result handled".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new();
        tools
            .register(ImmediateTool)
            .expect("echo tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("hook agent should be valid")
        .with_before_tool_hook(|context| {
            Box::pin(async move {
                assert_eq!(context.call.function.name, "echo_tool");
                Ok(BeforeToolDecision::Block {
                    reason: "approval required".to_string(),
                })
            })
        });

        let result = agent.run("call echo").await.expect("run should succeed");

        assert_eq!(result.messages.len(), 4);
        assert_eq!(result.messages[2].role, "tool");
        assert!(result.messages[2]
            .content
            .as_deref()
            .is_some_and(|content| content.contains("approval required")));
        let metadata = result.messages[2]
            .tool_result
            .as_ref()
            .expect("blocked tool should retain structured metadata");
        assert_eq!(metadata.details["status"], "rejected");
        assert!(!metadata.terminate);
    }

    #[tokio::test]
    async fn before_tool_hook_can_terminate_a_tool_batch_before_admission() {
        let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
            None,
            Some(vec![
                tool_call("echo_tool", "call-1"),
                tool_call("echo_tool", "call-2"),
            ]),
        )])));
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut tools = ToolRegistry::new();
        tools
            .register(ImmediateTool)
            .expect("echo tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("hook agent should be valid")
        .with_event_sink({
            let events = events.clone();
            move |event| {
                events
                    .lock()
                    .expect("event mutex should not poison")
                    .push(event)
            }
        })
        .with_before_tool_hook(|_context| {
            Box::pin(async {
                Ok(BeforeToolDecision::Terminate {
                    reason: "approval was denied".to_string(),
                })
            })
        });

        let error = agent
            .run("call echo")
            .await
            .expect_err("termination should stop the run");
        assert!(matches!(
            error,
            AgentError::ToolTerminated { reason } if reason.contains("approval was denied")
        ));

        let events = events.lock().expect("event mutex should not poison");
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolFinished {
                call_id,
                status,
                terminate: true,
                action_id: None,
                grant_id: None,
                ..
            } if call_id == "call-1" && status == "rejected"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolFinished {
                call_id,
                status,
                action_id: None,
                grant_id: None,
                ..
            } if call_id == "call-2" && status == "cancelled"
        )));
    }

    #[tokio::test]
    async fn before_tool_hook_obeys_the_run_deadline() {
        let directory = tempdir().expect("temporary directory should exist");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
            None,
            Some(vec![tool_call("echo_tool", "call-1")]),
        )])));
        let mut tools = ToolRegistry::new();
        tools
            .register(ImmediateTool)
            .expect("echo tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("hook agent should be valid")
        .with_session(session.clone())
        .with_before_tool_hook(|_context| {
            Box::pin(async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(BeforeToolDecision::Allow)
            })
        });

        let error = agent
            .run_with_timeout("call echo", Duration::from_millis(10))
            .await
            .expect_err("a hanging before hook should exceed the run deadline");

        assert!(matches!(error, AgentError::RunTimedOut { .. }));
        let records = session.load().await.expect("session should load");
        assert_eq!(records.len(), 3);
        assert!(records[2]
            .message
            .content
            .as_deref()
            .is_some_and(|content| content.contains("before_tool_call hook timed out")));
    }

    #[tokio::test]
    async fn invalid_tool_arguments_are_rejected_before_hooks_and_admission() {
        let directory = tempdir().expect("temporary directory should exist");
        let mut invalid_call = tool_call("write_file", "call-1");
        invalid_call.function.arguments = json!({
            "path": "preflight.txt",
            "content": 42
        })
        .to_string();
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![invalid_call])),
            Message::assistant(Some("handled".to_string()), None),
        ])));
        let before_calls = Arc::new(AtomicUsize::new(0));
        let config = Config {
            workspace: directory.path().to_path_buf(),
            ..Config::default()
        };
        let ledger_path = directory.path().join("runtime.jsonl");
        let mut tools = ToolRegistry::new();
        tools
            .register_write_file(ToolContext::from_config(&config))
            .expect("write tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("preflight agent should be valid")
        .with_before_tool_hook({
            let before_calls = before_calls.clone();
            move |_context| {
                before_calls.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Ok(BeforeToolDecision::Allow) })
            }
        })
        .with_runtime_ledger_file(ledger_path.clone());

        let result = agent
            .run("write with invalid arguments")
            .await
            .expect("invalid tool arguments should become a tool error");

        assert_eq!(before_calls.load(Ordering::SeqCst), 0);
        assert!(result.messages[2]
            .content
            .as_deref()
            .is_some_and(|content| content.contains("invalid tool arguments")));
        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("runtime events should be readable");
        assert!(!events
            .iter()
            .any(|event| event.draft.kind.as_str() == "action_admitted"));
    }

    #[tokio::test]
    async fn runtime_admission_rejections_are_structured_before_model_retry() {
        let directory = tempdir().expect("temporary directory should exist");
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![tool_call("unknown_risk_tool", "call-1")])),
            Message::assistant(Some("handled".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new();
        tools
            .register(UnknownRiskTool)
            .expect("unknown-risk tool should register");
        let ledger_path = directory.path().join("runtime.jsonl");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("admission rejection agent should be valid")
        .with_runtime_ledger_file(ledger_path.clone());

        let result = agent
            .run("call a tool without a risk declaration")
            .await
            .expect("admission rejection should be returned to the model");

        let metadata = result.messages[2]
            .tool_result
            .as_ref()
            .expect("rejected call should have structured metadata");
        assert_eq!(metadata.details["status"], "rejected");
        assert_eq!(metadata.details["executed"], false);
        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        assert!(ledger
            .scan_after(EventSequence::INITIAL)
            .expect("runtime events should be readable")
            .is_empty());
    }

    #[tokio::test]
    async fn prepared_tool_call_reuses_validation_for_execution() {
        let validations = Arc::new(AtomicUsize::new(0));
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(
                None,
                Some(vec![tool_call("counting_validation_tool", "call-1")]),
            ),
            Message::assistant(Some("finished".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new();
        tools
            .register(CountingValidationTool {
                validations: validations.clone(),
            })
            .expect("counting tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("prepared-call agent should be valid");

        let result = agent
            .run("use the counting tool")
            .await
            .expect("prepared tool call should complete");

        assert_eq!(result.final_message.content.as_deref(), Some("finished"));
        assert_eq!(validations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn model_cannot_invoke_a_hidden_tool_by_name() {
        let executions = Arc::new(AtomicUsize::new(0));
        let hidden_call = tool_call("hidden_model_tool", "call-hidden");
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![hidden_call])),
            Message::assistant(Some("recovered".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new();
        tools
            .register(HiddenModelTool {
                executions: executions.clone(),
            })
            .expect("hidden tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("hidden-tool agent should be valid");

        let result = agent
            .run("try the hidden tool")
            .await
            .expect("hidden call should become a recoverable tool error");

        assert_eq!(executions.load(Ordering::SeqCst), 0);
        assert!(result.messages.iter().any(|message| {
            message.role == "tool"
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("not available to the model"))
        }));
    }

    #[tokio::test]
    async fn truncated_tool_calls_are_persisted_as_unexecuted_errors() {
        let mut truncated =
            Message::assistant(None, Some(vec![tool_call("read_file", "call-truncated")]));
        truncated.stop_reason = Some("length".to_string());
        let responses = Arc::new(Mutex::new(VecDeque::from([
            truncated,
            Message::assistant(Some("recovered".to_string()), None),
        ])));
        let directory = tempdir().expect("temporary directory should exist");
        let config = Config {
            workspace: directory.path().to_path_buf(),
            ..Config::default()
        };
        let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
            .expect("read tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("truncation agent should be valid");

        let result = agent
            .run("recover a truncated tool call")
            .await
            .expect("model should recover after synthetic tool error");

        assert_eq!(result.final_message.content.as_deref(), Some("recovered"));
        assert_eq!(result.messages[1].stop_reason.as_deref(), Some("length"));
        let tool = &result.messages[2];
        assert!(tool
            .content
            .as_deref()
            .is_some_and(|content| content.contains("truncated")));
        assert_eq!(
            tool.tool_result.as_ref().map(|result| &result.details),
            Some(&json!({"status": "truncated", "executed": false}))
        );
    }

    #[tokio::test]
    async fn content_filtered_responses_stop_the_run_explicitly() {
        let mut filtered = Message::assistant(Some("blocked".to_string()), None);
        filtered.stop_reason = Some("content_filter".to_string());
        let agent = AgentLoop::new(
            FakeChatModel { response: filtered },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("content-filter agent should be valid");

        let error = agent
            .run("filtered")
            .await
            .expect_err("content-filtered response should terminate the run");

        assert!(matches!(
            error,
            AgentError::ModelTerminated { reason } if reason == "content_filter"
        ));
    }

    #[tokio::test]
    async fn model_error_finish_reason_is_terminal_failure() {
        let mut failed = Message::assistant(Some("provider failed".to_string()), None);
        failed.stop_reason = Some("error".to_string());
        let agent = AgentLoop::new(
            FakeChatModel { response: failed },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("model-error agent should be valid");

        let error = agent
            .run("handle provider error")
            .await
            .expect_err("provider error finish reason must fail the run");

        assert!(matches!(
            error,
            AgentError::ModelTerminated { reason } if reason == "error"
        ));
    }

    #[tokio::test]
    async fn model_aborted_tool_calls_are_closed_without_execution() {
        let directory = tempdir().expect("temporary directory should exist");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        let mut aborted = Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")]));
        aborted.stop_reason = Some("aborted".to_string());
        let mut tools = ToolRegistry::new();
        tools
            .register(ImmediateTool)
            .expect("echo tool should register");
        let agent = AgentLoop::new(
            FakeChatModel { response: aborted },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("aborted tool agent should be valid")
        .with_session(session.clone());

        let error = agent
            .run("handle aborted tool call")
            .await
            .expect_err("aborted model output must fail the run");
        assert!(matches!(
            error,
            AgentError::ModelTerminated { reason } if reason == "aborted"
        ));

        let records = session.load().await.expect("session should load");
        assert_eq!(records.len(), 3);
        let tool_result = records[2]
            .message
            .tool_result
            .as_ref()
            .expect("aborted tool call should have a synthetic result");
        assert_eq!(tool_result.details["status"], "aborted");
        assert_eq!(tool_result.details["executed"], false);
        assert!(tool_result.is_error);
    }

    #[tokio::test]
    async fn content_filter_closes_tool_calls_before_stopping() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        let mut filtered = Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")]));
        filtered.stop_reason = Some("content_filter".to_string());
        let mut tools = ToolRegistry::new();
        tools
            .register(ImmediateTool)
            .expect("echo tool should register");
        let agent = AgentLoop::new(
            FakeChatModel { response: filtered },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("content-filter tool agent should be valid")
        .with_session(session.clone())
        .with_runtime_ledger_file(ledger_path.clone());

        let error = agent
            .run("filtered tool call")
            .await
            .expect_err("content-filtered tool calls should terminate the run");
        assert!(matches!(
            error,
            AgentError::ModelTerminated { reason } if reason == "content_filter"
        ));

        let records = session.load().await.expect("session should load");
        assert_eq!(records.len(), 3);
        let tool_result = records[2]
            .message
            .tool_result
            .as_ref()
            .expect("content-filtered call should have a result");
        assert_eq!(tool_result.details["status"], "rejected");
        assert_eq!(tool_result.details["executed"], false);

        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        assert!(ledger
            .scan_after(EventSequence::INITIAL)
            .expect("runtime events should be readable")
            .is_empty());
    }

    #[tokio::test]
    async fn after_tool_hook_can_replace_a_tool_result() {
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
            Message::assistant(Some("redacted result handled".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new();
        tools
            .register(ImmediateTool)
            .expect("echo tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("hook agent should be valid")
        .with_after_tool_hook(|context| {
            Box::pin(async move {
                assert_eq!(context.content, "runtime-ok");
                Ok(ToolResultOverride {
                    content: Some("redacted".to_string()),
                    is_error: Some(false),
                    details: None,
                    terminate: None,
                })
            })
        });

        let result = agent.run("call echo").await.expect("run should succeed");

        assert_eq!(result.messages[2].content.as_deref(), Some("redacted"));
    }

    #[tokio::test]
    async fn after_tool_hook_results_are_bounded_before_persistence() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
            Message::assistant(Some("handled".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new().with_output_limit(64);
        tools
            .register(ImmediateTool)
            .expect("echo tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("hook agent should be valid")
        .with_runtime_ledger_file(ledger_path.clone())
        .with_after_tool_hook(|_context| {
            Box::pin(async {
                Ok(ToolResultOverride {
                    content: Some("你好".repeat(100)),
                    is_error: Some(true),
                    details: Some(json!({"payload": "x".repeat(1000)})),
                    terminate: Some(false),
                })
            })
        });

        let result = agent
            .run("call echo")
            .await
            .expect("bounded hook result should complete");

        let tool = &result.messages[2];
        assert!(tool
            .content
            .as_deref()
            .is_some_and(|content| content.len() <= 64));
        assert_eq!(
            tool.tool_result.as_ref().map(|value| value.is_error),
            Some(true)
        );
        assert_eq!(
            tool.tool_result.as_ref().map(|value| &value.details),
            Some(&Value::Null)
        );

        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("runtime events should be readable");
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].draft.kind.as_str(), "action_failed");
    }

    #[tokio::test]
    async fn after_tool_hook_obeys_the_run_deadline() {
        let directory = tempdir().expect("temporary directory should exist");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
            None,
            Some(vec![tool_call("echo_tool", "call-1")]),
        )])));
        let mut tools = ToolRegistry::new();
        tools
            .register(ImmediateTool)
            .expect("echo tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("hook agent should be valid")
        .with_session(session.clone())
        .with_after_tool_hook(|_context| {
            Box::pin(async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(ToolResultOverride::default())
            })
        });

        let error = agent
            .run_with_timeout("call echo", Duration::from_millis(10))
            .await
            .expect_err("a hanging after hook should exceed the run deadline");

        assert!(matches!(error, AgentError::RunTimedOut { .. }));
        let records = session.load().await.expect("session should load");
        assert_eq!(records.len(), 3);
        assert!(records[2]
            .message
            .content
            .as_deref()
            .is_some_and(|content| content.contains("after_tool_call hook timed out")));
    }

    #[tokio::test]
    async fn structured_tool_result_preserves_details_and_can_terminate() {
        let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
            None,
            Some(vec![tool_call("structured_tool", "call-1")]),
        )])));
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut tools = ToolRegistry::new();
        tools
            .register(StructuredTool)
            .expect("structured tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("structured result agent should be valid")
        .with_event_sink({
            let events = events.clone();
            move |event| {
                events
                    .lock()
                    .expect("event mutex should not poison")
                    .push(event)
            }
        });

        let result = agent
            .run("run structured tool")
            .await
            .expect("terminating tool should complete the run");

        assert_eq!(result.final_message.role, "tool");
        assert_eq!(result.final_message.content.as_deref(), Some("structured"));
        let metadata = result
            .final_message
            .tool_result
            .as_ref()
            .expect("structured metadata should be persisted");
        assert_eq!(metadata.details, json!({"source": "test"}));
        assert!(!metadata.is_error);
        assert!(metadata.terminate);
        let events = events.lock().expect("event mutex should not poison");
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolFinished {
                details,
                terminate: true,
                is_error: false,
                ..
            } if details == &json!({"source": "test"})
        )));
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentEvent::ModelStarted { step: 2, .. })));
    }

    #[tokio::test]
    async fn forwards_tool_progress_updates_to_the_event_sink() {
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![tool_call("progress_tool", "call-1")])),
            Message::assistant(Some("finished".to_string()), None),
        ])));
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut tools = ToolRegistry::new();
        tools
            .register(ProgressTool)
            .expect("progress tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("progress agent should be valid")
        .with_event_sink({
            let events = Arc::clone(&events);
            move |event| {
                events
                    .lock()
                    .expect("event mutex should not poison")
                    .push(event)
            }
        });

        agent
            .run("run progress tool")
            .await
            .expect("progress tool should complete");

        let events = events.lock().expect("event mutex should not poison");
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolUpdate {
                call_id,
                content,
                details,
                ..
            } if call_id == "call-1"
                && content == "halfway"
                && details == &json!({"progress": 0.5})
        )));
    }

    #[tokio::test]
    async fn terminate_only_ends_a_tool_batch_when_every_call_terminates() {
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(
                None,
                Some(vec![
                    tool_call("structured_tool", "call-terminate"),
                    tool_call("echo_tool", "call-continue"),
                ]),
            ),
            Message::assistant(Some("batch completed".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new();
        tools
            .register(StructuredTool)
            .expect("structured tool should register");
        tools
            .register(ImmediateTool)
            .expect("echo tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("batch termination agent should be valid");

        let result = agent
            .run("run a mixed tool batch")
            .await
            .expect("a non-terminating call should allow the next model turn");

        assert_eq!(
            result.final_message.content.as_deref(),
            Some("batch completed")
        );
        let tool_messages: Vec<_> = result
            .messages
            .iter()
            .filter(|message| message.role == "tool")
            .collect();
        assert_eq!(tool_messages.len(), 2);
        assert!(tool_messages
            .iter()
            .any(|message| message.tool_call_id.as_deref() == Some("call-terminate")));
        assert!(tool_messages
            .iter()
            .any(|message| message.tool_call_id.as_deref() == Some("call-continue")));
    }

    #[tokio::test]
    async fn accepts_an_injected_chat_model_without_http() {
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("fake response".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("fake model agent should be valid");

        let result = agent
            .run("hello")
            .await
            .expect("fake model run should succeed");

        assert_eq!(result.steps, 1);
        assert_eq!(
            result.final_message.content.as_deref(),
            Some("fake response")
        );
    }

    #[tokio::test]
    async fn after_run_hook_receives_a_successful_transcript_snapshot() {
        let observed = Arc::new(Mutex::new(None));
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("fake response".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("fake model agent should be valid")
        .with_after_run_hook({
            let observed = Arc::clone(&observed);
            move |request| {
                let observed = Arc::clone(&observed);
                Box::pin(async move {
                    *observed
                        .lock()
                        .expect("observation mutex should not poison") = Some(request);
                    Ok(())
                })
            }
        });

        agent
            .run("hello")
            .await
            .expect("fake model run should complete");

        let request = observed
            .lock()
            .expect("observation mutex should not poison")
            .take()
            .expect("after-run hook should be called");
        assert!(request.error.is_none());
        let result = request.result.expect("successful run should have a result");
        assert_eq!(
            result.final_message.content.as_deref(),
            Some("fake response")
        );
        assert!(result
            .messages
            .iter()
            .any(|message| message.role == "user" && message.content.as_deref() == Some("hello")));
    }

    #[tokio::test]
    async fn after_run_hook_runs_after_the_model_deadline() {
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("done".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("fake model agent should be valid")
        .with_after_run_hook(|_request| {
            Box::pin(async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(())
            })
        });

        let result = agent
            .run_with_timeout("hello", Duration::from_millis(10))
            .await
            .expect("after-run persistence should have its own lifetime");

        assert_eq!(result.final_message.content.as_deref(), Some("done"));
    }

    #[tokio::test]
    async fn after_run_hook_runs_after_caller_cancellation() {
        let observed = Arc::new(Mutex::new(None));
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("unused".to_owned()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("cancelled agent should be valid")
        .with_after_run_hook({
            let observed = Arc::clone(&observed);
            move |request| {
                let observed = Arc::clone(&observed);
                Box::pin(async move {
                    *observed
                        .lock()
                        .expect("observation mutex should not poison") = Some(request);
                    Ok(())
                })
            }
        });
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        let error = agent
            .run_with_cancellation("cancel before model", cancellation)
            .await
            .expect_err("cancelled run should fail");
        assert!(matches!(error, AgentError::RunCancelled));
        let request = observed
            .lock()
            .expect("observation mutex should not poison")
            .take()
            .expect("after-run hook should observe cancellation");
        assert_eq!(request.outcome, RunOutcome::Cancelled);
        assert!(request.transcript.is_empty());
    }

    #[tokio::test]
    async fn after_run_hook_receives_failure_and_propagates_hook_errors() {
        let observed = Arc::new(Mutex::new(None));
        let agent = AgentLoop::new(
            FailingChatModel,
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("failing model agent should be valid")
        .with_after_run_hook({
            let observed = Arc::clone(&observed);
            move |request| {
                let observed = Arc::clone(&observed);
                Box::pin(async move {
                    *observed
                        .lock()
                        .expect("observation mutex should not poison") = Some(request);
                    Ok(())
                })
            }
        });

        let error = agent
            .run("hello")
            .await
            .expect_err("model failure should be returned");
        assert!(matches!(
            error,
            AgentError::Model(ModelError::EmptyResponse)
        ));
        let request = observed
            .lock()
            .expect("observation mutex should not poison")
            .take()
            .expect("after-run hook should observe model failure");
        assert!(request.result.is_none());
        assert!(request
            .error
            .as_deref()
            .is_some_and(|error| error.contains("model returned no choices")));

        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("fake response".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("fake model agent should be valid")
        .with_after_run_hook(|_request| Box::pin(async { Err("persist failed".to_owned()) }));
        assert!(matches!(
            agent.run("hello").await,
            Err(AgentError::AfterRunHook(error)) if error == "persist failed"
        ));

        let agent = AgentLoop::new(
            FailingChatModel,
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("failing model agent should be valid")
        .with_after_run_hook(|_request| Box::pin(async { Err("persist failed".to_owned()) }));
        assert!(matches!(
            agent.run("hello").await,
            Err(AgentError::Model(ModelError::EmptyResponse))
        ));
    }

    #[tokio::test]
    async fn after_run_hook_keeps_partial_transcript_after_tool_failure() {
        let observed = Arc::new(Mutex::new(None));
        let mut tools = ToolRegistry::new();
        tools
            .register(ImmediateTool)
            .expect("echo tool should register");
        let agent = AgentLoop::new(
            FailingAfterToolChatModel {
                calls: Arc::new(AtomicUsize::new(0)),
            },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("partial failure agent should be valid")
        .with_after_run_hook({
            let observed = Arc::clone(&observed);
            move |request| {
                let observed = Arc::clone(&observed);
                Box::pin(async move {
                    *observed
                        .lock()
                        .expect("observation mutex should not poison") = Some(request);
                    Ok(())
                })
            }
        });

        let error = agent
            .run("call echo then fail")
            .await
            .expect_err("second model request should fail");
        assert!(matches!(
            error,
            AgentError::Model(ModelError::EmptyResponse)
        ));
        let request = observed
            .lock()
            .expect("observation mutex should not poison")
            .take()
            .expect("after-run hook should observe the failure");
        assert!(request.result.is_none());
        assert_eq!(
            request
                .transcript
                .iter()
                .map(|message| message.role.as_str())
                .collect::<Vec<_>>(),
            vec!["user", "assistant", "tool"]
        );
    }

    #[tokio::test]
    async fn evaluates_and_injects_memory_before_the_model_request() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let emitted_events = Arc::new(Mutex::new(Vec::new()));
        let agent = AgentLoop::new(
            RecordingChatModel {
                response: Message::assistant(Some("answer with memory".to_owned()), None),
                requests: requests.clone(),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("memory hook agent should be valid")
        .with_runtime_ledger_file(ledger_path.clone())
        .with_event_sink({
            let emitted_events = Arc::clone(&emitted_events);
            move |event| {
                emitted_events
                    .lock()
                    .expect("event mutex should not be poisoned")
                    .push(event);
            }
        })
        .with_memory_hook(
            MemoryHookConfig {
                namespace: "bench/text".to_owned(),
                max_tokens: 32,
                ..MemoryHookConfig::default()
            },
            |request| {
                Box::pin(async move {
                    Ok(Some(MemoryProposal {
                        proposal_id: Uuid::from_u128(7),
                        run_id: request.context.run_id,
                        namespace: request.context.namespace.clone(),
                        base_revision: request.context.current_revision,
                        memory_revision: request.context.memory_revision,
                        policy_revision: request.context.policy_revision,
                        candidates: vec![MemoryCandidate {
                            candidate_id: Uuid::from_u128(8),
                            projection: MemoryProjection {
                                projection_id: Uuid::from_u128(9),
                                source_memory_id: Uuid::from_u128(10),
                                source_version: 1,
                                content: "previous parser fix".to_owned(),
                                evidence_refs: Vec::new(),
                                provenance: "text-memory:record-1".to_owned(),
                                token_cost: 5,
                            },
                            score: 0.9,
                            confidence: 0.8,
                            expires_at_ms: None,
                        }],
                        created_at_ms: request.context.now_ms,
                        expires_at_ms: Some(request.context.now_ms + 1_000),
                        retrieval_summary: None,
                    }))
                })
            },
        );

        agent
            .run("fix the parser")
            .await
            .expect("run should succeed");

        let requests = requests
            .lock()
            .expect("request mutex should not be poisoned");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].len(), 2);
        assert_eq!(requests[0][0].role, "system");
        assert!(requests[0][0]
            .content
            .as_deref()
            .is_some_and(|content| content.contains("previous parser fix")));
        assert_eq!(requests[0][1].content.as_deref(), Some("fix the parser"));

        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("memory decision should be readable");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].draft.kind.as_str(), "memory_injected");

        let emitted_events = emitted_events
            .lock()
            .expect("event mutex should not be poisoned");
        assert!(emitted_events.iter().any(|event| matches!(
            event,
            AgentEvent::MemoryDecision {
                step: 1,
                proposal_id,
                event_sequence: 1,
                decision_kind,
                injection_count: 1,
                total_token_cost: 5,
                skip_reason: None,
                ..
            } if *proposal_id == Uuid::from_u128(7) && decision_kind == "injected"
        )));
    }

    #[tokio::test]
    async fn rejects_memory_that_would_exceed_context_before_runtime_audit() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let agent = AgentLoop::new_with_policy(
            RecordingChatModel {
                response: Message::assistant(Some("unused".to_owned()), None),
                requests: Arc::new(Mutex::new(Vec::new())),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                max_context_bytes: 128,
                ..RunPolicy::default()
            },
        )
        .expect("memory budget agent should be valid")
        .with_runtime_ledger_file(ledger_path.clone())
        .with_memory_hook(MemoryHookConfig::default(), |request| {
            Box::pin(async move {
                Ok(Some(MemoryProposal {
                    proposal_id: Uuid::from_u128(17),
                    run_id: request.context.run_id,
                    namespace: request.context.namespace.clone(),
                    base_revision: request.context.current_revision,
                    memory_revision: request.context.memory_revision,
                    policy_revision: request.context.policy_revision,
                    candidates: vec![MemoryCandidate {
                        candidate_id: Uuid::from_u128(18),
                        projection: MemoryProjection {
                            projection_id: Uuid::from_u128(19),
                            source_memory_id: Uuid::from_u128(20),
                            source_version: 1,
                            content: "x".repeat(512),
                            evidence_refs: Vec::new(),
                            provenance: "oversized-memory".to_owned(),
                            token_cost: 128,
                        },
                        score: 1.0,
                        confidence: 1.0,
                        expires_at_ms: None,
                    }],
                    created_at_ms: request.context.now_ms,
                    expires_at_ms: None,
                    retrieval_summary: None,
                }))
            })
        });

        let error = agent
            .run("prompt")
            .await
            .expect_err("oversized memory should fail before the model request");
        assert!(
            matches!(error, AgentError::ContextTooLarge { .. }),
            "{error:?}"
        );

        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("runtime events should be readable");
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn system_prompt_is_ephemeral_and_precedes_session_messages() {
        let directory = tempdir().expect("temporary directory should exist");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let agent = AgentLoop::new(
            RecordingChatModel {
                response: Message::assistant(Some("done".to_owned()), None),
                requests: requests.clone(),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("system prompt agent should be valid")
        .with_system_prompt("You are a careful coding agent.")
        .with_session(session.clone());

        agent
            .run("inspect the project")
            .await
            .expect("run should succeed");

        {
            let requests = requests
                .lock()
                .expect("request mutex should not be poisoned");
            assert_eq!(requests[0][0].role, "system");
            assert_eq!(
                requests[0][0].content.as_deref(),
                Some("You are a careful coding agent.")
            );
            assert_eq!(
                requests[0][1].content.as_deref(),
                Some("inspect the project")
            );
        }

        let records = session.load().await.expect("session should load");
        assert!(records.iter().all(|record| {
            record.message.content.as_deref() != Some("You are a careful coding agent.")
        }));
    }

    #[tokio::test]
    async fn rejects_a_second_run_while_the_agent_is_active() {
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let agent = Arc::new(
            AgentLoop::new(
                BlockingChatModel {
                    started: started.clone(),
                    release: release.clone(),
                    response: Message::assistant(Some("done".to_string()), None),
                },
                ToolRegistry::new(),
                "blocking-model",
                None,
                ChatOptions {
                    temperature: 0.0,
                    enable_thinking: Some(false),
                },
                1,
            )
            .expect("blocking agent should be valid"),
        );
        let first_agent = Arc::clone(&agent);
        let first = tokio::spawn(async move { first_agent.run("first").await });
        started.notified().await;

        let error = agent
            .run("second")
            .await
            .expect_err("a concurrent run should be rejected");
        assert!(matches!(error, AgentError::RunBusy));

        release.notify_one();
        first
            .await
            .expect("first run task should join")
            .expect("first run should complete");
    }

    #[tokio::test]
    async fn steering_is_injected_before_the_next_model_request() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let agent = AgentLoop::new(
            RecordingChatModel {
                response: Message::assistant(Some("answer".to_string()), None),
                requests: requests.clone(),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("steering agent should be valid");
        agent
            .steer("additional constraint")
            .expect("steering should queue");

        agent.run("initial task").await.expect("run should succeed");

        let requests = requests.lock().expect("request mutex should not poison");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0][0].content.as_deref(), Some("initial task"));
        assert_eq!(
            requests[0][1].content.as_deref(),
            Some("additional constraint")
        );
    }

    #[tokio::test]
    async fn follow_up_starts_after_a_natural_assistant_stop() {
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(Some("first answer".to_string()), None),
            Message::assistant(Some("follow-up answer".to_string()), None),
        ])));
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            3,
        )
        .expect("follow-up agent should be valid");
        agent
            .follow_up("continue with a second task")
            .expect("follow-up should queue");

        let result = agent.run("first task").await.expect("run should succeed");

        assert_eq!(result.steps, 2);
        assert_eq!(
            result.final_message.content.as_deref(),
            Some("follow-up answer")
        );
        assert_eq!(result.messages.len(), 4);
        assert_eq!(
            result.messages[2].content.as_deref(),
            Some("continue with a second task")
        );
    }

    #[tokio::test]
    async fn rejects_idle_control_messages_after_the_first_run() {
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("done".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("agent should be valid");

        agent
            .run("first task")
            .await
            .expect("first run should succeed");

        assert!(matches!(
            agent.steer("stale steering"),
            Err(AgentError::ControlQueueRequiresActiveRun { kind: "steering" })
        ));
        assert!(matches!(
            agent.follow_up("stale follow-up"),
            Err(AgentError::ControlQueueRequiresActiveRun { kind: "follow-up" })
        ));
    }

    #[tokio::test]
    async fn clears_unconsumed_control_messages_when_a_run_finishes() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let agent = AgentLoop::new(
            RecordingChatModel {
                response: Message::assistant(Some("done".to_string()), None),
                requests: Arc::clone(&requests),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("agent should be valid");
        agent
            .follow_up("stale follow-up")
            .expect("initial follow-up should be queueable");

        agent
            .run("first task")
            .await
            .expect("first run should succeed");
        agent
            .run("second task")
            .await
            .expect("second run should succeed");

        let requests = requests.lock().expect("request mutex should not poison");
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].len(), 1);
        assert_eq!(requests[1].len(), 1);
        assert_eq!(requests[1][0].content.as_deref(), Some("second task"));
    }

    #[tokio::test]
    async fn retries_transient_model_failures_without_duplicate_deltas() {
        let calls = Arc::new(AtomicUsize::new(0));
        let events = Arc::new(Mutex::new(Vec::new()));
        let agent = AgentLoop::new_with_policy(
            RetryChatModel {
                calls: calls.clone(),
                response: Message::assistant(Some("recovered".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                max_model_retries: 1,
                model_retry_backoff_ms: 0,
                ..RunPolicy::default()
            },
        )
        .expect("retry policy should be valid")
        .with_event_sink({
            let events = events.clone();
            move |event| {
                events
                    .lock()
                    .expect("event mutex should not poison")
                    .push(event)
            }
        });

        let result = agent.run("hello").await.expect("retry should recover");

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(result.final_message.content.as_deref(), Some("recovered"));
        let events = events.lock().expect("event mutex should not poison");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::AssistantDelta { .. }))
                .count(),
            1
        );
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ModelRetry {
                attempt: 1,
                delay_ms: 0,
                ..
            }
        )));
    }

    #[tokio::test]
    async fn resets_partial_stream_output_before_retrying() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let calls = Arc::new(AtomicUsize::new(0));
        let agent = AgentLoop::new_with_policy(
            PartialRetryChatModel {
                calls: calls.clone(),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(true),
            },
            RunPolicy {
                max_model_retries: 1,
                model_retry_backoff_ms: 0,
                ..RunPolicy::default()
            },
        )
        .expect("partial retry agent should be valid")
        .with_event_sink({
            let events = events.clone();
            move |event| {
                events
                    .lock()
                    .expect("event mutex should not be poisoned")
                    .push(event)
            }
        });

        let result = agent
            .run("stream a response")
            .await
            .expect("retry should recover after resetting partial output");

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(result.final_message.content.as_deref(), Some("Hello"));
        let events = events.lock().expect("event mutex should not be poisoned");
        let stream_events = events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    AgentEvent::AssistantDelta { .. }
                        | AgentEvent::ReasoningDelta { .. }
                        | AgentEvent::AssistantStreamReset { .. }
                        | AgentEvent::ModelRetry { .. }
                )
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            stream_events.as_slice(),
            [
                AgentEvent::ReasoningDelta { content, .. },
                AgentEvent::AssistantDelta { content: first, .. },
                AgentEvent::AssistantStreamReset { attempt: 1, .. },
                AgentEvent::ModelRetry { attempt: 1, .. },
                AgentEvent::AssistantDelta { content: second, .. },
            ] if content == "draft" && first == "Hel" && second == "Hello"
        ));
        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::AssistantMessage { .. })));
    }

    #[tokio::test]
    async fn emits_stream_delta_before_model_completion() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let agent = AgentLoop::new(
            GatedStreamingChatModel {
                started: started.clone(),
                release: release.clone(),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("streaming agent should be valid")
        .with_event_sink({
            let events = events.clone();
            move |event| {
                events
                    .lock()
                    .expect("event mutex should not be poisoned")
                    .push(event)
            }
        });

        let task = tokio::spawn(async move { agent.run("stream now").await });
        started.notified().await;
        assert!(events.lock().expect("event mutex should not be poisoned").iter().any(
            |event| matches!(event, AgentEvent::AssistantDelta { content, .. } if content == "partial")
        ));

        release.notify_one();
        task.await
            .expect("streaming task should join")
            .expect("streaming run should succeed");
    }

    #[tokio::test]
    async fn tool_execution_passes_through_runtime_admission() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
            Message::assistant(Some("tool completed".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new();
        tools
            .register(ImmediateTool)
            .expect("immediate tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("agent should be valid")
        .with_runtime_ledger_file(ledger_path.clone());

        let result = agent
            .run("use the echo tool")
            .await
            .expect("runtime-admitted tool run should succeed");

        assert_eq!(result.messages[2].content.as_deref(), Some("runtime-ok"));
        let metadata = result.messages[2]
            .tool_result
            .as_ref()
            .expect("runtime metadata should be persisted");
        assert!(metadata.action_id.is_some());
        assert!(metadata.grant_id.is_some());
        assert_eq!(
            result.final_message.content.as_deref(),
            Some("tool completed")
        );

        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("runtime events should be readable");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].draft.kind.as_str(), "action_admitted");
        assert_eq!(events[1].draft.kind.as_str(), "action_succeeded");
    }

    #[tokio::test]
    async fn structured_tool_errors_are_recorded_as_failed_runtime_actions() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![tool_call("error_tool", "call-1")])),
            Message::assistant(Some("recovered".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new();
        tools
            .register(ErrorResultTool)
            .expect("error tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("agent should be valid")
        .with_runtime_ledger_file(ledger_path.clone());

        let result = agent
            .run("use the error tool")
            .await
            .expect("structured tool errors should return to the model");
        assert_eq!(result.final_message.content.as_deref(), Some("recovered"));
        assert!(result.messages[2]
            .tool_result
            .as_ref()
            .is_some_and(|metadata| metadata.is_error));

        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("runtime events should be readable");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].draft.kind.as_str(), "action_admitted");
        assert_eq!(events[1].draft.kind.as_str(), "action_failed");
        assert_eq!(events[1].draft.payload["error"], "known failure");
    }

    #[tokio::test]
    async fn uncertain_structured_tool_results_remain_pending_for_reconciliation() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![tool_call("uncertain_tool", "call-1")])),
            Message::assistant(Some("must not run".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new();
        tools
            .register(UncertainResultTool)
            .expect("uncertain tool should register");
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("agent should be valid")
        .with_runtime_ledger_file(ledger_path.clone());

        let error = agent
            .run("use the uncertain tool")
            .await
            .expect_err("uncertain outcome must stop the run");
        assert!(matches!(error, AgentError::RuntimeActionUnknown { .. }));

        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("runtime events should be readable");
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].draft.kind.as_str(), "action_unknown");
    }

    #[tokio::test]
    async fn reconciled_runtime_results_are_projected_into_the_next_model_context() {
        let directory = tempdir().expect("temporary directory should exist");
        let session_path = directory.path().join("conversation.jsonl");
        let ledger_path = directory.path().join("runtime.jsonl");
        let session_id = Uuid::new_v4();
        let session = SessionStore::open(&session_path, session_id)
            .await
            .expect("session should open");
        let mut tools = ToolRegistry::new();
        tools
            .register(UncertainResultTool)
            .expect("uncertain tool should register");
        let first = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(
                    None,
                    Some(vec![tool_call("uncertain_tool", "call-1")]),
                ),
            },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("initial agent should be valid")
        .with_session(session)
        .with_runtime_ledger_file(ledger_path.clone());

        first
            .run("run an uncertain tool")
            .await
            .expect_err("uncertain execution should remain pending");
        let action_id = first
            .pending_runtime_actions()
            .expect("pending action should be visible")
            .first()
            .expect("one action should be pending")
            .action_id;
        first
            .reconcile_runtime_action(
                action_id,
                RuntimeReconcileOutcome::Succeeded {
                    output: json!("confirmed output"),
                    evidence: "provider-check-2".to_string(),
                },
            )
            .expect("reconciliation should be persisted");

        let requests = Arc::new(Mutex::new(Vec::new()));
        let reopened = SessionStore::open(&session_path, session_id)
            .await
            .expect("session should reopen");
        let second = AgentLoop::new(
            RecordingChatModel {
                response: Message::assistant(Some("continued".to_string()), None),
                requests: requests.clone(),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("recovery agent should be valid")
        .with_session(reopened)
        .with_runtime_ledger_file(ledger_path)
        .with_runtime_recovery();
        second
            .run("continue after reconciliation")
            .await
            .expect("reconciled context should resume");

        let requests = requests
            .lock()
            .expect("request mutex should not be poisoned");
        let tool_message = requests[0]
            .iter()
            .find(|message| message.role == "tool")
            .expect("the model request should include the reconciled tool result");
        assert_eq!(tool_message.content.as_deref(), Some("confirmed output"));
        assert_eq!(
            tool_message
                .tool_result
                .as_ref()
                .map(|metadata| metadata.details["status"].as_str()),
            Some(Some("reconciled"))
        );
    }

    #[tokio::test]
    async fn fresh_runtime_run_accepts_a_completed_ledger_without_recovery() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
            Message::assistant(Some("first run".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new();
        tools
            .register(ImmediateTool)
            .expect("immediate tool should register");
        AgentLoop::new(
            SequenceChatModel { responses },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("first agent should be valid")
        .with_runtime_ledger_file(ledger_path.clone())
        .run("first turn")
        .await
        .expect("first run should succeed");

        let second = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("second run".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("second agent should be valid")
        .with_runtime_ledger_file(ledger_path.clone());

        let result = second
            .run("second turn")
            .await
            .expect("completed old ledger must not be treated as a fresh-run mismatch");

        assert_eq!(result.final_message.content.as_deref(), Some("second run"));
    }

    #[tokio::test]
    async fn runtime_recovery_resumes_the_latest_run_revision() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");

        let first_responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
            Message::assistant(Some("first run".to_string()), None),
        ])));
        let mut first_tools = ToolRegistry::new();
        first_tools
            .register(ImmediateTool)
            .expect("immediate tool should register");
        let first = AgentLoop::new(
            SequenceChatModel {
                responses: first_responses,
            },
            first_tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("first agent should be valid")
        .with_runtime_ledger_file(ledger_path.clone());
        first
            .run("first turn")
            .await
            .expect("first run should succeed");

        let second_responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(None, Some(vec![tool_call("echo_tool", "call-2")])),
            Message::assistant(Some("resumed run".to_string()), None),
        ])));
        let mut second_tools = ToolRegistry::new();
        second_tools
            .register(ImmediateTool)
            .expect("immediate tool should register");
        let started_run_id = Arc::new(Mutex::new(None));
        let started_run_id_sink = Arc::clone(&started_run_id);
        let after_run_id = Arc::new(Mutex::new(None));
        let after_run_id_hook = Arc::clone(&after_run_id);
        let second = AgentLoop::new(
            SequenceChatModel {
                responses: second_responses,
            },
            second_tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("second agent should be valid")
        .with_runtime_ledger_file(ledger_path.clone())
        .with_runtime_recovery()
        .with_event_sink(move |event| {
            if let AgentEvent::RunStarted { context } = event {
                *started_run_id_sink
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(context.run_id);
            }
        })
        .with_memory_hook(
            MemoryHookConfig {
                namespace: "recovery-test".to_owned(),
                max_tokens: 32,
                ..MemoryHookConfig::default()
            },
            recovery_memory_hook,
        )
        .with_after_run_hook(move |request| {
            *after_run_id_hook
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(request.context.run_id);
            Box::pin(async { Ok(()) })
        });
        second
            .run("resumed turn")
            .await
            .expect("resumed run should succeed");

        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("runtime events should be readable");
        assert_eq!(events.len(), 6);
        assert_eq!(events[0].draft.kind.as_str(), "action_admitted");
        assert_eq!(events[1].draft.kind.as_str(), "action_succeeded");
        assert_eq!(events[2].draft.kind.as_str(), "memory_injected");
        assert_eq!(events[3].draft.kind.as_str(), "action_admitted");
        assert_eq!(events[4].draft.kind.as_str(), "action_succeeded");
        assert_eq!(events[5].draft.kind.as_str(), "memory_injected");
        assert_eq!(events[0].draft.run_id, events[3].draft.run_id);
        assert_eq!(events[3].draft.revision, pi_agent_runtime::Revision::new(1));
        assert_eq!(events[4].draft.revision, pi_agent_runtime::Revision::new(2));
        let recovered_run_id = events[3].draft.run_id.as_uuid();
        assert_eq!(
            *started_run_id
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            Some(recovered_run_id)
        );
        assert_eq!(
            *after_run_id
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            Some(recovered_run_id)
        );
    }

    #[tokio::test]
    async fn injects_skill_index_into_model_context() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let mut catalog = SkillCatalog::new();
        catalog
            .insert(
                Skill::new(
                    "research",
                    "Research tasks",
                    "Use primary sources.",
                    "/skills/research/SKILL.md",
                )
                .expect("test skill should be valid"),
            )
            .expect("test catalog should accept the skill");

        let agent = AgentLoop::new(
            RecordingChatModel {
                response: Message::assistant(Some("done".to_string()), None),
                requests: Arc::clone(&requests),
            },
            ToolRegistry::new(),
            "recording-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("recording model agent should be valid")
        .with_skills(catalog);

        assert!(agent
            .tools
            .model_definitions()
            .iter()
            .any(|definition| definition.function.name == "read_skill"));

        agent
            .run("find evidence")
            .await
            .expect("run should succeed");

        let requests = requests
            .lock()
            .expect("request mutex should not be poisoned");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0][0].role, "system");
        assert!(requests[0][0]
            .content
            .as_deref()
            .is_some_and(|content| content.contains("<name>research</name>")));
        assert_eq!(
            requests[0].last().map(|message| message.role.as_str()),
            Some("user")
        );
    }

    #[tokio::test]
    async fn steering_prompt_is_injected_before_the_next_model_request() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let agent = AgentLoop::new(
            RecordingChatModel {
                response: Message::assistant(Some("done".to_owned()), None),
                requests: Arc::clone(&requests),
            },
            ToolRegistry::new(),
            "recording-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("recording model agent should be valid");
        agent
            .steer("stop and inspect the result")
            .expect("steering prompt should be queued");

        agent
            .run("original request")
            .await
            .expect("run should succeed");

        let requests = requests
            .lock()
            .expect("request mutex should not be poisoned");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0][0].content.as_deref(), Some("original request"));
        assert_eq!(
            requests[0][1].content.as_deref(),
            Some("stop and inspect the result")
        );
    }

    #[tokio::test]
    async fn follow_up_prompt_runs_after_the_current_assistant_turn() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let agent = AgentLoop::new(
            RecordingChatModel {
                response: Message::assistant(Some("done".to_owned()), None),
                requests: Arc::clone(&requests),
            },
            ToolRegistry::new(),
            "recording-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("recording model agent should be valid");
        agent
            .follow_up("now summarize it")
            .expect("follow-up prompt should be queued");

        let result = agent
            .run("original request")
            .await
            .expect("run should succeed");

        let requests = requests
            .lock()
            .expect("request mutex should not be poisoned");
        assert_eq!(result.steps, 2);
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].len(), 1);
        assert_eq!(requests[0][0].content.as_deref(), Some("original request"));
        assert_eq!(requests[1].len(), 3);
        assert_eq!(requests[1][2].content.as_deref(), Some("now summarize it"));
    }

    #[tokio::test]
    async fn timeout_cancels_model_and_emits_one_cancelled_terminal_event() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = Arc::clone(&events);
        let agent = AgentLoop::new(
            SlowChatModel {
                delay: Duration::from_millis(100),
                response: Message::assistant(Some("too late".to_string()), None),
            },
            ToolRegistry::new(),
            "slow-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("slow model agent should be valid")
        .with_event_sink(move |event| {
            captured_events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event);
        });

        let error = agent
            .run_with_timeout("hello", Duration::from_millis(10))
            .await
            .expect_err("slow model should exceed the run deadline");

        assert!(matches!(error, AgentError::RunTimedOut { timeout_ms: 10 }));
        let events = events.lock().expect("event mutex should not be poisoned");
        assert_eq!(events.len(), 4);
        assert!(matches!(events[0], AgentEvent::RunStarted { .. }));
        assert!(matches!(events[1], AgentEvent::TurnStarted { step: 1, .. }));
        assert!(matches!(
            events[2],
            AgentEvent::ModelStarted { step: 1, .. }
        ));
        assert!(matches!(
            events[3],
            AgentEvent::RunFinished {
                outcome: RunOutcome::Cancelled,
                steps: 1,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn caller_cancellation_stops_model_and_emits_cancelled_terminal_event() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = Arc::clone(&events);
        let agent = AgentLoop::new(
            SlowChatModel {
                delay: Duration::from_millis(100),
                response: Message::assistant(Some("too late".to_string()), None),
            },
            ToolRegistry::new(),
            "slow-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("slow model agent should be valid")
        .with_event_sink(move |event| {
            captured_events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event);
        });
        let cancellation = CancellationToken::new();
        let run_cancellation = cancellation.clone();
        let task =
            tokio::spawn(
                async move { agent.run_with_cancellation("hello", run_cancellation).await },
            );

        tokio::time::sleep(Duration::from_millis(10)).await;
        cancellation.cancel();
        let error = task
            .await
            .expect("agent task should join")
            .expect_err("caller cancellation should stop the run");

        assert!(matches!(error, AgentError::RunCancelled));
        let events = events.lock().expect("event mutex should not be poisoned");
        assert!(matches!(
            events.last(),
            Some(AgentEvent::RunFinished {
                outcome: RunOutcome::Cancelled,
                steps: 1,
                error: Some(error),
                ..
            }) if error.contains("cancelled by the caller")
        ));
    }

    #[tokio::test]
    async fn empty_prompt_is_rejected_before_an_immediate_deadline() {
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("unused".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("fake model agent should be valid");

        let error = agent
            .run_with_timeout("  ", Duration::ZERO)
            .await
            .expect_err("empty prompt should be rejected");

        assert!(matches!(error, AgentError::EmptyPrompt));
    }

    #[tokio::test]
    async fn timeout_persists_cancelled_results_for_pending_tools() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        let events = Arc::new(Mutex::new(Vec::new()));
        let after_run_outcome = Arc::new(Mutex::new(None));
        let mut tools = ToolRegistry::new();
        tools.register(SlowTool).expect("slow tool should register");
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(None, Some(vec![tool_call("slow_tool", "call-1")])),
            },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("tool timeout agent should be valid")
        .with_session(session.clone())
        .with_runtime_ledger_file(ledger_path.clone())
        .with_event_sink({
            let events = Arc::clone(&events);
            move |event| {
                events
                    .lock()
                    .expect("event mutex should not be poisoned")
                    .push(event);
            }
        })
        .with_after_run_hook({
            let after_run_outcome = Arc::clone(&after_run_outcome);
            move |request| {
                let after_run_outcome = Arc::clone(&after_run_outcome);
                Box::pin(async move {
                    *after_run_outcome
                        .lock()
                        .expect("outcome mutex should not be poisoned") = Some(request.outcome);
                    Ok(())
                })
            }
        });

        let error = agent
            .run_with_timeout("use the slow tool", Duration::from_millis(10))
            .await
            .expect_err("slow tool should exceed the run deadline");

        assert!(matches!(error, AgentError::RunTimedOut { .. }));
        assert_eq!(
            *after_run_outcome
                .lock()
                .expect("outcome mutex should not be poisoned"),
            Some(RunOutcome::Unknown)
        );
        assert!(matches!(
            events
                .lock()
                .expect("event mutex should not be poisoned")
                .last(),
            Some(AgentEvent::RunFinished {
                outcome: RunOutcome::Unknown,
                ..
            })
        ));
        let records = session.load().await.expect("session should load");
        assert_eq!(records.len(), 3);
        assert_eq!(records[1].message.role, "assistant");
        assert_eq!(records[2].message.role, "tool");
        assert!(records[2]
            .message
            .content
            .as_deref()
            .is_some_and(|content| content.contains("cancelled")));
        assert!(records[2]
            .message
            .tool_result
            .as_ref()
            .is_some_and(|metadata| metadata.is_error && !metadata.terminate));
        assert!(records[2]
            .message
            .tool_result
            .as_ref()
            .is_some_and(|metadata| metadata.action_id.is_some() && metadata.grant_id.is_some()));
        assert_eq!(
            records[2]
                .message
                .tool_result
                .as_ref()
                .and_then(|metadata| metadata.details.get("status"))
                .and_then(Value::as_str),
            Some("unknown")
        );
        assert_eq!(
            records[2]
                .message
                .tool_result
                .as_ref()
                .and_then(|metadata| metadata.details.get("executed"))
                .and_then(Value::as_bool),
            Some(true)
        );

        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("runtime events should be readable");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].draft.kind.as_str(), "action_admitted");
        assert_eq!(events[1].draft.kind.as_str(), "action_unknown");
    }

    #[tokio::test]
    async fn admitted_caller_cancellation_reports_unknown_run_outcome() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        let events = Arc::new(Mutex::new(Vec::new()));
        let after_run_outcome = Arc::new(Mutex::new(None));
        let mut tools = ToolRegistry::new();
        tools
            .register(CancelInsideTool)
            .expect("cancellation tool should register");
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(
                    None,
                    Some(vec![tool_call("cancel_inside_tool", "call-1")]),
                ),
            },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("cancellation agent should be valid")
        .with_session(session.clone())
        .with_runtime_ledger_file(ledger_path.clone())
        .with_event_sink({
            let events = Arc::clone(&events);
            move |event| {
                events
                    .lock()
                    .expect("event mutex should not be poisoned")
                    .push(event);
            }
        })
        .with_after_run_hook({
            let after_run_outcome = Arc::clone(&after_run_outcome);
            move |request| {
                let after_run_outcome = Arc::clone(&after_run_outcome);
                Box::pin(async move {
                    *after_run_outcome
                        .lock()
                        .expect("outcome mutex should not be poisoned") = Some(request.outcome);
                    Ok(())
                })
            }
        });

        let cancellation = CancellationToken::new();
        let error = agent
            .run_with_cancellation("run the cancellation tool", cancellation)
            .await
            .expect_err("admitted cancellation should stop the run");

        assert!(matches!(error, AgentError::RunCancelled));
        assert_eq!(
            *after_run_outcome
                .lock()
                .expect("outcome mutex should not be poisoned"),
            Some(RunOutcome::Unknown)
        );
        assert!(matches!(
            events
                .lock()
                .expect("event mutex should not be poisoned")
                .last(),
            Some(AgentEvent::RunFinished {
                outcome: RunOutcome::Unknown,
                ..
            })
        ));

        let records = session.load().await.expect("session should load");
        let metadata = records[2]
            .message
            .tool_result
            .as_ref()
            .expect("cancelled tool should retain metadata");
        assert_eq!(metadata.details["status"], "unknown");
        assert_eq!(metadata.details["executed"], true);
        assert!(metadata.action_id.is_some() && metadata.grant_id.is_some());

        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("runtime events should be readable");
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].draft.kind.as_str(), "action_unknown");
    }

    #[tokio::test]
    async fn tool_timeout_stops_before_requesting_another_model_turn() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Message::assistant(
                None,
                Some(vec![tool_call("tool_timeout_tool", "call-timeout")]),
            ),
            Message::assistant(Some("this turn must not run".to_string()), None),
        ])));
        let mut tools = ToolRegistry::new();
        tools
            .register(ToolTimeoutTool)
            .expect("timeout tool should register");
        let agent = AgentLoop::new_with_policy(
            SequenceChatModel {
                responses: responses.clone(),
            },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                tool_timeout_secs: 1,
                ..RunPolicy::default()
            },
        )
        .expect("timeout agent should be valid")
        .with_session(session.clone())
        .with_runtime_ledger_file(ledger_path.clone());

        let error = agent
            .run("use the timeout tool")
            .await
            .expect_err("unknown tool outcome must stop the run");

        assert!(matches!(
            error,
            AgentError::RuntimeActionUnknown {
                tool_name,
                reason,
                ..
            } if tool_name == "tool_timeout_tool" && reason.contains("outcome is unknown")
        ));
        assert_eq!(
            responses
                .lock()
                .expect("response mutex should not poison")
                .len(),
            1,
            "the next model response must remain unused"
        );

        let records = session.load().await.expect("session should load");
        assert_eq!(records.len(), 3);
        assert_eq!(records[2].message.role, "tool");
        assert_eq!(
            records[2]
                .message
                .tool_result
                .as_ref()
                .map(|result| &result.details),
            Some(&serde_json::json!({
                "status": "unknown",
                "executed": true,
            }))
        );

        let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
        let events = ledger
            .scan_after(EventSequence::INITIAL)
            .expect("runtime events should be readable");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].draft.kind.as_str(), "action_admitted");
        assert_eq!(events[1].draft.kind.as_str(), "action_unknown");
        drop(ledger);
        assert_eq!(
            agent
                .pending_runtime_actions()
                .expect("pending action should be inspectable")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn exposes_and_reconciles_an_unknown_runtime_action() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let mut tools = ToolRegistry::new();
        tools.register(SlowTool).expect("slow tool should register");
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(None, Some(vec![tool_call("slow_tool", "call-1")])),
            },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("agent should be valid")
        .with_runtime_ledger_file(ledger_path);

        let error = agent
            .run_with_timeout("use the slow tool", Duration::from_millis(10))
            .await
            .expect_err("tool should exceed the run deadline");
        assert!(matches!(error, AgentError::RunTimedOut { .. }));

        let pending = agent
            .pending_runtime_actions()
            .expect("pending action should be inspectable");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].tool_name, "slow_tool");
        assert_eq!(pending[0].replay_policy, ToolReplayPolicy::Never);
        let action_id = pending[0].action_id;

        agent
            .reconcile_runtime_action(
                action_id,
                RuntimeReconcileOutcome::Failed {
                    error: "provider confirmed no execution".to_string(),
                    evidence: "provider-check-1".to_string(),
                },
            )
            .expect("external evidence should reconcile the action");
        assert!(agent
            .pending_runtime_actions()
            .expect("pending actions should be inspectable after reconciliation")
            .is_empty());
    }

    #[tokio::test]
    async fn runtime_recovery_refuses_unresolved_actions() {
        let directory = tempdir().expect("temporary directory should exist");
        let ledger_path = directory.path().join("runtime.jsonl");
        let mut tools = ToolRegistry::new();
        tools.register(SlowTool).expect("slow tool should register");
        AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(None, Some(vec![tool_call("slow_tool", "call-1")])),
            },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("initial agent should be valid")
        .with_runtime_ledger_file(ledger_path.clone())
        .run_with_timeout("create an unresolved action", Duration::from_millis(10))
        .await
        .expect_err("the initial action should become unresolved");

        let mut recovery_tools = ToolRegistry::new();
        recovery_tools
            .register(SlowTool)
            .expect("recovery tool should register");
        let recovery = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("must not run".to_string()), None),
            },
            recovery_tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("recovery agent should be valid")
        .with_runtime_ledger_file(ledger_path)
        .with_runtime_recovery();

        let error = recovery
            .run("resume after the interrupted action")
            .await
            .expect_err("recovery must require reconciliation first");
        assert!(matches!(
            error,
            AgentError::Runtime(message) if message.contains("unresolved actions")
        ));
    }

    #[tokio::test]
    async fn rejects_session_runtime_identity_without_ledger_evidence() {
        let directory = tempdir().expect("temporary directory should exist");
        let session_id = Uuid::new_v4();
        let session = SessionStore::open(directory.path().join("session.jsonl"), session_id)
            .await
            .expect("session should open");
        session
            .append(&Message::assistant(
                None,
                Some(vec![tool_call("echo_tool", "call-1")]),
            ))
            .await
            .expect("assistant tool call should append");
        session
            .append(&Message::tool_with_result(
                "call-1",
                "stale result",
                ToolResultMetadata::new(Value::Null, false, false)
                    .with_runtime_identity(Uuid::new_v4(), Uuid::new_v4()),
            ))
            .await
            .expect("stale tool result should append");

        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("must not run".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("agent should be valid")
        .with_session(session)
        .with_runtime_ledger_file(directory.path().join("runtime.jsonl"));

        let error = agent
            .run("resume stale transcript")
            .await
            .expect_err("stale runtime identity must fail closed");
        assert!(matches!(
            error,
            AgentError::RuntimeSessionMismatch { reason, .. }
                if reason == "durable admission event is missing"
        ));
    }

    #[tokio::test]
    async fn rejects_anonymous_uncertain_tool_results_with_runtime_ledger() {
        let directory = tempdir().expect("temporary directory should exist");
        let session_id = Uuid::new_v4();
        let session = SessionStore::open(directory.path().join("session.jsonl"), session_id)
            .await
            .expect("session should open");
        session
            .append(&Message::assistant(
                None,
                Some(vec![tool_call("echo_tool", "call-1")]),
            ))
            .await
            .expect("assistant tool call should append");
        session
            .append(&Message::tool_with_result(
                "call-1",
                "recovered unknown result",
                ToolResultMetadata::new(json!({"status": "unknown"}), true, false),
            ))
            .await
            .expect("anonymous uncertain result should append");

        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("must not run".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("agent should be valid")
        .with_session(session)
        .with_runtime_ledger_file(directory.path().join("runtime.jsonl"));

        let error = agent
            .run("resume anonymous unknown result")
            .await
            .expect_err("anonymous uncertain result must fail closed");
        assert!(matches!(
            error,
            AgentError::UnidentifiedToolOutcome { call_id, status }
                if call_id == "call-1" && status == "unknown"
        ));
    }

    #[tokio::test]
    async fn caller_cancellation_persists_cancelled_results_for_pending_tools() {
        let directory = tempdir().expect("temporary directory should exist");
        let session_path = directory.path().join("conversation.jsonl");
        let session_id = Uuid::new_v4();
        let session = SessionStore::open(&session_path, session_id)
            .await
            .expect("session should open");
        let ledger_path = directory.path().join("runtime.jsonl");
        let mut tools = ToolRegistry::new();
        tools.register(SlowTool).expect("slow tool should register");
        let cancellation = CancellationToken::new();
        let cancel_on_tool_start = cancellation.clone();
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(None, Some(vec![tool_call("slow_tool", "call-1")])),
            },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("tool cancellation agent should be valid")
        .with_session(session.clone())
        .with_runtime_ledger_file(ledger_path.clone())
        .with_event_sink(move |event| {
            if matches!(event, AgentEvent::ToolStarted { .. }) {
                cancel_on_tool_start.cancel();
            }
        });
        let run_cancellation = cancellation.clone();
        let task = tokio::spawn(async move {
            agent
                .run_with_cancellation("use the slow tool", run_cancellation)
                .await
        });

        let error = task
            .await
            .expect("agent task should join")
            .expect_err("caller cancellation should stop the tool");

        assert!(matches!(error, AgentError::RunCancelled));
        let records = session.load().await.expect("session should load");
        assert_eq!(records.len(), 3);
        assert_eq!(records[1].message.role, "assistant");
        assert_eq!(records[2].message.role, "tool");
        assert_eq!(
            records[2].message.content.as_deref(),
            Some("Tool execution cancelled by the caller")
        );

        let reopened_session = SessionStore::open(&session_path, session_id)
            .await
            .expect("the cancelled session should reopen");
        let resumed = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("resumed".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("resume agent should be valid")
        .with_session(reopened_session)
        .with_runtime_ledger_file(ledger_path);

        let result = resumed
            .run("resume after cancellation")
            .await
            .expect("an unexecuted cancellation should not block recovery");
        assert_eq!(result.final_message.content.as_deref(), Some("resumed"));
    }

    #[tokio::test]
    async fn agent_loads_a_repaired_tail_before_requesting_the_next_turn() {
        let directory = tempdir().expect("temporary directory should exist");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        session
            .append(&Message::user("previous question"))
            .await
            .expect("previous user message should append");
        session
            .append(&Message::assistant(
                None,
                Some(vec![tool_call("read_file", "orphan-call")]),
            ))
            .await
            .expect("orphan assistant call should append");

        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("recovered context".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("agent should be valid")
        .with_session(session.clone());

        let result = agent
            .run("new question")
            .await
            .expect("repaired transcript should be accepted");

        assert_eq!(result.messages[2].role, "tool");
        assert_eq!(
            result.messages[2].tool_call_id.as_deref(),
            Some("orphan-call")
        );
        assert_eq!(
            result.final_message.content.as_deref(),
            Some("recovered context")
        );
        assert_eq!(
            session.load().await.expect("session should reload").len(),
            5
        );
    }

    #[tokio::test]
    async fn emits_reasoning_delta_events_from_thinking_models() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let agent = AgentLoop::new(
            ReasoningChatModel,
            ToolRegistry::new(),
            "thinking-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(true),
            },
            1,
        )
        .expect("reasoning agent should be valid")
        .with_event_sink({
            let events = Arc::clone(&events);
            move |event| {
                events
                    .lock()
                    .expect("event mutex should not poison")
                    .push(event)
            }
        });

        let result = agent.run("think").await.expect("run should succeed");

        assert_eq!(
            result.final_message.reasoning_content.as_deref(),
            Some("thinking")
        );
        let events = events.lock().expect("event mutex should not poison");
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ReasoningDelta { content, step: 1, .. } if content == "thinking"
        )));
    }

    #[tokio::test]
    async fn emits_lifecycle_events_for_a_simple_run() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = Arc::clone(&events);
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("event response".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("fake model agent should be valid")
        .with_event_sink(move |event| {
            captured_events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event);
        });

        agent.run("hello").await.expect("run should succeed");

        let events = events.lock().expect("event mutex should not be poisoned");
        assert_eq!(events.len(), 7);
        assert!(matches!(events[0], AgentEvent::RunStarted { .. }));
        assert!(matches!(events[1], AgentEvent::TurnStarted { step: 1, .. }));
        assert!(matches!(
            events[2],
            AgentEvent::ModelStarted { step: 1, .. }
        ));
        assert!(matches!(
            events[3],
            AgentEvent::AssistantDelta {
                step: 1,
                ref content,
                ..
            } if content == "event response"
        ));
        assert!(matches!(
            events[4],
            AgentEvent::AssistantMessage { step: 1, .. }
        ));
        assert!(matches!(
            events[5],
            AgentEvent::TurnFinished { step: 1, .. }
        ));
        assert!(matches!(
            events[6],
            AgentEvent::RunFinished {
                outcome: RunOutcome::Completed,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn run_with_context_preserves_external_identity_in_lifecycle_events() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = Arc::clone(&events);
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("linked response".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("fake model agent should be valid")
        .with_event_sink(move |event| {
            captured_events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event);
        });
        let context = RunContext {
            session_id: Some(Uuid::from_u128(1)),
            run_id: Uuid::from_u128(2),
            turn_id: Uuid::from_u128(3),
        };

        agent
            .run_with_context("linked prompt", context)
            .await
            .expect("linked run should succeed");

        let events = events.lock().expect("event mutex should not be poisoned");
        assert!(matches!(
            events.first(),
            Some(AgentEvent::RunStarted { context: actual }) if *actual == context
        ));
        assert!(matches!(
            events.last(),
            Some(AgentEvent::RunFinished { context: actual, .. }) if *actual == context
        ));
    }

    #[tokio::test]
    async fn run_with_context_rejects_a_mismatched_attached_session() {
        let directory = tempdir().expect("temporary directory should exist");
        let expected = Uuid::new_v4();
        let session = SessionStore::open(directory.path().join("session.jsonl"), expected)
            .await
            .expect("session should open");
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("must not run".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("fake model agent should be valid")
        .with_session(session);
        let context = RunContext {
            session_id: Some(Uuid::new_v4()),
            run_id: Uuid::new_v4(),
            turn_id: Uuid::new_v4(),
        };

        let error = agent
            .run_with_context("mismatched session", context)
            .await
            .expect_err("a mismatched session identity must be rejected");

        assert!(matches!(
            error,
            AgentError::SessionContextMismatch {
                expected: actual,
                provided: Some(_),
            } if actual == expected
        ));
    }

    #[tokio::test]
    async fn returns_final_assistant_message_without_tools() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(streaming_response_with_usage(
                Message::assistant(
                    Some("KV cache stores attention keys and values.".to_string()),
                    None,
                ),
                json!({
                    "prompt_tokens": 17,
                    "completion_tokens": 9,
                    "total_tokens": 26
                }),
            ))
            .expect(1)
            .mount(&server)
            .await;

        let agent = AgentLoop::from_config(&test_config(server.uri()), ToolRegistry::new())
            .expect("test agent should be valid");
        let result = agent
            .run("What is KV cache?")
            .await
            .expect("run should succeed");

        assert_eq!(result.steps, 1);
        assert_eq!(result.messages.len(), 2);
        assert_eq!(
            result.final_message.content.as_deref(),
            Some("KV cache stores attention keys and values.")
        );
        assert_eq!(result.input_tokens, Some(17));
        assert_eq!(result.output_tokens, Some(9));
        assert_eq!(result.total_tokens, Some(26));
    }

    #[tokio::test]
    async fn forwards_tool_choice_to_model_client() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("\"tool_choice\":\"none\""))
            .respond_with(streaming_response(Message::assistant(
                Some("tools disabled".to_string()),
                None,
            )))
            .expect(1)
            .mount(&server)
            .await;

        let directory = tempdir().expect("temporary directory should exist");
        let mut config = test_config(server.uri());
        config.workspace = directory.path().to_path_buf();
        let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
            .expect("read_file should register");
        let agent = AgentLoop::from_config(&config, tools)
            .expect("test agent should be valid")
            .with_tool_choice(ToolChoice::None);

        let result = agent
            .run("Answer without calling tools")
            .await
            .expect("run should succeed");

        assert_eq!(
            result.final_message.content.as_deref(),
            Some("tools disabled")
        );
    }

    #[tokio::test]
    async fn resumes_previous_session_messages_before_new_prompt() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("previous answer"))
            .respond_with(streaming_response(Message::assistant(
                Some("new answer".to_string()),
                None,
            )))
            .expect(1)
            .mount(&server)
            .await;

        let directory = tempdir().expect("temporary directory should exist");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        session
            .append(&Message::user("previous question"))
            .await
            .expect("previous user message should append");
        session
            .append(&Message::assistant(
                Some("previous answer".to_string()),
                None,
            ))
            .await
            .expect("previous assistant message should append");

        let agent = AgentLoop::from_config(&test_config(server.uri()), ToolRegistry::new())
            .expect("test agent should be valid")
            .with_session(session.clone());
        let result = agent
            .run("current question")
            .await
            .expect("resumed run should succeed");

        assert_eq!(result.messages.len(), 4);
        assert_eq!(
            result.messages[0].content.as_deref(),
            Some("previous question")
        );
        assert_eq!(
            result.messages[1].content.as_deref(),
            Some("previous answer")
        );
        assert_eq!(
            result.messages[2].content.as_deref(),
            Some("current question")
        );
        assert_eq!(result.messages[3].content.as_deref(), Some("new answer"));
        assert_eq!(
            session.load().await.expect("session should reload").len(),
            4
        );
    }

    #[tokio::test]
    async fn forwards_thinking_setting_from_config() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("\"enable_thinking\":false"))
            .respond_with(streaming_response(Message::assistant(
                Some("thinking disabled".to_string()),
                None,
            )))
            .expect(1)
            .mount(&server)
            .await;

        let mut config = test_config(server.uri());
        config.enable_thinking = Some(false);
        let agent = AgentLoop::from_config(&config, ToolRegistry::new())
            .expect("test agent should be valid");

        let result = agent
            .run("Answer directly")
            .await
            .expect("run should succeed");

        assert_eq!(
            result.final_message.content.as_deref(),
            Some("thinking disabled")
        );
    }

    #[tokio::test]
    async fn executes_tool_then_returns_final_assistant_message() {
        let server = MockServer::start().await;
        let mut read_file_call = tool_call("read_file", "call-1");
        read_file_call.function.arguments = r#"{"path":"hello.txt"}"#.to_string();
        Mock::given(method("POST"))
            .respond_with(streaming_response(Message::assistant(
                None,
                Some(vec![read_file_call]),
            )))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(streaming_response(Message::assistant(
                Some("The file says: hello from the tool.".to_string()),
                None,
            )))
            .expect(1)
            .mount(&server)
            .await;

        let directory = tempdir().expect("temporary directory should exist");
        fs::write(directory.path().join("hello.txt"), "hello from the tool")
            .expect("fixture should be written");
        let mut config = test_config(server.uri());
        config.workspace = directory.path().to_path_buf();
        let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
            .expect("read_file should register");
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = Arc::clone(&events);
        let agent = AgentLoop::from_config(&config, tools)
            .expect("test agent should be valid")
            .with_event_sink(move |event| {
                captured_events
                    .lock()
                    .expect("event mutex should not be poisoned")
                    .push(event);
            });

        let result = agent
            .run("Read hello.txt")
            .await
            .expect("run should succeed");

        assert_eq!(result.steps, 2);
        assert_eq!(result.messages.len(), 4);
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(result.messages[1].role, "assistant");
        assert_eq!(result.messages[2].role, "tool");
        assert_eq!(
            result.messages[2].content.as_deref(),
            Some("hello from the tool")
        );
        assert_eq!(result.messages[3].role, "assistant");

        let events = events.lock().expect("event mutex should not be poisoned");
        assert_eq!(events.len(), 14);
        assert!(matches!(events[0], AgentEvent::RunStarted { .. }));
        assert!(matches!(events[1], AgentEvent::TurnStarted { step: 1, .. }));
        assert!(matches!(
            events[2],
            AgentEvent::ModelStarted { step: 1, .. }
        ));
        assert!(matches!(
            events[3],
            AgentEvent::ToolCallDelta {
                step: 1,
                index: 0,
                ..
            }
        ));
        assert!(matches!(
            events[4],
            AgentEvent::AssistantMessage { step: 1, .. }
        ));
        assert!(matches!(events[5], AgentEvent::ToolStarted { step: 1, .. }));
        assert!(matches!(
            events[6],
            AgentEvent::ToolFinished {
                step: 1,
                is_error: false,
                ..
            }
        ));
        assert!(matches!(
            events[7],
            AgentEvent::TurnFinished { step: 1, .. }
        ));
        assert!(matches!(events[8], AgentEvent::TurnStarted { step: 2, .. }));
        assert!(matches!(
            events[9],
            AgentEvent::ModelStarted { step: 2, .. }
        ));
        assert!(matches!(
            events[10],
            AgentEvent::AssistantDelta {
                step: 2,
                ref content,
                ..
            } if content == "The file says: hello from the tool."
        ));
        assert!(matches!(
            events[11],
            AgentEvent::AssistantMessage { step: 2, .. }
        ));
        assert!(matches!(
            events[12],
            AgentEvent::TurnFinished { step: 2, .. }
        ));
        assert!(matches!(
            events[13],
            AgentEvent::RunFinished {
                outcome: RunOutcome::Completed,
                steps: 2,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn does_not_execute_tools_on_final_allowed_step() {
        let server = MockServer::start().await;
        let unknown_call = tool_call("unknown_tool", "call-1");
        Mock::given(method("POST"))
            .respond_with(streaming_response(Message::assistant(
                None,
                Some(vec![unknown_call]),
            )))
            .expect(1)
            .mount(&server)
            .await;

        let mut config = test_config(server.uri());
        config.max_steps = 1;
        let agent = AgentLoop::from_config(&config, ToolRegistry::new())
            .expect("test agent should be valid");

        let error = agent
            .run("Call a tool")
            .await
            .expect_err("tool call on final step should stop the run");

        assert!(matches!(
            error,
            AgentError::MaxStepsExceeded { max_steps: 1 }
        ));
    }

    #[tokio::test]
    async fn applies_tool_call_limit_from_config_policy() {
        let server = MockServer::start().await;
        let first_call = tool_call("first", "call-1");
        let second_call = tool_call("second", "call-2");
        Mock::given(method("POST"))
            .respond_with(streaming_response(Message::assistant(
                None,
                Some(vec![first_call, second_call]),
            )))
            .expect(1)
            .mount(&server)
            .await;

        let mut config = test_config(server.uri());
        config.max_tool_calls_per_step = 1;
        let agent = AgentLoop::from_config(&config, ToolRegistry::new())
            .expect("test agent should be valid");

        let error = agent
            .run("Call tools")
            .await
            .expect_err("the configured tool limit should be enforced");

        assert!(matches!(
            error,
            AgentError::TooManyToolCalls {
                step: 1,
                count: 2,
                limit: 1
            }
        ));
    }

    #[tokio::test]
    async fn persists_rejected_tool_results_for_an_invalid_tool_batch() {
        let directory = tempdir().expect("temporary directory should exist");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
            None,
            Some(vec![
                tool_call("echo_tool", "duplicate"),
                tool_call("echo_tool", "duplicate"),
            ]),
        )])));
        let agent = AgentLoop::new(
            SequenceChatModel { responses },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("agent should be valid")
        .with_session(session.clone());

        let error = agent
            .run("call the tool twice")
            .await
            .expect_err("duplicate tool IDs should be rejected");

        assert!(matches!(
            error,
            AgentError::InvalidToolCall { index: 1, .. }
        ));
        let records = session.load().await.expect("session should load");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].message.role, "user");
        let repaired = session
            .load_repaired()
            .await
            .expect("invalid assistant output must not corrupt the session");
        assert_eq!(repaired.len(), 1);
    }

    #[tokio::test]
    async fn rejects_empty_prompt_before_calling_model() {
        let agent = AgentLoop::new(
            ModelClient::new("http://localhost:1", Duration::from_secs(1))
                .expect("endpoint should be valid"),
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("test agent should be valid");

        let error = agent.run("  ").await.expect_err("empty prompt should fail");

        assert!(matches!(error, AgentError::EmptyPrompt));
    }

    #[test]
    fn rejects_zero_max_steps() {
        let error = AgentLoop::new(
            ModelClient::new("http://localhost:1", Duration::from_secs(1))
                .expect("endpoint should be valid"),
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            0,
        )
        .err()
        .expect("zero max_steps should fail");

        assert!(matches!(error, AgentError::InvalidMaxSteps));
    }

    #[test]
    fn rejects_zero_tool_call_limit() {
        let error = AgentLoop::new_with_policy(
            ModelClient::new("http://localhost:1", Duration::from_secs(1))
                .expect("endpoint should be valid"),
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                max_tool_calls_per_step: 0,
                ..RunPolicy::default()
            },
        )
        .err()
        .expect("zero tool calls should fail validation");

        assert!(matches!(
            error,
            AgentError::InvalidRunPolicy {
                field: "max_tool_calls_per_step"
            }
        ));
    }

    #[test]
    fn rejects_zero_parallel_tool_limit() {
        let error = AgentLoop::new_with_policy(
            ModelClient::new("http://localhost:1", Duration::from_secs(1))
                .expect("endpoint should be valid"),
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                max_parallel_tool_calls: 0,
                ..RunPolicy::default()
            },
        )
        .err()
        .expect("zero parallel tools should fail validation");

        assert!(matches!(
            error,
            AgentError::InvalidRunPolicy {
                field: "max_parallel_tool_calls"
            }
        ));
    }

    #[test]
    fn rejects_zero_context_byte_limit() {
        let error = AgentLoop::new_with_policy(
            ModelClient::new("http://localhost:1", Duration::from_secs(1))
                .expect("endpoint should be valid"),
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                max_context_bytes: 0,
                ..RunPolicy::default()
            },
        )
        .err()
        .expect("zero context bytes should fail validation");

        assert!(matches!(
            error,
            AgentError::InvalidRunPolicy {
                field: "max_context_bytes"
            }
        ));
    }

    #[test]
    fn model_retry_backoff_is_exponential_and_bounded() {
        assert_eq!(model_retry_delay_ms(250, 1), 250);
        assert_eq!(model_retry_delay_ms(250, 2), 500);
        assert_eq!(model_retry_delay_ms(250, 3), 1_000);
        assert_eq!(model_retry_delay_ms(10_000, 8), 30_000);
    }

    #[test]
    fn stores_run_timeout_from_policy() {
        let agent = AgentLoop::new_with_policy(
            ModelClient::new("http://localhost:1", Duration::from_secs(1))
                .expect("endpoint should be valid"),
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                run_timeout_secs: Some(9),
                ..RunPolicy::default()
            },
        )
        .expect("run timeout should be valid");

        assert_eq!(agent.run_timeout, Some(Duration::from_secs(9)));
    }

    #[test]
    fn from_config_stores_run_timeout() {
        let config = Config {
            base_url: "http://localhost:1".to_string(),
            run_timeout_secs: Some(12),
            ..Config::default()
        };

        let agent = AgentLoop::from_config(&config, ToolRegistry::new())
            .expect("configured agent should be valid");

        assert_eq!(agent.run_timeout, Some(Duration::from_secs(12)));
    }

    #[test]
    fn rejects_zero_run_timeout() {
        let error = AgentLoop::new_with_policy(
            ModelClient::new("http://localhost:1", Duration::from_secs(1))
                .expect("endpoint should be valid"),
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                run_timeout_secs: Some(0),
                ..RunPolicy::default()
            },
        )
        .err()
        .expect("zero run timeout should fail validation");

        assert!(matches!(
            error,
            AgentError::InvalidRunPolicy {
                field: "run_timeout_secs"
            }
        ));
    }

    #[test]
    fn aggregates_complete_model_usage() {
        let mut stats = RunStats::default();
        stats.record_model_usage(Some(ModelUsage {
            input_tokens: Some(7),
            output_tokens: Some(3),
            total_tokens: Some(10),
        }));
        stats.record_model_usage(Some(ModelUsage {
            input_tokens: Some(11),
            output_tokens: Some(5),
            total_tokens: Some(16),
        }));

        assert_eq!(stats.model_usage(), (Some(18), Some(8), Some(26)));
    }

    #[test]
    fn incomplete_model_usage_is_not_reported_as_zero() {
        let mut stats = RunStats::default();
        stats.record_model_usage(Some(ModelUsage {
            input_tokens: Some(7),
            output_tokens: Some(3),
            total_tokens: Some(10),
        }));
        stats.record_model_usage(None);

        assert_eq!(stats.model_usage(), (None, None, None));
    }

    #[test]
    fn preserves_usage_fields_when_one_provider_field_is_missing() {
        let mut stats = RunStats::default();
        stats.record_model_usage(Some(ModelUsage {
            input_tokens: Some(7),
            output_tokens: Some(3),
            total_tokens: None,
        }));

        assert_eq!(stats.model_usage(), (Some(7), Some(3), None));
    }
}
