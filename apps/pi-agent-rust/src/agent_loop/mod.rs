mod context;
mod control;
mod error;
mod execution;
mod lifecycle;
mod model_turn;
mod outcomes;
mod parallel;
mod request_context;
mod results;
mod runtime;
mod sequential;
mod tool_batch;
mod tool_validation;
mod turn;
mod types;

use crate::events::RunContext;
#[cfg(test)]
use context::{message_size, truncate_context};
pub use control::CancellationToken;
use control::ControlState;
pub use error::AgentError;
#[cfg(test)]
use outcomes::run_outcome_for_error;
use outcomes::{tool_result_has_unknown_outcome, tool_result_status};
#[cfg(test)]
use request_context::format_memory_context;
use tool_batch::{await_tool_hook, ToolHookWait};
#[cfg(test)]
use tool_validation::validate_tool_call_batch;
pub(crate) use types::RunStats;
pub use types::{
    AfterRunFuture, AfterRunRequest, BeforeToolDecision, ContextCompactionFuture,
    ContextCompactionReason, ContextCompactionRequest, MemoryHookConfig, MemoryHookFuture,
    MemoryHookRequest, PendingRuntimeAction, RunResult, RuntimeReconcileOutcome, ToolHookContext,
    ToolHookFuture, ToolResultContext, ToolResultOverride,
};
use types::{
    AfterRunHook, AfterToolHook, BeforeToolHook, ContextCompactionHook, MemoryProposalHook,
};

use std::{
    sync::{Arc, Mutex as StdMutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde_json::Value;

#[cfg(test)]
use pi_agent_runtime::{MemoryInjection, MemoryProposal, RiskClass};

use crate::{
    config::{Config, RunPolicy},
    events::AgentEvent,
    message::ToolCall,
    model::{ChatModel, ChatOptions, ModelClient, ToolChoice},
    session::SessionStore,
    skills::SkillCatalog,
    tools::{PreparedToolCall, ToolGateway, ToolRegistry},
};

#[cfg(test)]
use crate::{
    events::RunOutcome,
    message::{Message, ToolResultMetadata},
    model::ModelUsage,
};

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

    fn emit(&self, event: AgentEvent) {
        if let Some(sink) = &self.event_sink {
            sink.emit(event);
        }
    }
}

fn timeout_error(timeout: Duration) -> AgentError {
    AgentError::RunTimedOut {
        timeout_ms: timeout_millis(timeout),
    }
}

pub(crate) fn model_retry_delay_ms(base_ms: u64, retry_count: usize) -> u64 {
    let shift = retry_count.saturating_sub(1).min(6) as u32;
    base_ms.saturating_mul(1_u64 << shift).min(30_000)
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

#[cfg(test)]
mod tests;
