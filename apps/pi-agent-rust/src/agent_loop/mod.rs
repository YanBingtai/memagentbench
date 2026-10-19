mod builder;
mod context;
mod control;
mod error;
mod execution;
mod helpers;
mod lifecycle;
mod model_turn;
mod outcomes;
mod parallel;
mod request_context;
mod results;
mod runtime;
mod sequential;
mod tool_batch;
mod tool_execution;
mod turn;
mod types;

use crate::events::RunContext;

#[cfg(test)]
use context::{message_size, truncate_context};
pub use control::CancellationToken;
use control::ControlState;
pub use error::AgentError;
pub(crate) use helpers::model_retry_delay_ms;
use helpers::{
    elapsed_millis, preflight_model_tool_call, timeout_error, timeout_millis, unix_timestamp_ms,
    validate_policy,
};
#[cfg(test)]
use outcomes::run_outcome_for_error;
use outcomes::tool_result_status;
#[cfg(test)]
use request_context::format_memory_context;
use tool_batch::{await_tool_hook, ToolHookWait};
#[cfg(test)]
use turn::validation::validate_tool_call_batch;
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
    time::Duration,
};

#[cfg(test)]
use pi_agent_runtime::{MemoryInjection, MemoryProposal, RiskClass};

#[cfg(test)]
use serde_json::Value;

use crate::{
    config::RunPolicy,
    events::AgentEvent,
    model::{ChatOptions, ModelClient},
    session::SessionStore,
    skills::SkillCatalog,
    tools::ToolGateway,
};

#[cfg(test)]
use crate::{
    config::Config,
    model::{ChatModel, ToolChoice},
    tools::ToolRegistry,
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

impl<M> AgentLoop<M> {
    pub(super) fn emit(&self, event: AgentEvent) {
        if let Some(sink) = &self.event_sink {
            sink.emit(event);
        }
    }
}

#[cfg(test)]
mod tests;
