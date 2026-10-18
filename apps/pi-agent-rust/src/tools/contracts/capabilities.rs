//! Scheduler, replay, exposure, and workspace capability contracts.

use std::path::PathBuf;

use pi_agent_runtime::RiskClass;

use crate::config::Config;

/// Concurrency contract used by the agent scheduler.
///
/// Tools are exclusive by default because future tools may mutate files,
/// sessions, or external systems. A tool opts into parallel execution only
/// when its implementation is safe to run concurrently with another call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolConcurrency {
    Exclusive,
    Parallel,
}

/// Declares whether an interrupted invocation may be replayed automatically.
///
/// `Safe` is reserved for tools whose only observable effect is reading
/// already-existing state. Mutating tools remain `Never` unless their
/// implementation provides an idempotency contract that the runtime can
/// enforce during recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolReplayPolicy {
    Never,
    Safe,
}

/// Controls whether a registered tool is advertised to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolExposure {
    /// Include the tool in the model-visible function list.
    Direct,
    /// Keep the tool executable for trusted callers without advertising it.
    Hidden,
}

/// Immutable capability metadata for one registered tool.
///
/// The model-facing registry owns discovery and schema snapshots. This small
/// value is the execution-side contract that an agent loop or adapter can
/// inspect before it submits an action to the runtime. It deliberately does
/// not contain sandbox or runtime implementation details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolCapabilities {
    pub concurrency: ToolConcurrency,
    pub replay_policy: ToolReplayPolicy,
    pub exposure: ToolExposure,
    pub risk_class: RiskClass,
}

/// Shared limits and filesystem root available to built-in tools.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub workspace: PathBuf,
    pub max_file_bytes: usize,
    pub max_tool_output_bytes: usize,
}

impl ToolContext {
    pub fn from_config(config: &Config) -> Self {
        Self {
            workspace: config.workspace.clone(),
            max_file_bytes: config.max_file_bytes,
            max_tool_output_bytes: config.max_tool_output_bytes,
        }
    }
}
