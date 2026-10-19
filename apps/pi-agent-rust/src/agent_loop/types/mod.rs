//! Public contracts shared by the agent loop implementation.

mod context;
mod hooks;
mod memory;
mod results;
mod stats;

pub(super) use context::ContextCompactionHook;
pub use context::{ContextCompactionFuture, ContextCompactionReason, ContextCompactionRequest};
pub use hooks::{
    AfterRunFuture, AfterRunRequest, BeforeToolDecision, ToolHookContext, ToolHookFuture,
    ToolResultContext, ToolResultOverride,
};
pub(super) use hooks::{AfterRunHook, AfterToolHook, BeforeToolHook};
pub(super) use memory::MemoryProposalHook;
pub use memory::{MemoryHookConfig, MemoryHookFuture, MemoryHookRequest};
pub use results::{PendingRuntimeAction, RunResult, RuntimeReconcileOutcome};
pub(crate) use stats::RunStats;
