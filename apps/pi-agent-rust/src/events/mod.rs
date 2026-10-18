//! Events emitted by the agent runtime.
//!
//! Events are an ephemeral protocol for a CLI, TUI, or service consumer.
//! Durable session storage keeps canonical messages separately; streaming
//! deltas should not be persisted as independent transcript messages.

mod context;
mod event;
#[cfg(test)]
mod tests;

pub use context::{MemoryTrace, RunContext, RunOutcome};
pub use event::AgentEvent;
