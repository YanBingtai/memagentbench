//! Request-context assembly for one AgentLoop model turn.
//!
//! The implementation is split by responsibility: context budgeting, memory
//! injection, and queued user-message control. The facade keeps the existing
//! AgentLoop method surface intact for the turn runner.

mod budget;
mod control;
mod memory;

#[cfg(test)]
#[allow(unused_imports)]
pub(super) use memory::format_memory_context;
