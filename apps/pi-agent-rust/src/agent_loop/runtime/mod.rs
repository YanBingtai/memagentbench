//! Runtime ledger lifecycle and session reconciliation support.
//!
//! The submodules separate recovery, admission policy, consistency checks, and
//! durable result commits while preserving the agent loop runtime API.

mod commit;
mod consistency;
mod policy;
mod recovery;

pub(super) use commit::commit_tool_result;
pub(super) use consistency::{apply_reconciled_tool_results, validate_session_runtime_consistency};
