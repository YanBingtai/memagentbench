//! The confined DeltaBox worker process.
//!
//! The worker deliberately owns only request decoding and execution of the
//! fixed native tool set.  Sandbox creation and supervision stay in the
//! parent process, so this entry point cannot initialize another model or
//! discover a different provider.

mod execution;
mod lease;
mod request;

use crate::{ToolError, ToolResult};

use super::shell_error;

pub(super) use self::request::WorkerRequest;
pub(super) use self::request::{chunk_request, parse_request};

/// Internal process entry point. It never starts a model or reads API keys.
pub const DELTABOX_WORKER_FLAG: &str = "--deltabox-tool-worker";
pub(super) const WORKER_ARG_BYTES: usize = 32 * 1024;

/// Execute one fixed native tool in the inherited DeltaBox view.
pub async fn run_deltabox_worker(chunks: Vec<String>) -> Result<ToolResult, ToolError> {
    let request = parse_request(&chunks)?;
    execution::run(request).await
}

pub(super) fn invalid_request(message: impl std::fmt::Display) -> ToolError {
    shell_error(message)
}
