//! Core tool adapter for a DeltaBox workspace.
//!
//! Every filesystem tool runs the same native Rust implementation in a
//! confined child of this executable. This keeps bash and file tools on the
//! same private view without duplicating path validation or file-edit logic.

mod sandbox;
mod session;
mod tools;
mod worker;

use serde_json::json;

use crate::tools::{ToolError, ToolResult};

const MAX_WORKER_REQUEST_BYTES: usize = 1_100_000;
const CLEANUP_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Debug, Clone, Copy)]
pub enum DeltaBoxDisposition {
    /// Keep the session private and available to explicit review commands.
    Review,
    Commit,
    Abort,
}

pub use sandbox::DeltaBoxSandbox;
pub use worker::{run_deltabox_worker, DELTABOX_WORKER_FLAG};

fn shell_error(error: impl std::fmt::Display) -> ToolError {
    ToolError::Shell {
        message: error.to_string(),
    }
}

fn unknown_result(session_id: &str, error: impl std::fmt::Display) -> ToolResult {
    ToolResult::error(format!(
        "DeltaBox execution outcome is unknown in session {session_id}: {error}"
    ))
    .with_details(json!({"status":"unknown", "executed":true}))
    .terminating()
}

#[cfg(test)]
mod tests {
    use super::worker::{chunk_request, run_deltabox_worker, WorkerRequest, WORKER_ARG_BYTES};
    use serde_json::json;
    use std::path::PathBuf;

    #[test]
    fn large_unicode_requests_use_small_argv_chunks_without_changing_content() {
        let text = "智能体🦀".repeat(20_000);
        let chunks = chunk_request(&text);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.len() <= WORKER_ARG_BYTES));
        assert_eq!(chunks.concat(), text);
    }

    #[tokio::test]
    async fn missing_lease_prevents_writes() {
        let request = WorkerRequest {
            name: "write_file".into(),
            arguments: json!({"path":"never-written.txt", "content":"test"}),
            max_file_bytes: 1024,
            max_output_bytes: 1024,
            lease: PathBuf::from("/tmp/pi-agent-nonexistent-lease"),
            timeout_ms: 1000,
        };
        let result = run_deltabox_worker(vec![serde_json::to_string(&request).unwrap()])
            .await
            .unwrap();
        assert_eq!(result.details["executed"], false);
    }
}
