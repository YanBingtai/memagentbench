use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    CancellationToken, EditFileTool, ListFilesTool, ReadFileTool, SearchFilesTool, ShellTool, Tool,
    ToolContext, ToolError, ToolExecutionContext, ToolResult, WriteFileTool,
};

use super::{shell_error, MAX_WORKER_REQUEST_BYTES};

/// Internal process entry point. It never starts a model or reads API keys.
pub const DELTABOX_WORKER_FLAG: &str = "--deltabox-tool-worker";
pub(super) const WORKER_ARG_BYTES: usize = 32 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkerRequest {
    pub(super) name: String,
    pub(super) arguments: Value,
    pub(super) max_file_bytes: usize,
    pub(super) max_output_bytes: usize,
    pub(super) lease: PathBuf,
    pub(super) timeout_ms: u64,
}

/// Internal CLI worker. Execute a fixed native tool in the inherited DeltaBox
/// view; never discover providers, load config, or initialize another sandbox.
pub async fn run_deltabox_worker(chunks: Vec<String>) -> Result<ToolResult, ToolError> {
    if chunks.iter().map(String::len).sum::<usize>() > MAX_WORKER_REQUEST_BYTES {
        return Err(shell_error("DeltaBox worker request is too large"));
    }
    let request: WorkerRequest = serde_json::from_str(&chunks.concat()).map_err(shell_error)?;
    if request.max_output_bytes == 0 || request.timeout_ms == 0 {
        return Err(shell_error("DeltaBox worker limits must be positive"));
    }
    let context = ToolContext {
        workspace: std::env::current_dir().map_err(shell_error)?,
        max_file_bytes: request.max_file_bytes,
        max_tool_output_bytes: request.max_output_bytes,
    };
    let tool: Box<dyn Tool> = match request.name.as_str() {
        "read_file" => Box::new(ReadFileTool::new(context)),
        "list_files" => Box::new(ListFilesTool::new(context)),
        "search_files" => Box::new(SearchFilesTool::new(context)),
        "write_file" => Box::new(WriteFileTool::new(context)),
        "edit_file" => Box::new(EditFileTool::new(context)),
        "bash" => Box::new(ShellTool::new(context)),
        _ => return Err(shell_error("unsupported DeltaBox worker tool")),
    };
    tool.validate_arguments(&request.arguments)?;
    let cancellation = CancellationToken::new();
    let execution_context = ToolExecutionContext::new(cancellation.clone());
    // A missing lease means the owning future was dropped. Denied access is a
    // configuration error, not evidence that the caller cancelled.
    if !lease_is_live(&request.lease)? {
        return Ok(ToolResult::error("tool cancelled before execution")
            .with_details(json!({"status":"cancelled", "executed":false})));
    }
    let work = tool.execute_result_with_context(request.arguments, execution_context);
    tokio::pin!(work);
    let watch = async {
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            if !lease_is_live(&request.lease)? {
                return Ok::<(), ToolError>(());
            }
        }
    };
    let result = tokio::select! {
        result = &mut work => result,
        missing = watch => {
            cancellation.cancel();
            let result = work.await;
            missing?;
            result
        }
        _ = tokio::time::sleep(Duration::from_millis(request.timeout_ms)) => {
            cancellation.cancel();
            let _ = work.await;
            Ok(ToolResult::error("tool wall-time limit exceeded")
                .with_details(json!({"status":"timed_out", "executed":true})))
        }
    };
    Ok(match result {
        Ok(result) => result,
        Err(error) => ToolResult::error(error.to_string()),
    }
    .bounded(request.max_output_bytes))
}

fn lease_is_live(path: &Path) -> Result<bool, ToolError> {
    match fs::metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(shell_error(format!(
            "cannot read cancellation lease: {error}"
        ))),
    }
}

pub(super) fn chunk_request(mut request: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    while !request.is_empty() {
        let mut end = request.len().min(WORKER_ARG_BYTES);
        while !request.is_char_boundary(end) {
            end -= 1;
        }
        chunks.push(request[..end].to_string());
        request = &request[end..];
    }
    chunks
}
