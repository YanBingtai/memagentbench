use std::time::Duration;

use serde_json::json;

use crate::{
    CancellationToken, EditFileTool, ListFilesTool, ReadFileTool, SearchFilesTool, ShellTool, Tool,
    ToolContext, ToolError, ToolExecutionContext, ToolResult, WriteFileTool,
};

use super::{lease, WorkerRequest};

pub(super) async fn run(request: WorkerRequest) -> Result<ToolResult, ToolError> {
    let context = ToolContext {
        workspace: std::env::current_dir().map_err(super::invalid_request)?,
        max_file_bytes: request.max_file_bytes,
        max_tool_output_bytes: request.max_output_bytes,
    };
    let tool = build_tool(&request.name, context)?;
    tool.validate_arguments(&request.arguments)?;

    if !lease::is_live(&request.lease)? {
        return Ok(cancelled_before_execution());
    }

    let max_output_bytes = request.max_output_bytes;
    let result = execute_with_monitor(tool, request).await;
    Ok(result
        .unwrap_or_else(|error| ToolResult::error(error.to_string()))
        .bounded(max_output_bytes))
}

fn build_tool(name: &str, context: ToolContext) -> Result<Box<dyn Tool>, ToolError> {
    match name {
        "read_file" => Ok(Box::new(ReadFileTool::new(context))),
        "list_files" => Ok(Box::new(ListFilesTool::new(context))),
        "search_files" => Ok(Box::new(SearchFilesTool::new(context))),
        "write_file" => Ok(Box::new(WriteFileTool::new(context))),
        "edit_file" => Ok(Box::new(EditFileTool::new(context))),
        "bash" => Ok(Box::new(ShellTool::new(context))),
        _ => Err(super::invalid_request("unsupported DeltaBox worker tool")),
    }
}

async fn execute_with_monitor(
    tool: Box<dyn Tool>,
    request: WorkerRequest,
) -> Result<ToolResult, ToolError> {
    let cancellation = CancellationToken::new();
    let execution_context = ToolExecutionContext::new(cancellation.clone());
    let work = tool.execute_result_with_context(request.arguments, execution_context);
    tokio::pin!(work);

    let result = tokio::select! {
        result = &mut work => result,
        missing = lease::wait_for_release(&request.lease) => {
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
    }?;
    Ok(result)
}

fn cancelled_before_execution() -> ToolResult {
    ToolResult::error("tool cancelled before execution")
        .with_details(json!({"status":"cancelled", "executed":false}))
}
