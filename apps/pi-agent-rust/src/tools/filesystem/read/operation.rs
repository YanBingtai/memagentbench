//! Bounded file reading for `read_file`.

use crate::tools::{
    errors::ToolError,
    filesystem::{path::resolve_workspace_file, read::arguments::ReadFileArguments},
    limits::truncate_output,
    ToolContext,
};

pub(super) async fn execute(
    context: &ToolContext,
    arguments: ReadFileArguments,
) -> Result<String, ToolError> {
    let path = resolve_workspace_file(&context.workspace, &arguments.path, "read_file").await?;
    let metadata = tokio::fs::metadata(&path)
        .await
        .map_err(|source| ToolError::InspectFile {
            path: path.clone(),
            source,
        })?;

    if !metadata.is_file() {
        return Err(ToolError::InvalidPath {
            tool: "read_file".to_string(),
            path,
        });
    }
    if metadata.len() > context.max_file_bytes as u64 {
        return Err(ToolError::FileTooLarge {
            path,
            size: metadata.len(),
            limit: context.max_file_bytes,
        });
    }

    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|source| ToolError::ReadFile {
            path: path.clone(),
            source,
        })?;
    let text =
        String::from_utf8(bytes).map_err(|_| ToolError::InvalidUtf8 { path: path.clone() })?;
    let lines: Vec<&str> = text.split('\n').collect();
    let start = arguments.offset.unwrap_or(1);
    if start == 0 || start > lines.len() {
        return Err(ToolError::InvalidRange {
            tool: "read_file".to_string(),
            reason: format!(
                "offset {start} is outside the file, which contains {} lines",
                lines.len()
            ),
        });
    }
    let start_index = start - 1;
    let end_index = match arguments.limit {
        Some(0) => {
            return Err(ToolError::InvalidRange {
                tool: "read_file".to_string(),
                reason: "limit must be greater than zero".to_string(),
            });
        }
        Some(limit) => start_index.saturating_add(limit).min(lines.len()),
        None => lines.len(),
    };
    let selected = lines[start_index..end_index].join("\n");

    Ok(truncate_output(selected, context.max_tool_output_bytes))
}
