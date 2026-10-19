//! Deterministic directory traversal for `list_files`.

use std::path::PathBuf;

use crate::tools::{
    errors::ToolError,
    filesystem::{
        list::arguments::{validate_limits, ListFilesArguments},
        path::{collect_directory_entries, display_relative_path, resolve_workspace_directory},
    },
    limits::{truncate_output, DEFAULT_LIST_MAX_DEPTH, DEFAULT_LIST_MAX_ENTRIES},
    ToolContext,
};

pub(super) async fn execute(
    context: &ToolContext,
    arguments: ListFilesArguments,
) -> Result<String, ToolError> {
    let max_depth = arguments.max_depth.unwrap_or(DEFAULT_LIST_MAX_DEPTH);
    let max_entries = arguments.max_entries.unwrap_or(DEFAULT_LIST_MAX_ENTRIES);
    validate_limits(Some(max_depth), Some(max_entries))?;

    let requested = arguments.path.unwrap_or_else(|| PathBuf::from("."));
    let root = resolve_workspace_directory(&context.workspace, &requested, "list_files").await?;
    let prefix = display_relative_path(&requested);
    let (entries, truncated) =
        collect_directory_entries(&root, &root, &prefix, max_depth, max_entries).await?;

    let mut output = if entries.is_empty() {
        "(empty)".to_string()
    } else {
        entries.join("\n")
    };
    if truncated {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(&format!("[truncated after {max_entries} entries]"));
    }
    Ok(truncate_output(output, context.max_tool_output_bytes))
}
