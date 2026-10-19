//! Exact-match file replacement for `edit_file`.

use crate::tools::{
    errors::ToolError,
    filesystem::{
        edit::arguments::EditFileArguments, path::resolve_workspace_file,
        write::atomic_write_workspace_file,
    },
    ToolContext,
};

pub(super) async fn execute(
    context: &ToolContext,
    arguments: EditFileArguments,
) -> Result<String, ToolError> {
    if arguments.old_text.is_empty() {
        return Err(ToolError::EditTargetEmpty {
            path: arguments.path,
        });
    }

    let path = resolve_workspace_file(&context.workspace, &arguments.path, "edit_file").await?;
    let metadata = tokio::fs::metadata(&path)
        .await
        .map_err(|source| ToolError::InspectFile {
            path: path.clone(),
            source,
        })?;
    if !metadata.is_file() {
        return Err(ToolError::InvalidPath {
            tool: "edit_file".to_string(),
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
    let matches = text.match_indices(&arguments.old_text).count();
    match matches {
        0 => return Err(ToolError::EditTargetNotFound { path }),
        1 => {}
        matches => return Err(ToolError::EditTargetNotUnique { path, matches }),
    }

    let updated = text.replacen(&arguments.old_text, &arguments.new_text, 1);
    if updated.len() > context.max_file_bytes {
        return Err(ToolError::FileTooLarge {
            path: arguments.path,
            size: updated.len() as u64,
            limit: context.max_file_bytes,
        });
    }
    atomic_write_workspace_file(&context.workspace, &arguments.path, updated.as_bytes()).await?;

    Ok(format!("Edited {}", arguments.path.display()))
}
