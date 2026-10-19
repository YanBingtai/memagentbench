//! Portable atomic replacement fallback for non-Unix platforms.

use std::path::Path;

use crate::tools::errors::ToolError;

pub(super) async fn write(
    temporary: &Path,
    target: &Path,
    content: &[u8],
) -> Result<(), ToolError> {
    if let Err(source) = tokio::fs::write(temporary, content).await {
        let _ = tokio::fs::remove_file(temporary).await;
        return Err(ToolError::WriteFile {
            path: temporary.to_path_buf(),
            source,
        });
    }

    if let Err(source) = tokio::fs::rename(temporary, target).await {
        let _ = tokio::fs::remove_file(temporary).await;
        return Err(ToolError::RenameFile {
            path: target.to_path_buf(),
            temporary: temporary.to_path_buf(),
            source,
        });
    }

    Ok(())
}
