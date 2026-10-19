//! Workspace traversal and bounded output for `search_files`.

use std::path::PathBuf;

use serde_json::Value;

use super::{definition, format::format_report, walk::search_directory, SearchFilesTool};
use crate::tools::{
    errors::ToolError,
    filesystem::path::{display_relative_path, resolve_workspace_directory},
    limits::DEFAULT_SEARCH_MAX_RESULTS,
};

impl SearchFilesTool {
    pub(super) async fn search(&self, arguments: Value) -> Result<String, ToolError> {
        let arguments = definition::parse(arguments)?;
        let max_results = arguments.max_results.unwrap_or(DEFAULT_SEARCH_MAX_RESULTS);
        let requested = arguments.path.unwrap_or_else(|| PathBuf::from("."));
        let root = resolve_workspace_directory(&self.context.workspace, &requested, "search_files")
            .await?;
        let prefix = display_relative_path(&requested);
        let report = search_directory(
            &root,
            &root,
            &prefix,
            &arguments.query,
            max_results,
            self.context.max_file_bytes,
        )
        .await?;

        Ok(format_report(
            report,
            max_results,
            self.context.max_tool_output_bytes,
        ))
    }
}
