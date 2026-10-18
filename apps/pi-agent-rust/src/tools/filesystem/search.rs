//! Implementation of the bounded literal-text `search_files` tool.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::protocol::ToolDefinition;
use pi_agent_runtime::RiskClass;

use super::super::{
    errors::ToolError, limits::*, registry::Tool, ToolConcurrency, ToolContext, ToolFuture,
    ToolReplayPolicy,
};

use std::path::{Path, PathBuf};

use super::path::{display_relative_path, resolve_workspace_directory};

/// Search UTF-8 text files inside the configured workspace.
#[derive(Debug, Clone)]
pub struct SearchFilesTool {
    context: ToolContext,
}

impl SearchFilesTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for SearchFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "search_files",
            "Search UTF-8 text files for a literal substring inside the configured workspace.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Literal, case-sensitive substring to find"
                    },
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory; defaults to the workspace root"
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_RESULTS,
                        "description": "Maximum matching lines to return"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let parsed: SearchFilesArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "search_files".to_string(),
                    source,
                }
            })?;
        if parsed.query.is_empty() {
            return Err(ToolError::InvalidArgumentValue {
                name: "search_files".to_string(),
                reason: "query must not be empty".to_string(),
            });
        }
        if parsed
            .max_results
            .is_some_and(|results| results == 0 || results > MAX_SEARCH_RESULTS)
        {
            return Err(ToolError::InvalidArgumentValue {
                name: "search_files".to_string(),
                reason: format!("max_results must be between 1 and {MAX_SEARCH_RESULTS}"),
            });
        }
        Ok(())
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Safe
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: SearchFilesArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "search_files".to_string(),
                        source,
                    }
                })?;
            let max_results = arguments.max_results.unwrap_or(DEFAULT_SEARCH_MAX_RESULTS);
            let requested = arguments.path.unwrap_or_else(|| PathBuf::from("."));
            let root =
                resolve_workspace_directory(&self.context.workspace, &requested, "search_files")
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

            let mut output = if report.matches.is_empty() {
                "(no matches)".to_string()
            } else {
                report.matches.join("\n")
            };
            if report.skipped_files > 0 {
                output.push_str(&format!(
                    "\n[skipped {} unreadable, binary, or oversized files]",
                    report.skipped_files
                ));
            }
            if report.truncated {
                if report.file_limit_reached {
                    output.push_str(&format!(
                        "\n[search truncated after scanning {MAX_SEARCH_FILES} files]"
                    ));
                } else {
                    output.push_str(&format!(
                        "\n[search truncated after {max_results} matching lines]"
                    ));
                }
            }
            Ok(truncate_output(output, self.context.max_tool_output_bytes))
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchFilesArguments {
    query: String,
    path: Option<PathBuf>,
    max_results: Option<usize>,
}

#[derive(Debug, Default)]
struct SearchReport {
    matches: Vec<String>,
    skipped_files: usize,
    truncated: bool,
    file_limit_reached: bool,
}

async fn search_directory(
    workspace: &Path,
    root: &Path,
    prefix: &Path,
    query: &str,
    max_results: usize,
    max_file_bytes: usize,
) -> Result<SearchReport, ToolError> {
    let mut pending = vec![(root.to_path_buf(), PathBuf::new())];
    let mut report = SearchReport::default();
    let mut scanned_files = 0usize;

    while let Some((directory, relative_directory)) = pending.pop() {
        let mut reader =
            tokio::fs::read_dir(&directory)
                .await
                .map_err(|source| ToolError::ReadDirectory {
                    path: directory.clone(),
                    source,
                })?;
        let mut children = Vec::new();
        while let Some(entry) =
            reader
                .next_entry()
                .await
                .map_err(|source| ToolError::ReadDirectory {
                    path: directory.clone(),
                    source,
                })?
        {
            children.push(entry);
        }
        children.sort_by_key(|entry| entry.file_name());

        let mut child_directories = Vec::new();
        for entry in children {
            let name = entry.file_name();
            let relative = if relative_directory.as_os_str().is_empty() {
                PathBuf::from(&name)
            } else {
                relative_directory.join(&name)
            };
            let file_type = entry
                .file_type()
                .await
                .map_err(|source| ToolError::InspectFile {
                    path: entry.path(),
                    source,
                })?;

            if file_type.is_dir() {
                if should_skip_search_directory(&name) {
                    continue;
                }
                let canonical = tokio::fs::canonicalize(entry.path())
                    .await
                    .map_err(|source| ToolError::InspectFile {
                        path: entry.path(),
                        source,
                    })?;
                if !canonical.starts_with(workspace) {
                    return Err(ToolError::PathOutsideWorkspace {
                        tool: "search_files".to_string(),
                        path: entry.path(),
                    });
                }
                child_directories.push((canonical, relative));
                continue;
            }
            if !file_type.is_file() {
                continue;
            }

            if scanned_files == MAX_SEARCH_FILES {
                report.truncated = true;
                report.file_limit_reached = true;
                return Ok(report);
            }
            scanned_files += 1;

            let path = entry.path();
            let metadata = match tokio::fs::metadata(&path).await {
                Ok(metadata) => metadata,
                Err(_) => {
                    report.skipped_files += 1;
                    continue;
                }
            };
            if metadata.len() > max_file_bytes as u64 {
                report.skipped_files += 1;
                continue;
            }
            let text = match tokio::fs::read_to_string(&path).await {
                Ok(text) => text,
                Err(_) => {
                    report.skipped_files += 1;
                    continue;
                }
            };
            for (line_index, line) in text.lines().enumerate() {
                if !line.contains(query) {
                    continue;
                }
                let display_path = if prefix.as_os_str().is_empty() {
                    relative.clone()
                } else {
                    prefix.join(&relative)
                };
                report.matches.push(format!(
                    "{}:{}:{}",
                    display_path.display(),
                    line_index + 1,
                    truncate_output(line.to_string(), MAX_SEARCH_LINE_BYTES)
                ));
                if report.matches.len() == max_results {
                    report.truncated = true;
                    return Ok(report);
                }
            }
        }

        for child in child_directories.into_iter().rev() {
            pending.push(child);
        }
    }

    Ok(report)
}

fn should_skip_search_directory(name: &std::ffi::OsStr) -> bool {
    matches!(
        name.to_str(),
        Some(".git") | Some("target") | Some("node_modules")
    )
}
