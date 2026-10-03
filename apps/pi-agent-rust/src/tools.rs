use std::{
    collections::BTreeMap,
    future::Future,
    path::{Component, Path, PathBuf},
    pin::Pin,
    sync::Arc,
};

use serde::Deserialize;
use serde_json::{json, Value};
use thiserror::Error;

use crate::{config::Config, message::ToolCall, message::ToolDefinition};

pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + 'a>>;

/// Shared limits and filesystem root available to built-in tools.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub workspace: PathBuf,
    pub max_file_bytes: usize,
    pub max_tool_output_bytes: usize,
}

impl ToolContext {
    pub fn from_config(config: &Config) -> Self {
        Self {
            workspace: config.workspace.clone(),
            max_file_bytes: config.max_file_bytes,
            max_tool_output_bytes: config.max_tool_output_bytes,
        }
    }
}

/// Errors produced while looking up, validating, or executing a tool.
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("unknown tool: {name}")]
    UnknownTool { name: String },

    #[error("invalid arguments for tool {name}: {source}")]
    InvalidArguments {
        name: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid path for tool {tool}: {path}")]
    InvalidPath { tool: String, path: PathBuf },

    #[error("path escapes workspace for tool {tool}: {path}")]
    PathOutsideWorkspace { tool: String, path: PathBuf },

    #[error("failed to resolve workspace {path}: {source}")]
    ResolveWorkspace {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to resolve file {path}: {source}")]
    ResolveFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to inspect file {path}: {source}")]
    InspectFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("file is too large: {path} ({size} bytes, limit {limit} bytes)")]
    FileTooLarge {
        path: PathBuf,
        size: u64,
        limit: usize,
    },

    #[error("failed to read file {path}: {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("file is not valid UTF-8: {path}")]
    InvalidUtf8 { path: PathBuf },
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("a tool named {name} is already registered")]
    DuplicateTool { name: String },
}

/// A dynamically dispatchable asynchronous tool.
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a>;
}

/// Registry used by the agent to advertise and execute tools.
#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_read_file(context: ToolContext) -> Result<Self, RegistryError> {
        let mut registry = Self::new();
        registry.register(ReadFileTool::new(context))?;
        Ok(registry)
    }

    pub fn register<T>(&mut self, tool: T) -> Result<(), RegistryError>
    where
        T: Tool + 'static,
    {
        let definition = tool.definition();
        let name = definition.function.name;
        if self.tools.contains_key(&name) {
            return Err(RegistryError::DuplicateTool { name });
        }

        self.tools.insert(name, Arc::new(tool));
        Ok(())
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|tool| tool.definition()).collect()
    }

    pub async fn execute(&self, call: &ToolCall) -> Result<String, ToolError> {
        let tool = self
            .tools
            .get(&call.function.name)
            .cloned()
            .ok_or_else(|| ToolError::UnknownTool {
                name: call.function.name.clone(),
            })?;
        let arguments = serde_json::from_str(&call.function.arguments).map_err(|source| {
            ToolError::InvalidArguments {
                name: call.function.name.clone(),
                source,
            }
        })?;

        tool.execute(arguments).await
    }
}

#[derive(Debug, Clone)]
pub struct ReadFileTool {
    context: ToolContext,
}

impl ReadFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "read_file",
            "Read a UTF-8 text file inside the configured workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path of the file to read"
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        )
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: ReadFileArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "read_file".to_string(),
                        source,
                    }
                })?;
            let path = resolve_workspace_file(&self.context.workspace, &arguments.path).await?;
            let metadata =
                tokio::fs::metadata(&path)
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
            if metadata.len() > self.context.max_file_bytes as u64 {
                return Err(ToolError::FileTooLarge {
                    path,
                    size: metadata.len(),
                    limit: self.context.max_file_bytes,
                });
            }

            let bytes = tokio::fs::read(&path)
                .await
                .map_err(|source| ToolError::ReadFile {
                    path: path.clone(),
                    source,
                })?;
            let text = String::from_utf8(bytes)
                .map_err(|_| ToolError::InvalidUtf8 { path: path.clone() })?;

            Ok(truncate_output(text, self.context.max_tool_output_bytes))
        })
    }
}

#[derive(Debug, Deserialize)]
struct ReadFileArguments {
    path: PathBuf,
}

async fn resolve_workspace_file(workspace: &Path, requested: &Path) -> Result<PathBuf, ToolError> {
    if requested.as_os_str().is_empty()
        || requested.is_absolute()
        || requested
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(ToolError::InvalidPath {
            tool: "read_file".to_string(),
            path: requested.to_path_buf(),
        });
    }

    let workspace =
        tokio::fs::canonicalize(workspace)
            .await
            .map_err(|source| ToolError::ResolveWorkspace {
                path: workspace.to_path_buf(),
                source,
            })?;
    let candidate = workspace.join(requested);
    let canonical = tokio::fs::canonicalize(&candidate)
        .await
        .map_err(|source| ToolError::ResolveFile {
            path: candidate.clone(),
            source,
        })?;

    if !canonical.starts_with(&workspace) {
        return Err(ToolError::PathOutsideWorkspace {
            tool: "read_file".to_string(),
            path: requested.to_path_buf(),
        });
    }

    Ok(canonical)
}

fn truncate_output(text: String, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text;
    }
    if max_bytes == 0 {
        return String::new();
    }

    const MARKER: &str = "…";
    let content_limit = max_bytes.saturating_sub(MARKER.len());
    let mut end = content_limit.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }

    format!("{}{}", &text[..end], MARKER)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{FunctionCall, ToolCall};
    use std::fs;
    use tempfile::tempdir;

    fn read_call(path: &str) -> ToolCall {
        ToolCall {
            id: "call-1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: serde_json::json!({ "path": path }).to_string(),
            },
        }
    }

    #[tokio::test]
    async fn read_file_returns_workspace_content() {
        let directory = tempdir().expect("temporary directory should exist");
        fs::write(directory.path().join("hello.txt"), "hello from workspace")
            .expect("fixture should be written");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let registry = ToolRegistry::with_read_file(context).expect("tool should register");

        let result = registry
            .execute(&read_call("hello.txt"))
            .await
            .expect("read_file should succeed");

        assert_eq!(result, "hello from workspace");
    }

    #[tokio::test]
    async fn read_file_rejects_parent_directory() {
        let directory = tempdir().expect("temporary directory should exist");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let registry = ToolRegistry::with_read_file(context).expect("tool should register");

        let error = registry
            .execute(&read_call("../secret.txt"))
            .await
            .expect_err("parent path should be rejected");

        assert!(matches!(error, ToolError::InvalidPath { .. }));
    }

    #[test]
    fn truncate_output_keeps_utf8_and_limit() {
        let result = truncate_output("你好，世界".to_string(), 7);

        assert!(result.is_char_boundary(result.len()));
        assert!(result.len() <= 7);
        assert!(result.ends_with('…'));
    }
}
