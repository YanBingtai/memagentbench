//! Implementation of `write_file` and its platform-specific atomic writer.

use std::{
    io,
    path::{Component, Path, PathBuf},
};

#[cfg(unix)]
use std::{fs::File, io::Write};

#[cfg(unix)]
use rustix::fs::{openat, renameat, unlinkat, AtFlags, Mode, OFlags};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::protocol::ToolDefinition;
use pi_agent_runtime::RiskClass;

use super::super::{errors::ToolError, registry::Tool, ToolConcurrency, ToolContext, ToolFuture};
use super::path::resolve_workspace_write_path;

#[derive(Debug, Clone)]
pub struct WriteFileTool {
    context: ToolContext,
}

impl WriteFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "write_file",
            "Atomically write UTF-8 text to a file inside the configured workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path of the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 file contents"
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        )
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let parsed: WriteFileArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "write_file".to_string(),
                    source,
                }
            })?;
        let size = parsed.content.len() as u64;
        if size > self.context.max_file_bytes as u64 {
            return Err(ToolError::FileTooLarge {
                path: parsed.path,
                size,
                limit: self.context.max_file_bytes,
            });
        }
        Ok(())
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::Reversible
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: WriteFileArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "write_file".to_string(),
                        source,
                    }
                })?;
            let size = arguments.content.len() as u64;
            if size > self.context.max_file_bytes as u64 {
                return Err(ToolError::FileTooLarge {
                    path: arguments.path.clone(),
                    size,
                    limit: self.context.max_file_bytes,
                });
            }

            atomic_write_workspace_file(
                &self.context.workspace,
                &arguments.path,
                arguments.content.as_bytes(),
            )
            .await?;

            Ok(format!(
                "Wrote {} bytes to {}",
                size,
                arguments.path.display()
            ))
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteFileArguments {
    path: PathBuf,
    content: String,
}

pub(super) async fn atomic_write_workspace_file(
    workspace: &Path,
    requested: &Path,
    content: &[u8],
) -> Result<(), ToolError> {
    let (target, temporary) = resolve_workspace_write_path(workspace, requested).await?;

    #[cfg(unix)]
    {
        let canonical_workspace = tokio::fs::canonicalize(workspace).await.map_err(|source| {
            ToolError::ResolveWorkspace {
                path: workspace.to_path_buf(),
                source,
            }
        })?;
        let parent = temporary
            .parent()
            .ok_or_else(|| ToolError::InvalidPath {
                tool: "write_file".to_string(),
                path: requested.to_path_buf(),
            })?
            .to_path_buf();
        let temporary_name = temporary
            .file_name()
            .ok_or_else(|| ToolError::InvalidPath {
                tool: "write_file".to_string(),
                path: requested.to_path_buf(),
            })?
            .to_os_string();
        let target_name = target
            .file_name()
            .ok_or_else(|| ToolError::InvalidPath {
                tool: "write_file".to_string(),
                path: requested.to_path_buf(),
            })?
            .to_os_string();
        let content = content.to_vec();
        tokio::task::spawn_blocking(move || {
            atomic_write_workspace_file_unix(
                &canonical_workspace,
                &parent,
                &temporary_name,
                &target_name,
                &content,
            )
        })
        .await
        .map_err(|error| ToolError::WriteFile {
            path: target.clone(),
            source: io::Error::other(format!("write worker failed: {error}")),
        })?
        .map_err(|source| ToolError::WriteFile {
            path: target,
            source,
        })?;
        Ok(())
    }

    #[cfg(not(unix))]
    {
        if let Err(source) = tokio::fs::write(&temporary, content).await {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(ToolError::WriteFile {
                path: temporary,
                source,
            });
        }

        if let Err(source) = tokio::fs::rename(&temporary, &target).await {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(ToolError::RenameFile {
                path: target,
                temporary,
                source,
            });
        }

        Ok(())
    }
}

#[cfg(unix)]
fn atomic_write_workspace_file_unix(
    workspace: &Path,
    parent: &Path,
    temporary_name: &std::ffi::OsStr,
    target_name: &std::ffi::OsStr,
    content: &[u8],
) -> io::Result<()> {
    let relative_parent = parent.strip_prefix(workspace).map_err(|_| {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "write parent escaped workspace",
        )
    })?;
    let mut directory = File::open(workspace)?;
    for component in relative_parent.components() {
        let Component::Normal(name) = component else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "write parent contains an invalid path component",
            ));
        };
        let next = openat(
            &directory,
            name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )?;
        directory = File::from(next);
    }

    let temporary = openat(
        &directory,
        temporary_name,
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o600),
    )?;
    let mut temporary = File::from(temporary);
    if let Err(error) = temporary
        .write_all(content)
        .and_then(|_| temporary.sync_all())
    {
        let _ = unlinkat(&directory, temporary_name, AtFlags::empty());
        return Err(error);
    }
    drop(temporary);
    if let Err(error) = renameat(&directory, temporary_name, &directory, target_name) {
        let _ = unlinkat(&directory, temporary_name, AtFlags::empty());
        return Err(error.into());
    }
    Ok(())
}
