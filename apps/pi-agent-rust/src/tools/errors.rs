//! Error types exposed by tool registration and execution.

use std::path::PathBuf;

use serde_json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("invalid tool call {call_id}: {reason}")]
    InvalidCall { call_id: String, reason: String },

    #[error("unknown tool: {name}")]
    UnknownTool { name: String },

    #[error("tool is not available to the model: {name}")]
    ToolNotModelVisible { name: String },

    #[error("invalid arguments for tool {name}: {source}")]
    InvalidArguments {
        name: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid argument for tool {name}: {reason}")]
    InvalidArgumentValue { name: String, reason: String },

    #[error("arguments for tool {name} violate its schema at {path}: {reason}")]
    SchemaViolation {
        name: String,
        path: String,
        reason: String,
    },

    #[error("arguments for tool {name} exceed the {limit}-byte limit ({size} bytes)")]
    ArgumentsTooLarge {
        name: String,
        size: usize,
        limit: usize,
    },

    #[error("arguments for tool {name} exceed the depth limit of {limit}")]
    ArgumentsTooDeep { name: String, limit: usize },

    #[error("invalid range for tool {tool}: {reason}")]
    InvalidRange { tool: String, reason: String },

    #[error("subagent failed: {message}")]
    Subagent { message: String },

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

    #[error("failed to read directory {path}: {source}")]
    ReadDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to write file {path}: {source}")]
    WriteFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to replace file {path} from temporary file {temporary}: {source}")]
    RenameFile {
        path: PathBuf,
        temporary: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("file is not valid UTF-8: {path}")]
    InvalidUtf8 { path: PathBuf },

    #[error("edit target was not found in {path}")]
    EditTargetNotFound { path: PathBuf },

    #[error("edit target must not be empty in {path}")]
    EditTargetEmpty { path: PathBuf },

    #[error("edit target occurs {matches} times in {path}; expected exactly once")]
    EditTargetNotUnique { path: PathBuf, matches: usize },

    #[error("unknown skill: {name}")]
    UnknownSkill { name: String },

    #[error("skill output is too large: {name} ({size} bytes, limit {limit} bytes)")]
    SkillTooLarge {
        name: String,
        size: usize,
        limit: usize,
    },

    #[error("failed to format skill {name}: {message}")]
    SkillInvocation { name: String, message: String },

    #[error("web search failed: {message}")]
    WebSearch { message: String },

    #[error("bash execution failed: {message}")]
    Shell { message: String },
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("a tool named {name} is already registered")]
    DuplicateTool { name: String },

    #[error("invalid schema for tool {name}: {reason}")]
    InvalidDefinition { name: String, reason: String },
}
