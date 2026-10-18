//! Typed errors for durable session storage.

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("failed to create session directory {path}: {source}")]
    CreateDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to open session file {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to read session file {path} at line {line}: {source}")]
    Read {
        path: PathBuf,
        line: usize,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid JSON in session file {path} at line {line}: {source}")]
    Decode {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to encode session record: {0}")]
    Encode(#[source] serde_json::Error),

    #[error("failed to append session file {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to lock session file {path}: {source}")]
    Lock {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("session record at {path} line {line} exceeds the {limit}-byte limit")]
    RecordTooLarge {
        path: PathBuf,
        line: usize,
        limit: usize,
    },

    #[error("session contains incomplete tool calls outside the transcript tail: {call_ids:?}")]
    OrphanToolCall { call_ids: Vec<String> },

    #[error("tool result is missing its tool_call_id")]
    ToolResultMissingCallId,

    #[error("tool result refers to unknown tool call {call_id:?}")]
    UnexpectedToolResult { call_id: String },

    #[error("tool call {call_id:?} has more than one result")]
    DuplicateToolResult { call_id: String },

    #[error("tool call id {call_id:?} is declared more than once")]
    DuplicateToolCall { call_id: String },

    #[error("system clock is before the Unix epoch: {0}")]
    Clock(#[source] std::time::SystemTimeError),
}
