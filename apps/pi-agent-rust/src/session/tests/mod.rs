use super::limits::MAX_SESSION_RECORD_BYTES;
use super::*;
use crate::message::{FunctionCall, Message, ToolCall};
use tempfile::tempdir;
use tokio::fs;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

fn tool_call(id: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "read_file".to_string(),
            arguments: r#"{"path":"note.txt"}"#.to_string(),
        },
    }
}

mod repair;
mod store;
mod validation;
