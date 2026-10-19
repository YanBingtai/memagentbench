//! Structured output returned by tool executions.

mod bounded;

use std::{future::Future, pin::Pin};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::super::errors::ToolError;

pub type ToolResultFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ToolResult, ToolError>> + Send + 'a>>;

/// Structured output from a tool execution.
///
/// `content` is the compatibility text sent back to the model. `details` is
/// kept for event consumers and runtime auditing, while `terminate` lets a
/// tool finish the current agent run without asking the model for another
/// turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub content: String,
    #[serde(default)]
    pub details: Value,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub terminate: bool,
}

impl ToolResult {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            details: Value::Null,
            is_error: false,
            terminate: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            is_error: true,
            ..Self::text(content)
        }
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }

    pub fn terminating(mut self) -> Self {
        self.terminate = true;
        self
    }
}
