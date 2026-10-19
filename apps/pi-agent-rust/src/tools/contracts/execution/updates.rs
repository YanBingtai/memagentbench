//! Progress updates emitted while a tool is running.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::super::result::ToolResult;

/// An incremental update emitted while a tool is running.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolUpdate {
    pub content: String,
    #[serde(default)]
    pub details: Value,
}

impl ToolUpdate {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            details: Value::Null,
        }
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }

    /// Bound a progress update before it reaches an event sink.
    pub fn bounded(self, max_bytes: usize) -> Self {
        let result = ToolResult::text(self.content)
            .with_details(self.details)
            .bounded(max_bytes);
        Self {
            content: result.content,
            details: result.details,
        }
    }
}

/// Receives bounded progress updates from a running tool.
pub type ToolUpdateSink = std::sync::Arc<dyn Fn(ToolUpdate) + Send + Sync>;
