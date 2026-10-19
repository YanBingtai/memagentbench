//! Output-size bounding for structured tool results.

use serde_json::{json, Value};

use super::ToolResult;
use crate::tools::limits::truncate_output;

impl ToolResult {
    /// Bound content and structured details before they cross the agent boundary.
    pub fn bounded(mut self, max_bytes: usize) -> Self {
        self.content = truncate_output(self.content, max_bytes);
        let details_bytes = serde_json::to_vec(&self.details)
            .map(|value| value.len())
            .unwrap_or(usize::MAX);
        if details_bytes > max_bytes {
            let preserved_status = self
                .details
                .get("status")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .filter(|status| {
                    matches!(
                        status.as_str(),
                        "cancelled"
                            | "timed_out"
                            | "unknown"
                            | "rejected"
                            | "truncated"
                            | "max_steps"
                            | "reconciled"
                    )
                });
            let summary = match preserved_status.as_deref() {
                Some(status) => json!({
                    "status": status,
                    "details_truncated": true,
                    "original_bytes": details_bytes,
                    "limit_bytes": max_bytes,
                }),
                None => json!({
                    "status": "details_truncated",
                    "original_bytes": details_bytes,
                    "limit_bytes": max_bytes,
                }),
            };
            self.details = if max_bytes > 0
                && serde_json::to_vec(&summary)
                    .map(|value| value.len() <= max_bytes)
                    .unwrap_or(false)
            {
                summary
            } else if let Some(status) = preserved_status.as_deref() {
                let minimal = json!({"status": status});
                if serde_json::to_vec(&minimal)
                    .map(|value| value.len() <= max_bytes)
                    .unwrap_or(false)
                {
                    minimal
                } else {
                    Value::Null
                }
            } else {
                Value::Null
            };
        }
        self
    }
}
