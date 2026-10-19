use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Structured metadata retained alongside a tool result message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResultMetadata {
    #[serde(default)]
    pub details: Value,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub terminate: bool,
    /// Runtime action identity retained for session/runtime reconciliation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_id: Option<Uuid>,
    /// Runtime grant identity retained for session/runtime reconciliation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grant_id: Option<Uuid>,
}

impl ToolResultMetadata {
    pub fn new(details: Value, is_error: bool, terminate: bool) -> Self {
        Self {
            details,
            is_error,
            terminate,
            action_id: None,
            grant_id: None,
        }
    }

    pub fn with_runtime_identity(mut self, action_id: Uuid, grant_id: Uuid) -> Self {
        self.action_id = Some(action_id);
        self.grant_id = Some(grant_id);
        self
    }
}
