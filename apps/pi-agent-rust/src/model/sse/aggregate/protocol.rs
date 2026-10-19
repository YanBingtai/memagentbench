//! JSON contracts emitted by an OpenAI-compatible streaming endpoint.

use serde::Deserialize;

use crate::model::api::ModelUsage;

#[derive(Debug, Deserialize)]
pub(crate) struct StreamEnvelope {
    #[serde(default)]
    pub(crate) choices: Vec<StreamChoice>,
    #[serde(default)]
    pub(crate) usage: Option<ModelUsage>,
    #[serde(default)]
    pub(crate) error: Option<StreamErrorBody>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamChoice {
    #[serde(default)]
    pub(crate) index: usize,
    #[serde(default)]
    pub(crate) delta: Option<StreamDelta>,
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamDelta {
    #[serde(default)]
    pub(crate) content: Option<String>,
    #[serde(default)]
    pub(crate) reasoning_content: Option<String>,
    #[serde(default)]
    pub(crate) tool_calls: Option<Vec<StreamToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamToolCallDelta {
    #[serde(default)]
    pub(crate) index: usize,
    #[serde(default, rename = "id")]
    pub(crate) id: Option<String>,
    #[serde(default, rename = "type")]
    pub(crate) call_type: Option<String>,
    #[serde(default)]
    pub(crate) function: Option<StreamFunctionDelta>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamFunctionDelta {
    #[serde(default)]
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamErrorBody {
    pub(crate) message: Option<String>,
}
