use serde::Deserialize;

use crate::{model::api::ModelUsage, protocol::Message};

#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionResponse {
    pub(crate) choices: Vec<Choice>,
    #[serde(default)]
    pub(crate) usage: Option<ModelUsage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Choice {
    pub(crate) message: Message,
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
}
