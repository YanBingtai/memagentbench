//! Provider request projection and generation options.

mod choice;
mod projection;
mod request;
mod thinking;

pub use choice::ToolChoice;
pub(crate) use projection::{provider_messages, ProviderMessage};
pub(crate) use request::{ChatCompletionRequest, ChatTemplateKwargs};
pub(crate) use thinking::{
    provider_request_body, thinking_enabled_for_request, validate_chat_options,
};

/// Generation settings understood by OpenAI-compatible local model servers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChatOptions {
    pub temperature: f32,
    pub enable_thinking: Option<bool>,
}
