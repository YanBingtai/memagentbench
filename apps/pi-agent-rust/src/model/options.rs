//! Provider request projection and generation options.

use serde::{Serialize, Serializer};
use serde_json::{json, Value};

use crate::protocol::{Message, ToolCall, ToolDefinition};

use super::error::ModelError;

/// Generation settings understood by OpenAI-compatible local model servers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChatOptions {
    pub temperature: f32,
    pub enable_thinking: Option<bool>,
}

/// OpenAI-compatible policy for selecting tools during a chat completion.
///
/// `Function` uses the object form required by the Chat Completions API:
/// `{ "type": "function", "function": { "name": "..." } }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function { name: String },
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function { name } => serde_json::json!({
                "type": "function",
                "function": { "name": name },
            })
            .serialize(serializer),
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatCompletionRequest<'a> {
    pub(crate) model: &'a str,
    pub(crate) messages: &'a [ProviderMessage<'a>],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<&'a [ToolDefinition]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<ToolChoice>,
    pub(crate) temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) chat_template_kwargs: Option<ChatTemplateKwargs>,
}

/// Provider-facing projection of a transcript message.
///
/// `Message` also carries agent-only tool metadata for session recovery and
/// event consumers. That metadata must not leak into an OpenAI-compatible
/// request body, so requests are serialized through this explicit projection.
#[derive(Debug, Serialize)]
pub(crate) struct ProviderMessage<'a> {
    pub(crate) role: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<&'a String>,
    /// Qwen-style thinking providers need the prior reasoning channel to
    /// continue a tool-calling turn. Keep it opt-in so ordinary providers do
    /// not receive agent-internal reasoning content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_content: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_id: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<&'a Vec<ToolCall>>,
}

impl<'a> ProviderMessage<'a> {
    fn new(message: &'a Message, include_reasoning: bool) -> Self {
        Self {
            role: &message.role,
            content: message.content.as_ref(),
            reasoning_content: if include_reasoning && message.role == "assistant" {
                message.reasoning_content.as_ref()
            } else {
                None
            },
            name: message.name.as_ref(),
            tool_call_id: message.tool_call_id.as_ref(),
            tool_calls: message.tool_calls.as_ref(),
        }
    }
}

pub(crate) fn provider_messages(
    messages: &[Message],
    include_reasoning: bool,
) -> Vec<ProviderMessage<'_>> {
    messages
        .iter()
        .map(|message| ProviderMessage::new(message, include_reasoning))
        .collect()
}

pub(crate) fn thinking_enabled_for_request(model: &str, configured: Option<bool>) -> bool {
    configured.unwrap_or_else(|| model.starts_with("mimo-"))
}

pub(crate) fn validate_chat_options(options: ChatOptions) -> Result<(), ModelError> {
    if !options.temperature.is_finite() || !(0.0..=2.0).contains(&options.temperature) {
        return Err(ModelError::InvalidTemperature);
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatTemplateKwargs {
    pub(crate) enable_thinking: bool,
}

/// Add provider-specific request fields without coupling the agent loop to a
/// model vendor. Qwen-compatible local servers use `chat_template_kwargs`,
/// while MiMo exposes the same choice as `thinking.type`.
pub(crate) fn provider_request_body(
    model: &str,
    request: &ChatCompletionRequest<'_>,
    enable_thinking: Option<bool>,
) -> Result<Value, ModelError> {
    let mut body = serde_json::to_value(request).map_err(ModelError::Encode)?;
    if model.starts_with("mimo-") {
        if matches!(
            request.tool_choice,
            Some(ToolChoice::None) | Some(ToolChoice::Required) | Some(ToolChoice::Function { .. })
        ) {
            return Err(ModelError::UnsupportedToolChoice {
                model: model.to_owned(),
            });
        }
        let enabled = thinking_enabled_for_request(model, enable_thinking);
        body["thinking"] = json!({
            "type": if enabled { "enabled" } else { "disabled" },
        });
        // MiMo does not use the Qwen-specific extension. Removing it also
        // keeps the request valid for providers that reject unknown keys.
        if let Some(fields) = body.as_object_mut() {
            fields.remove("chat_template_kwargs");
            if enabled {
                // MiMo fixes sampling defaults while thinking is enabled.
                fields.remove("temperature");
            }
        }
    }
    Ok(body)
}
