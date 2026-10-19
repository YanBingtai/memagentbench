use crate::protocol::{Message, ToolCall};

/// Provider-facing projection of a transcript message.
///
/// `Message` also carries agent-only tool metadata for session recovery and
/// event consumers. That metadata must not leak into an OpenAI-compatible
/// request body, so requests are serialized through this explicit projection.
#[derive(Debug, serde::Serialize)]
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
    fn from_message(message: &'a Message, include_reasoning: bool) -> Self {
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
        .map(|message| ProviderMessage::from_message(message, include_reasoning))
        .collect()
}
