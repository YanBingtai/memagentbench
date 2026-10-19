use crate::protocol::Message;

/// Normalize providers that put a complete thinking block in `content`.
///
/// Some local servers omit the opening `<think>` tag but keep the closing tag,
/// so the closing marker is the reliable boundary here.
pub(crate) fn normalize_inline_thinking(mut message: Message) -> Message {
    if message.role != "assistant" {
        return message;
    }
    let Some(content) = message.content.take() else {
        return message;
    };
    let Some(end) = content.find("</think>") else {
        message.content = Some(content);
        return message;
    };

    let mut reasoning = content[..end].trim();
    if let Some(stripped) = reasoning.strip_prefix("<think>") {
        reasoning = stripped.trim();
    }
    if message.reasoning_content.is_none() && !reasoning.is_empty() {
        message.reasoning_content = Some(reasoning.to_owned());
    }

    let answer = content[end + "</think>".len()..].trim_start();
    message.content = (!answer.is_empty()).then(|| answer.to_owned());
    message
}
