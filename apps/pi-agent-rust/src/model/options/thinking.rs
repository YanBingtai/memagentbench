use serde_json::{json, Value};

use crate::model::error::ModelError;

use super::{ChatCompletionRequest, ChatOptions, ToolChoice};

pub(crate) fn thinking_enabled_for_request(model: &str, configured: Option<bool>) -> bool {
    configured.unwrap_or_else(|| model.starts_with("mimo-"))
}

pub(crate) fn validate_chat_options(options: ChatOptions) -> Result<(), ModelError> {
    if !options.temperature.is_finite() || !(0.0..=2.0).contains(&options.temperature) {
        return Err(ModelError::InvalidTemperature);
    }
    Ok(())
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
    if !model.starts_with("mimo-") {
        return Ok(body);
    }

    reject_unsupported_tool_choice(model, request.tool_choice.as_ref())?;

    let enabled = thinking_enabled_for_request(model, enable_thinking);
    body["thinking"] = json!({
        "type": if enabled { "enabled" } else { "disabled" },
    });
    remove_qwen_fields(&mut body, enabled);
    Ok(body)
}

fn reject_unsupported_tool_choice(
    model: &str,
    tool_choice: Option<&ToolChoice>,
) -> Result<(), ModelError> {
    if matches!(
        tool_choice,
        Some(ToolChoice::None | ToolChoice::Required | ToolChoice::Function { .. })
    ) {
        return Err(ModelError::UnsupportedToolChoice {
            model: model.to_owned(),
        });
    }
    Ok(())
}

fn remove_qwen_fields(body: &mut Value, thinking_enabled: bool) {
    let Some(fields) = body.as_object_mut() else {
        return;
    };
    fields.remove("chat_template_kwargs");
    if thinking_enabled {
        // MiMo fixes sampling defaults while thinking is enabled.
        fields.remove("temperature");
    }
}
