use std::collections::BTreeMap;

use serde::Deserialize;

use crate::protocol::{FunctionCall, Message, ToolCall};

use super::super::{
    api::{
        DeltaCallback, ModelUsage, ReasoningDeltaCallback, ToolCallDelta, ToolCallDeltaCallback,
    },
    error::ModelError,
    limits::*,
};

#[derive(Debug, Default)]
pub(crate) struct StreamToolAccumulator {
    id: String,
    call_type: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Default)]
pub(crate) struct StreamAggregate {
    content: String,
    content_bytes: usize,
    reasoning_content: String,
    reasoning_bytes: usize,
    tools: BTreeMap<usize, StreamToolAccumulator>,
    tool_bytes: usize,
    finish_reason: Option<String>,
    usage: Option<ModelUsage>,
    /// Set only after the provider sends the SSE `[DONE]` marker.
    /// `finish_reason` is metadata for the assistant message and does not
    /// prove that the transport reached a clean terminal boundary.
    pub(crate) done_marker: bool,
}

impl StreamAggregate {
    #[cfg(test)]
    pub(crate) fn apply_json(
        &mut self,
        payload: &str,
        on_delta: DeltaCallback<'_>,
    ) -> Result<(), ModelError> {
        fn ignore_reasoning(_: &str) {}
        fn ignore_tool_call(_: &ToolCallDelta) {}
        let mut ignored = ignore_reasoning;
        let mut ignored_tool_call = ignore_tool_call;
        self.apply_json_with_reasoning_and_tools(
            payload,
            on_delta,
            &mut ignored,
            &mut ignored_tool_call,
        )
    }

    #[allow(dead_code)]
    fn apply_json_with_reasoning(
        &mut self,
        payload: &str,
        on_delta: DeltaCallback<'_>,
        on_reasoning: ReasoningDeltaCallback<'_>,
    ) -> Result<(), ModelError> {
        fn ignore_tool_call(_: &ToolCallDelta) {}
        let mut ignored_tool_call = ignore_tool_call;
        self.apply_json_with_reasoning_and_tools(
            payload,
            on_delta,
            on_reasoning,
            &mut ignored_tool_call,
        )
    }

    pub(crate) fn apply_json_with_reasoning_and_tools(
        &mut self,
        payload: &str,
        on_delta: DeltaCallback<'_>,
        on_reasoning: ReasoningDeltaCallback<'_>,
        on_tool_call: ToolCallDeltaCallback<'_>,
    ) -> Result<(), ModelError> {
        let envelope: StreamEnvelope = serde_json::from_str(payload).map_err(ModelError::Decode)?;
        if let Some(error) = envelope.error {
            return Err(ModelError::StreamProtocol(
                error
                    .message
                    .unwrap_or_else(|| "provider returned a stream error".to_string()),
            ));
        }
        if envelope.usage.is_some() {
            self.usage = envelope.usage;
        }

        for choice in envelope.choices {
            if choice.index != 0 {
                continue;
            }
            if let Some(delta) = choice.delta {
                if let Some(reasoning) = delta.reasoning_content {
                    append_bounded(
                        &mut self.reasoning_content,
                        &reasoning,
                        &mut self.reasoning_bytes,
                        MAX_STREAM_CONTENT_BYTES,
                        "stream reasoning content",
                    )?;
                    on_reasoning(&reasoning);
                }
                if let Some(content) = delta.content {
                    append_bounded(
                        &mut self.content,
                        &content,
                        &mut self.content_bytes,
                        MAX_STREAM_CONTENT_BYTES,
                        "stream content",
                    )?;
                    on_delta(&content);
                }
                if let Some(tool_calls) = delta.tool_calls {
                    for tool_call in tool_calls {
                        self.apply_tool_delta(tool_call, on_tool_call)?;
                    }
                }
            }
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }
        }
        Ok(())
    }

    pub(crate) fn apply_tool_delta(
        &mut self,
        delta: StreamToolCallDelta,
        on_tool_call: ToolCallDeltaCallback<'_>,
    ) -> Result<(), ModelError> {
        let callback_delta = ToolCallDelta {
            index: delta.index,
            id: delta.id.clone(),
            call_type: delta.call_type.clone(),
            name: delta
                .function
                .as_ref()
                .and_then(|function| function.name.clone()),
            arguments: delta
                .function
                .as_ref()
                .and_then(|function| function.arguments.clone()),
        };
        on_tool_call(&callback_delta);

        if delta.index > MAX_STREAM_TOOL_INDEX {
            return Err(ModelError::StreamLimit {
                limit: "tool call index",
            });
        }
        if !self.tools.contains_key(&delta.index) && self.tools.len() >= MAX_STREAM_TOOL_COUNT {
            return Err(ModelError::StreamLimit {
                limit: "tool call count",
            });
        }
        let tool = self.tools.entry(delta.index).or_default();
        if let Some(value) = delta.id {
            merge_metadata(&mut tool.id, &value, "tool call id")?;
        }
        if let Some(value) = delta.call_type {
            merge_metadata(&mut tool.call_type, &value, "tool call type")?;
        }
        if let Some(function) = delta.function {
            if let Some(value) = function.name {
                append_fragment(
                    &mut tool.name,
                    &value,
                    &mut self.tool_bytes,
                    MAX_STREAM_TOOL_BYTES,
                    "tool arguments",
                )?;
            }
            if let Some(value) = function.arguments {
                append_bounded(
                    &mut tool.arguments,
                    &value,
                    &mut self.tool_bytes,
                    MAX_STREAM_TOOL_BYTES,
                    "tool arguments",
                )?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn finish(self) -> Result<Message, ModelError> {
        self.finish_with_usage().map(|(message, _)| message)
    }

    pub(crate) fn finish_with_usage(self) -> Result<(Message, Option<ModelUsage>), ModelError> {
        let usage = self.usage;
        let incomplete_tool_metadata = self.tools.values().any(|tool| {
            tool.id.is_empty()
                || tool.call_type.is_empty()
                || tool.call_type != "function"
                || tool.name.is_empty()
        });
        let discard_incomplete_tools = incomplete_tool_metadata
            && matches!(
                self.finish_reason.as_deref(),
                Some("length" | "content_filter")
            );
        let tool_calls = if self.tools.is_empty() || discard_incomplete_tools {
            None
        } else {
            let mut calls = Vec::with_capacity(self.tools.len());
            for (_, tool) in self.tools {
                if tool.id.is_empty() || tool.call_type.is_empty() || tool.name.is_empty() {
                    return Err(ModelError::StreamProtocol(
                        "tool call is missing id, type, or function name".to_string(),
                    ));
                }
                if tool.call_type != "function" {
                    return Err(ModelError::StreamProtocol(format!(
                        "unsupported tool call type {:?}",
                        tool.call_type
                    )));
                }
                calls.push(ToolCall {
                    id: tool.id,
                    call_type: tool.call_type,
                    function: FunctionCall {
                        name: tool.name,
                        arguments: tool.arguments,
                    },
                });
            }
            Some(calls)
        };

        let mut message = Message::assistant(
            (!self.content.is_empty()).then_some(self.content),
            tool_calls,
        );
        message.reasoning_content =
            (!self.reasoning_content.is_empty()).then_some(self.reasoning_content);
        message.stop_reason = self.finish_reason;
        Ok((message, usage))
    }
}

fn append_bounded(
    target: &mut String,
    value: &str,
    total: &mut usize,
    limit: usize,
    label: &'static str,
) -> Result<(), ModelError> {
    *total = total
        .checked_add(value.len())
        .ok_or(ModelError::StreamLimit { limit: label })?;
    if *total > limit {
        return Err(ModelError::StreamLimit { limit: label });
    }
    target.push_str(value);
    Ok(())
}

fn merge_metadata(target: &mut String, value: &str, field: &'static str) -> Result<(), ModelError> {
    if value.is_empty() {
        return Ok(());
    }
    if target.is_empty() {
        target.push_str(value);
    } else if target != value {
        return Err(ModelError::StreamProtocol(format!(
            "conflicting {field}: existing {:?}, received {:?}",
            target, value
        )));
    }
    Ok(())
}

fn append_fragment(
    target: &mut String,
    value: &str,
    total: &mut usize,
    limit: usize,
    label: &'static str,
) -> Result<(), ModelError> {
    if value.is_empty() || value == target {
        return Ok(());
    }
    let fragment = value.strip_prefix(target.as_str()).unwrap_or(value);
    append_bounded(target, fragment, total, limit, label)
}

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
