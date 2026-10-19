use super::super::*;

pub(crate) fn streaming_response(message: Message) -> ResponseTemplate {
    streaming_response_with_optional_usage(message, None)
}

pub(crate) fn streaming_response_with_usage(message: Message, usage: Value) -> ResponseTemplate {
    streaming_response_with_optional_usage(message, Some(usage))
}

pub(crate) fn streaming_response_with_optional_usage(
    message: Message,
    usage: Option<Value>,
) -> ResponseTemplate {
    let mut delta = serde_json::to_value(message.clone()).expect("message should serialize");
    if let Some(tool_calls) = delta
        .get_mut("tool_calls")
        .and_then(|value| value.as_array_mut())
    {
        for (index, tool_call) in tool_calls.iter_mut().enumerate() {
            tool_call["index"] = json!(index);
        }
    }
    let finish_reason = if message
        .tool_calls
        .as_ref()
        .is_some_and(|calls| !calls.is_empty())
    {
        "tool_calls"
    } else {
        "stop"
    };
    let mut event = json!({
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    });
    if let Some(usage) = usage {
        event["usage"] = usage;
    }
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(format!(
            "data: {}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&event).expect("stream event should serialize")
        ))
}
