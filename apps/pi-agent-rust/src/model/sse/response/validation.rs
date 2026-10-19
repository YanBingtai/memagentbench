use crate::{
    model::{
        error::ModelError,
        limits::{MAX_STREAM_TOOL_BYTES, MAX_STREAM_TOOL_COUNT},
    },
    protocol::Message,
};

pub(crate) fn validate_non_streaming_message(message: &Message) -> Result<(), ModelError> {
    let Some(tool_calls) = &message.tool_calls else {
        return Ok(());
    };
    if tool_calls.len() > MAX_STREAM_TOOL_COUNT {
        return Err(ModelError::ToolCallCountLimit {
            limit: MAX_STREAM_TOOL_COUNT,
        });
    }
    for call in tool_calls {
        if call.function.arguments.len() > MAX_STREAM_TOOL_BYTES {
            return Err(ModelError::ResponseLimit {
                limit: MAX_STREAM_TOOL_BYTES,
            });
        }
    }
    Ok(())
}
