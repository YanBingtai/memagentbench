//! Build the final assistant message from aggregate state.

use super::{tool_calls, StreamAggregate};
use crate::{
    model::{api::ModelUsage, error::ModelError},
    protocol::Message,
};

impl StreamAggregate {
    #[cfg(test)]
    pub(crate) fn finish(self) -> Result<Message, ModelError> {
        self.finish_with_usage().map(|(message, _)| message)
    }

    pub(crate) fn finish_with_usage(self) -> Result<(Message, Option<ModelUsage>), ModelError> {
        let StreamAggregate {
            content,
            reasoning_content,
            tools,
            finish_reason,
            usage,
            ..
        } = self;
        let tool_calls = tool_calls::finish(tools, finish_reason.as_deref())?;
        let mut message = Message::assistant((!content.is_empty()).then_some(content), tool_calls);
        message.reasoning_content = (!reasoning_content.is_empty()).then_some(reasoning_content);
        message.stop_reason = finish_reason;
        Ok((message, usage))
    }
}
