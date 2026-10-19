//! Apply decoded streaming responses to the aggregate state.

mod delta;
mod envelope;

use super::{protocol::StreamToolCallDelta, StreamAggregate};
use crate::model::api::{DeltaCallback, ReasoningDeltaCallback, ToolCallDeltaCallback};
use crate::model::error::ModelError;

impl StreamAggregate {
    #[cfg(test)]
    pub(crate) fn apply_json(
        &mut self,
        payload: &str,
        on_delta: DeltaCallback<'_>,
    ) -> Result<(), ModelError> {
        fn ignore_reasoning(_: &str) {}
        fn ignore_tool_call(_: &crate::model::api::ToolCallDelta) {}

        let mut ignored_reasoning = ignore_reasoning;
        let mut ignored_tool_call = ignore_tool_call;
        self.apply_json_with_reasoning_and_tools(
            payload,
            on_delta,
            &mut ignored_reasoning,
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
        fn ignore_tool_call(_: &crate::model::api::ToolCallDelta) {}

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
        envelope::apply_json(self, payload, on_delta, on_reasoning, on_tool_call)
    }

    pub(crate) fn apply_tool_delta(
        &mut self,
        delta: StreamToolCallDelta,
        on_tool_call: ToolCallDeltaCallback<'_>,
    ) -> Result<(), ModelError> {
        delta::apply_tool_delta(self, delta, on_tool_call)
    }
}
