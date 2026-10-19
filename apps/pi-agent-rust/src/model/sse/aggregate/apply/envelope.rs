//! Route one decoded response envelope to the aggregate.

use super::super::{StreamAggregate, StreamChoice, StreamEnvelope};
use super::delta;
use crate::model::{
    api::{DeltaCallback, ReasoningDeltaCallback, ToolCallDeltaCallback},
    error::ModelError,
};

pub(super) fn apply_json(
    aggregate: &mut StreamAggregate,
    payload: &str,
    on_delta: DeltaCallback<'_>,
    on_reasoning: ReasoningDeltaCallback<'_>,
    on_tool_call: ToolCallDeltaCallback<'_>,
) -> Result<(), ModelError> {
    let envelope: StreamEnvelope = serde_json::from_str(payload).map_err(ModelError::Decode)?;
    apply_envelope(aggregate, envelope, on_delta, on_reasoning, on_tool_call)
}

fn apply_envelope(
    aggregate: &mut StreamAggregate,
    envelope: StreamEnvelope,
    on_delta: DeltaCallback<'_>,
    on_reasoning: ReasoningDeltaCallback<'_>,
    on_tool_call: ToolCallDeltaCallback<'_>,
) -> Result<(), ModelError> {
    if let Some(error) = envelope.error {
        return Err(ModelError::StreamProtocol(
            error
                .message
                .unwrap_or_else(|| "provider returned a stream error".to_string()),
        ));
    }
    if envelope.usage.is_some() {
        aggregate.usage = envelope.usage;
    }

    for choice in envelope.choices {
        apply_choice(aggregate, choice, on_delta, on_reasoning, on_tool_call)?;
    }
    Ok(())
}

fn apply_choice(
    aggregate: &mut StreamAggregate,
    choice: StreamChoice,
    on_delta: DeltaCallback<'_>,
    on_reasoning: ReasoningDeltaCallback<'_>,
    on_tool_call: ToolCallDeltaCallback<'_>,
) -> Result<(), ModelError> {
    if choice.index != 0 {
        return Ok(());
    }
    if let Some(delta) = choice.delta {
        delta::apply_delta(aggregate, delta, on_delta, on_reasoning, on_tool_call)?;
    }
    if let Some(reason) = choice.finish_reason {
        aggregate.finish_reason = Some(reason);
    }
    Ok(())
}
