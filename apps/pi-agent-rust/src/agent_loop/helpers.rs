use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::{
    config::RunPolicy,
    message::ToolCall,
    tools::{PreparedToolCall, ToolGateway},
};

use super::AgentError;

pub(super) fn timeout_error(timeout: Duration) -> AgentError {
    AgentError::RunTimedOut {
        timeout_ms: timeout_millis(timeout),
    }
}

pub(crate) fn model_retry_delay_ms(base_ms: u64, retry_count: usize) -> u64 {
    let shift = retry_count.saturating_sub(1).min(6) as u32;
    base_ms.saturating_mul(1_u64 << shift).min(30_000)
}

pub(super) fn unix_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64)
        .unwrap_or(0)
}

pub(super) fn timeout_millis(timeout: Duration) -> u64 {
    timeout.as_millis().min(u64::MAX as u128) as u64
}

pub(super) fn elapsed_millis(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u64::MAX as u128) as u64
}

pub(super) fn preflight_model_tool_call(
    tools: &ToolGateway,
    tool_call: &ToolCall,
) -> (Value, Option<PreparedToolCall>, Option<String>) {
    let arguments = match serde_json::from_str::<Value>(&tool_call.function.arguments) {
        Ok(value) if value.is_object() => value,
        Ok(_) => {
            return (
                Value::Null,
                None,
                Some("tool arguments must be a JSON object".to_string()),
            );
        }
        Err(_) => {
            return (
                Value::Null,
                None,
                Some("tool arguments must be valid JSON".to_string()),
            );
        }
    };

    match tools.prepare_model_call(tool_call, arguments.clone()) {
        Ok(prepared) => {
            let arguments = prepared.arguments().clone();
            (arguments, Some(prepared), None)
        }
        Err(error) => (
            arguments,
            None,
            Some(format!("invalid tool arguments: {error}")),
        ),
    }
}

pub(super) fn validate_policy(policy: RunPolicy) -> Result<(), AgentError> {
    if policy.max_steps == 0 {
        return Err(AgentError::InvalidMaxSteps);
    }
    if policy.max_tool_calls_per_step == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "max_tool_calls_per_step",
        });
    }
    if policy.max_parallel_tool_calls == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "max_parallel_tool_calls",
        });
    }
    if policy.tool_timeout_secs == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "tool_timeout_secs",
        });
    }
    if policy.run_timeout_secs == Some(0) {
        return Err(AgentError::InvalidRunPolicy {
            field: "run_timeout_secs",
        });
    }
    if policy.max_context_messages == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "max_context_messages",
        });
    }
    if policy.max_context_bytes == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "max_context_bytes",
        });
    }
    Ok(())
}
