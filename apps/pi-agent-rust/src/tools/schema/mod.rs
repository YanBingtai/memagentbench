//! Small, provider-neutral JSON schema validation facade.
//!
//! Registration validates schema definitions; execution validates one
//! already-parsed argument value. Shared scalar and error helpers stay here
//! so both paths report the same types and limits.

mod definition;
mod value;

use std::{cmp::Ordering as CompareOrdering, collections::BTreeSet};

use serde_json::Value;

use super::{errors::ToolError, limits::*};

#[derive(Debug)]
pub(crate) struct SchemaProblem {
    pub(super) path: String,
    pub(super) reason: String,
}

pub(crate) fn argument_depth(value: &Value, depth: usize) -> usize {
    match value {
        Value::Array(values) => values
            .iter()
            .map(|value| argument_depth(value, depth.saturating_add(1)))
            .max()
            .unwrap_or(depth),
        Value::Object(values) => values
            .values()
            .map(|value| argument_depth(value, depth.saturating_add(1)))
            .max()
            .unwrap_or(depth),
        _ => depth,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntegerValue {
    Signed(i64),
    Unsigned(u64),
}

pub(crate) fn integer_value(value: &Value) -> Option<IntegerValue> {
    value
        .as_i64()
        .map(IntegerValue::Signed)
        .or_else(|| value.as_u64().map(IntegerValue::Unsigned))
}

pub(crate) fn compare_integers(left: IntegerValue, right: IntegerValue) -> CompareOrdering {
    match (left, right) {
        (IntegerValue::Signed(left), IntegerValue::Signed(right)) => left.cmp(&right),
        (IntegerValue::Unsigned(left), IntegerValue::Unsigned(right)) => left.cmp(&right),
        (IntegerValue::Signed(left), IntegerValue::Unsigned(right)) => {
            if left < 0 {
                CompareOrdering::Less
            } else {
                (left as u64).cmp(&right)
            }
        }
        (IntegerValue::Unsigned(left), IntegerValue::Signed(right)) => {
            if right < 0 {
                CompareOrdering::Greater
            } else {
                left.cmp(&(right as u64))
            }
        }
    }
}

pub(crate) fn json_value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

pub(crate) fn schema_type_matches_value(type_name: &str, value: &Value) -> bool {
    match type_name {
        "integer" => integer_value(value).is_some(),
        "number" => value.is_number(),
        _ => json_value_type(value) == type_name,
    }
}

pub(crate) use definition::validate_schema_definition;
pub(crate) use value::validate_schema_arguments;
#[allow(unused_imports)]
pub(crate) use value::validate_schema_value;
