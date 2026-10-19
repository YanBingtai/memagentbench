//! Scalar constraints for execution-time schema validation.

use std::cmp::Ordering;

use serde_json::{Map, Value};

use crate::tools::schema::{compare_integers, integer_value, SchemaProblem};

pub(super) fn validate_enum(
    schema: &Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), SchemaProblem> {
    let Some(enum_values) = schema.get("enum").and_then(Value::as_array) else {
        return Ok(());
    };
    if enum_values.iter().any(|candidate| candidate == value) {
        return Ok(());
    }
    Err(SchemaProblem::new(
        path,
        "value is not one of the allowed enum values",
    ))
}

pub(super) fn validate_string(
    schema: &Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), SchemaProblem> {
    let length = value
        .as_str()
        .expect("string type was checked above")
        .chars()
        .count() as u64;
    if let Some(minimum) = schema.get("minLength").and_then(Value::as_u64) {
        if length < minimum {
            return Err(SchemaProblem::new(
                path,
                format!("string length must be at least {minimum}"),
            ));
        }
    }
    if let Some(maximum) = schema.get("maxLength").and_then(Value::as_u64) {
        if length > maximum {
            return Err(SchemaProblem::new(
                path,
                format!("string length must be at most {maximum}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_integer(
    schema: &Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), SchemaProblem> {
    let integer = integer_value(value).expect("integer type was checked above");
    if let Some(minimum) = schema.get("minimum").and_then(integer_value) {
        if compare_integers(integer, minimum) == Ordering::Less {
            return Err(SchemaProblem::new(
                path,
                format!("integer must be at least {minimum:?}"),
            ));
        }
    }
    if let Some(maximum) = schema.get("maximum").and_then(integer_value) {
        if compare_integers(integer, maximum) == Ordering::Greater {
            return Err(SchemaProblem::new(
                path,
                format!("integer must be at most {maximum:?}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_number(
    schema: &Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), SchemaProblem> {
    let number = value.as_f64().expect("numeric type was checked above");
    if let Some(minimum) = schema.get("minimum").and_then(Value::as_f64) {
        if number < minimum {
            return Err(SchemaProblem::new(
                path,
                format!("number must be at least {minimum}"),
            ));
        }
    }
    if let Some(maximum) = schema.get("maximum").and_then(Value::as_f64) {
        if number > maximum {
            return Err(SchemaProblem::new(
                path,
                format!("number must be at most {maximum}"),
            ));
        }
    }
    Ok(())
}
