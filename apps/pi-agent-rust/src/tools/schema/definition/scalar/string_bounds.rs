use serde_json::{Map, Value};

use crate::tools::schema::SchemaProblem;

pub(in crate::tools::schema::definition) fn validate_string_bounds(
    object: &Map<String, Value>,
    type_name: &str,
    path: &str,
) -> Result<(), SchemaProblem> {
    if let Some(min_length) = object.get("minLength") {
        if type_name != "string" || min_length.as_u64().is_none() {
            return Err(SchemaProblem::new(
                format!("{path}.minLength"),
                "minLength must be a non-negative integer on a string schema",
            ));
        }
    }
    if let Some(max_length) = object.get("maxLength") {
        if type_name != "string" || max_length.as_u64().is_none() {
            return Err(SchemaProblem::new(
                format!("{path}.maxLength"),
                "maxLength must be a non-negative integer on a string schema",
            ));
        }
    }
    if let (Some(min_length), Some(max_length)) = (
        object.get("minLength").and_then(Value::as_u64),
        object.get("maxLength").and_then(Value::as_u64),
    ) {
        if min_length > max_length {
            return Err(SchemaProblem::new(
                path,
                "minLength must not exceed maxLength",
            ));
        }
    }
    Ok(())
}
