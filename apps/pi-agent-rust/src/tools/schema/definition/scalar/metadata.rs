use serde_json::{Map, Value};

use crate::tools::schema::{json_value_type, schema_type_matches_value, SchemaProblem};

pub(in crate::tools::schema::definition) fn validate_description(
    object: &Map<String, Value>,
    path: &str,
) -> Result<(), SchemaProblem> {
    if let Some(description) = object.get("description") {
        if !description.is_string() {
            return Err(SchemaProblem::new(
                format!("{path}.description"),
                "description must be a string",
            ));
        }
    }
    Ok(())
}

pub(in crate::tools::schema::definition) fn validate_enum(
    object: &Map<String, Value>,
    type_name: &str,
    path: &str,
) -> Result<(), SchemaProblem> {
    let Some(enum_values) = object.get("enum") else {
        return Ok(());
    };
    let enum_values = enum_values.as_array().ok_or_else(|| {
        SchemaProblem::new(format!("{path}.enum"), "enum must be a non-empty array")
    })?;
    if enum_values.is_empty() {
        return Err(SchemaProblem::new(
            format!("{path}.enum"),
            "enum must be a non-empty array",
        ));
    }
    for (index, value) in enum_values.iter().enumerate() {
        if !schema_type_matches_value(type_name, value) {
            return Err(SchemaProblem::new(
                format!("{path}.enum[{index}]"),
                format!(
                    "enum value has type {}, expected {type_name}",
                    json_value_type(value)
                ),
            ));
        }
        if enum_values[..index].contains(value) {
            return Err(SchemaProblem::new(
                format!("{path}.enum[{index}]"),
                "enum values must be unique",
            ));
        }
        if type_name == "string" {
            let length = value
                .as_str()
                .expect("string enum type was checked above")
                .chars()
                .count() as u64;
            if object
                .get("minLength")
                .and_then(Value::as_u64)
                .is_some_and(|minimum| length < minimum)
                || object
                    .get("maxLength")
                    .and_then(Value::as_u64)
                    .is_some_and(|maximum| length > maximum)
            {
                return Err(SchemaProblem::new(
                    format!("{path}.enum[{index}]"),
                    "enum value violates the string length bounds",
                ));
            }
        }
    }
    Ok(())
}
