//! Execution-time validation of parsed tool arguments.

mod scalar;
mod structure;

use serde_json::Value;

use crate::protocol::ToolDefinition;
use crate::tools::errors::ToolError;

use super::{SchemaProblem, MAX_SCHEMA_DEPTH};

pub(crate) fn validate_schema_arguments(
    definition: &ToolDefinition,
    arguments: &Value,
) -> Result<(), ToolError> {
    validate_schema_value(&definition.function.parameters, arguments, "arguments", 0).map_err(
        |problem| ToolError::SchemaViolation {
            name: definition.function.name.clone(),
            path: problem.path,
            reason: problem.reason,
        },
    )
}

pub(crate) fn validate_schema_value(
    schema: &Value,
    value: &Value,
    path: &str,
    depth: usize,
) -> Result<(), SchemaProblem> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(SchemaProblem::new(
            path,
            format!("schema depth exceeds the limit of {MAX_SCHEMA_DEPTH}"),
        ));
    }
    let object = schema
        .as_object()
        .ok_or_else(|| SchemaProblem::new(path, "schema must be a JSON object"))?;
    let type_name = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| SchemaProblem::new(path, "schema must declare a type"))?;
    if !super::schema_type_matches_value(type_name, value) {
        return Err(SchemaProblem::new(
            path,
            format!(
                "expected {type_name}, got {}",
                super::json_value_type(value)
            ),
        ));
    }
    scalar::validate_enum(object, value, path)?;

    match type_name {
        "object" => structure::validate_object(object, value, path, depth),
        "array" => structure::validate_array(object, value, path, depth),
        "string" => scalar::validate_string(object, value, path),
        "integer" => scalar::validate_integer(object, value, path),
        "number" => scalar::validate_number(object, value, path),
        "boolean" | "null" => Ok(()),
        _ => unreachable!("schema definitions reject unsupported types"),
    }
}
