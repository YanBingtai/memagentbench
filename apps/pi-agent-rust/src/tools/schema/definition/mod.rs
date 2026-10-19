//! Registration-time validation of the JSON Schema subset understood locally.

mod scalar;
mod structure;

use serde_json::Value;

use crate::protocol::ToolDefinition;

use super::SchemaProblem;

/// Validate the intentionally small schema subset understood by the agent.
///
/// The provider still receives the original JSON Schema. Locally we enforce
/// the object/primitive fields used by built-in tools, together with required
/// fields, additionalProperties, enum, string length bounds, numeric bounds,
/// and array items. Unsupported keywords are rejected at registration instead
/// of being silently ignored.
pub(crate) fn validate_schema_definition(definition: &ToolDefinition) -> Result<(), String> {
    validate_schema_node(&definition.function.parameters, "$.parameters", true, 0)
        .map_err(|problem| format!("{}: {}", problem.path, problem.reason))
}

pub(super) fn validate_schema_node(
    schema: &Value,
    path: &str,
    root: bool,
    depth: usize,
) -> Result<(), SchemaProblem> {
    if depth > super::MAX_SCHEMA_DEPTH {
        return Err(SchemaProblem::new(
            path,
            format!(
                "schema depth exceeds the limit of {}",
                super::MAX_SCHEMA_DEPTH
            ),
        ));
    }

    let object = schema
        .as_object()
        .ok_or_else(|| SchemaProblem::new(path, "schema must be a JSON object"))?;
    validate_keywords(object, path)?;

    let type_name = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| SchemaProblem::new(path, "schema must declare a type"))?;
    validate_type(type_name, path, root)?;
    scalar::validate_description(object, path)?;
    scalar::validate_enum(object, type_name, path)?;

    let properties = structure::properties(object, path)?;
    if properties.is_some() && type_name != "object" {
        return Err(SchemaProblem::new(
            path,
            "properties is only supported for object schemas",
        ));
    }
    structure::validate_properties(properties, path, depth)?;
    structure::validate_required(object, properties, type_name, path)?;
    structure::validate_additional_properties(object, type_name, path)?;
    scalar::validate_string_bounds(object, type_name, path)?;
    scalar::validate_numeric_bounds(object, type_name, path)?;
    structure::validate_items(object, type_name, path, depth)?;

    Ok(())
}

fn validate_keywords(
    object: &serde_json::Map<String, Value>,
    path: &str,
) -> Result<(), SchemaProblem> {
    for keyword in object.keys() {
        if !matches!(
            keyword.as_str(),
            "type"
                | "properties"
                | "required"
                | "additionalProperties"
                | "enum"
                | "minimum"
                | "maximum"
                | "minLength"
                | "maxLength"
                | "items"
                | "description"
        ) {
            return Err(SchemaProblem::new(
                path,
                format!("unsupported schema keyword {keyword:?}"),
            ));
        }
    }
    Ok(())
}

fn validate_type(type_name: &str, path: &str, root: bool) -> Result<(), SchemaProblem> {
    if root && type_name != "object" {
        return Err(SchemaProblem::new(
            path,
            "tool parameters must have type object",
        ));
    }
    if !matches!(
        type_name,
        "object" | "array" | "string" | "integer" | "number" | "boolean" | "null"
    ) {
        return Err(SchemaProblem::new(
            path,
            format!("unsupported schema type {type_name:?}"),
        ));
    }
    Ok(())
}
