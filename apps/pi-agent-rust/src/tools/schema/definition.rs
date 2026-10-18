//! Registration-time schema definition validation.

use serde_json::Value;

use crate::protocol::ToolDefinition;

use super::*;

/// Validate the intentionally small schema subset understood by the agent.
///
/// The provider still receives the original JSON Schema. Locally we enforce
/// the object/primitive fields used by built-in tools, together with required
/// fields, additionalProperties, enum, string length bounds, numeric bounds,
/// and array items.
/// Unsupported keywords are rejected at registration instead of being
/// silently ignored.
pub(crate) fn validate_schema_definition(definition: &ToolDefinition) -> Result<(), String> {
    validate_schema_node(&definition.function.parameters, "$.parameters", true, 0)
        .map_err(|problem| format!("{}: {}", problem.path, problem.reason))
}

fn validate_schema_node(
    schema: &Value,
    path: &str,
    root: bool,
    depth: usize,
) -> Result<(), SchemaProblem> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(SchemaProblem {
            path: path.to_string(),
            reason: format!("schema depth exceeds the limit of {MAX_SCHEMA_DEPTH}"),
        });
    }
    let object = schema.as_object().ok_or_else(|| SchemaProblem {
        path: path.to_string(),
        reason: "schema must be a JSON object".to_string(),
    })?;
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
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: format!("unsupported schema keyword {keyword:?}"),
            });
        }
    }

    let type_name = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| SchemaProblem {
            path: path.to_string(),
            reason: "schema must declare a type".to_string(),
        })?;
    if root && type_name != "object" {
        return Err(SchemaProblem {
            path: path.to_string(),
            reason: "tool parameters must have type object".to_string(),
        });
    }
    if !matches!(
        type_name,
        "object" | "array" | "string" | "integer" | "number" | "boolean" | "null"
    ) {
        return Err(SchemaProblem {
            path: path.to_string(),
            reason: format!("unsupported schema type {type_name:?}"),
        });
    }

    if let Some(description) = object.get("description") {
        if !description.is_string() {
            return Err(SchemaProblem {
                path: format!("{path}.description"),
                reason: "description must be a string".to_string(),
            });
        }
    }

    if let Some(enum_values) = object.get("enum") {
        let enum_values = enum_values.as_array().ok_or_else(|| SchemaProblem {
            path: format!("{path}.enum"),
            reason: "enum must be a non-empty array".to_string(),
        })?;
        if enum_values.is_empty() {
            return Err(SchemaProblem {
                path: format!("{path}.enum"),
                reason: "enum must be a non-empty array".to_string(),
            });
        }
        for (index, value) in enum_values.iter().enumerate() {
            if !schema_type_matches_value(type_name, value) {
                return Err(SchemaProblem {
                    path: format!("{path}.enum[{index}]"),
                    reason: format!(
                        "enum value has type {}, expected {type_name}",
                        json_value_type(value)
                    ),
                });
            }
            if enum_values[..index].contains(value) {
                return Err(SchemaProblem {
                    path: format!("{path}.enum[{index}]"),
                    reason: "enum values must be unique".to_string(),
                });
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
                    return Err(SchemaProblem {
                        path: format!("{path}.enum[{index}]"),
                        reason: "enum value violates the string length bounds".to_string(),
                    });
                }
            }
        }
    }

    let properties = match object.get("properties") {
        Some(value) => Some(value.as_object().ok_or_else(|| SchemaProblem {
            path: format!("{path}.properties"),
            reason: "properties must be an object".to_string(),
        })?),
        None => None,
    };
    if properties.is_some() && type_name != "object" {
        return Err(SchemaProblem {
            path: path.to_string(),
            reason: "properties is only supported for object schemas".to_string(),
        });
    }
    if let Some(properties) = properties {
        for (name, property_schema) in properties {
            validate_schema_node(
                property_schema,
                &format!("{path}.properties.{name}"),
                false,
                depth.saturating_add(1),
            )?;
        }
    }

    if let Some(required) = object.get("required") {
        if type_name != "object" {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: "required is only supported for object schemas".to_string(),
            });
        }
        let required = required.as_array().ok_or_else(|| SchemaProblem {
            path: format!("{path}.required"),
            reason: "required must be an array of strings".to_string(),
        })?;
        let mut seen = BTreeSet::new();
        for item in required {
            let name = item.as_str().ok_or_else(|| SchemaProblem {
                path: format!("{path}.required"),
                reason: "required must contain only strings".to_string(),
            })?;
            if !seen.insert(name) {
                return Err(SchemaProblem {
                    path: format!("{path}.required"),
                    reason: format!("duplicate required property {name:?}"),
                });
            }
            if !properties.is_some_and(|properties| properties.contains_key(name)) {
                return Err(SchemaProblem {
                    path: format!("{path}.required"),
                    reason: format!("required property {name:?} is not declared"),
                });
            }
        }
    }

    if let Some(additional) = object.get("additionalProperties") {
        if type_name != "object" {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: "additionalProperties is only supported for object schemas".to_string(),
            });
        }
        if !additional.is_boolean() {
            return Err(SchemaProblem {
                path: format!("{path}.additionalProperties"),
                reason: "additionalProperties must be a boolean".to_string(),
            });
        }
    }

    if let Some(min_length) = object.get("minLength") {
        if type_name != "string" || min_length.as_u64().is_none() {
            return Err(SchemaProblem {
                path: format!("{path}.minLength"),
                reason: "minLength must be a non-negative integer on a string schema".to_string(),
            });
        }
    }
    if let Some(max_length) = object.get("maxLength") {
        if type_name != "string" || max_length.as_u64().is_none() {
            return Err(SchemaProblem {
                path: format!("{path}.maxLength"),
                reason: "maxLength must be a non-negative integer on a string schema".to_string(),
            });
        }
    }
    if let (Some(min_length), Some(max_length)) = (
        object.get("minLength").and_then(Value::as_u64),
        object.get("maxLength").and_then(Value::as_u64),
    ) {
        if min_length > max_length {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: "minLength must not exceed maxLength".to_string(),
            });
        }
    }

    for keyword in ["minimum", "maximum"] {
        if let Some(value) = object.get(keyword) {
            if !matches!(type_name, "integer" | "number") || !value.is_number() {
                return Err(SchemaProblem {
                    path: format!("{path}.{keyword}"),
                    reason: format!("{keyword} must be a number on a numeric schema"),
                });
            }
            if type_name == "integer" && value.as_i64().is_none() && value.as_u64().is_none() {
                return Err(SchemaProblem {
                    path: format!("{path}.{keyword}"),
                    reason: format!("{keyword} must be an integer on an integer schema"),
                });
            }
        }
    }
    if let (Some(minimum), Some(maximum)) = (object.get("minimum"), object.get("maximum")) {
        if minimum.as_f64() > maximum.as_f64() {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: "minimum must not exceed maximum".to_string(),
            });
        }
    }

    if let Some(items) = object.get("items") {
        if type_name != "array" {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: "items is only supported for array schemas".to_string(),
            });
        }
        validate_schema_node(
            items,
            &format!("{path}.items"),
            false,
            depth.saturating_add(1),
        )?;
    }

    Ok(())
}
