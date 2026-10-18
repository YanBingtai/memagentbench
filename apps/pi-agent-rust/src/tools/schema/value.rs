//! Execution-time validation of parsed tool arguments.

use serde_json::Value;

use crate::protocol::ToolDefinition;

use super::*;

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
        return Err(SchemaProblem {
            path: path.to_string(),
            reason: format!("schema depth exceeds the limit of {MAX_SCHEMA_DEPTH}"),
        });
    }
    let object = schema.as_object().ok_or_else(|| SchemaProblem {
        path: path.to_string(),
        reason: "schema must be a JSON object".to_string(),
    })?;
    let type_name = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| SchemaProblem {
            path: path.to_string(),
            reason: "schema must declare a type".to_string(),
        })?;
    let type_matches = match type_name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    };
    if !type_matches {
        return Err(SchemaProblem {
            path: path.to_string(),
            reason: format!("expected {type_name}, got {}", json_value_type(value)),
        });
    }
    if let Some(enum_values) = object.get("enum").and_then(Value::as_array) {
        if !enum_values.iter().any(|candidate| candidate == value) {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: "value is not one of the allowed enum values".to_string(),
            });
        }
    }

    match type_name {
        "object" => {
            let values = value.as_object().expect("object type was checked above");
            let properties = object
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            if let Some(required) = object.get("required").and_then(Value::as_array) {
                for required in required.iter().filter_map(Value::as_str) {
                    if !values.contains_key(required) {
                        return Err(SchemaProblem {
                            path: format!("{path}.{required}"),
                            reason: "required property is missing".to_string(),
                        });
                    }
                }
            }
            if object.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
                if let Some(unknown) = values.keys().find(|name| !properties.contains_key(*name)) {
                    return Err(SchemaProblem {
                        path: format!("{path}.{unknown}"),
                        reason: "additional property is not allowed".to_string(),
                    });
                }
            }
            for (name, property_schema) in properties {
                if let Some(property_value) = values.get(&name) {
                    validate_schema_value(
                        &property_schema,
                        property_value,
                        &format!("{path}.{name}"),
                        depth.saturating_add(1),
                    )?;
                }
            }
        }
        "array" => {
            if let Some(item_schema) = object.get("items") {
                for (index, item) in value
                    .as_array()
                    .expect("array type was checked above")
                    .iter()
                    .enumerate()
                {
                    validate_schema_value(
                        item_schema,
                        item,
                        &format!("{path}[{index}]"),
                        depth.saturating_add(1),
                    )?;
                }
            }
        }
        "string" => {
            let length = value
                .as_str()
                .expect("string type was checked above")
                .chars()
                .count() as u64;
            if let Some(min_length) = object.get("minLength").and_then(Value::as_u64) {
                if length < min_length {
                    return Err(SchemaProblem {
                        path: path.to_string(),
                        reason: format!("string length must be at least {min_length}"),
                    });
                }
            }
            if let Some(max_length) = object.get("maxLength").and_then(Value::as_u64) {
                if length > max_length {
                    return Err(SchemaProblem {
                        path: path.to_string(),
                        reason: format!("string length must be at most {max_length}"),
                    });
                }
            }
        }
        "integer" => {
            let integer = integer_value(value).expect("integer type was checked above");
            if let Some(minimum) = object.get("minimum").and_then(integer_value) {
                if compare_integers(integer, minimum) == CompareOrdering::Less {
                    return Err(SchemaProblem {
                        path: path.to_string(),
                        reason: format!("integer must be at least {minimum:?}"),
                    });
                }
            }
            if let Some(maximum) = object.get("maximum").and_then(integer_value) {
                if compare_integers(integer, maximum) == CompareOrdering::Greater {
                    return Err(SchemaProblem {
                        path: path.to_string(),
                        reason: format!("integer must be at most {maximum:?}"),
                    });
                }
            }
        }
        "number" => {
            let number = value.as_f64().expect("numeric type was checked above");
            if let Some(minimum) = object.get("minimum").and_then(Value::as_f64) {
                if number < minimum {
                    return Err(SchemaProblem {
                        path: path.to_string(),
                        reason: format!("number must be at least {minimum}"),
                    });
                }
            }
            if let Some(maximum) = object.get("maximum").and_then(Value::as_f64) {
                if number > maximum {
                    return Err(SchemaProblem {
                        path: path.to_string(),
                        reason: format!("number must be at most {maximum}"),
                    });
                }
            }
        }
        "boolean" | "null" => {}
        _ => unreachable!("schema definitions reject unsupported types"),
    }

    Ok(())
}
