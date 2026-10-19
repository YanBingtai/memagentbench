//! Object and array traversal for execution-time schema validation.

use serde_json::{Map, Value};

use super::validate_schema_value;
use crate::tools::schema::SchemaProblem;

pub(super) fn validate_object(
    schema: &Map<String, Value>,
    value: &Value,
    path: &str,
    depth: usize,
) -> Result<(), SchemaProblem> {
    let values = value.as_object().expect("object type was checked above");
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    validate_required(schema, values, path)?;
    validate_additional_properties(schema, values, &properties, path)?;
    for (name, property_schema) in properties {
        let Some(property_value) = values.get(&name) else {
            continue;
        };
        validate_schema_value(
            &property_schema,
            property_value,
            &format!("{path}.{name}"),
            depth.saturating_add(1),
        )?;
    }
    Ok(())
}

fn validate_required(
    schema: &Map<String, Value>,
    values: &Map<String, Value>,
    path: &str,
) -> Result<(), SchemaProblem> {
    let Some(required) = schema.get("required").and_then(Value::as_array) else {
        return Ok(());
    };
    for required in required.iter().filter_map(Value::as_str) {
        if !values.contains_key(required) {
            return Err(SchemaProblem::new(
                format!("{path}.{required}"),
                "required property is missing",
            ));
        }
    }
    Ok(())
}

fn validate_additional_properties(
    schema: &Map<String, Value>,
    values: &Map<String, Value>,
    properties: &Map<String, Value>,
    path: &str,
) -> Result<(), SchemaProblem> {
    if schema.get("additionalProperties").and_then(Value::as_bool) != Some(false) {
        return Ok(());
    }
    let Some(unknown) = values.keys().find(|name| !properties.contains_key(*name)) else {
        return Ok(());
    };
    Err(SchemaProblem::new(
        format!("{path}.{unknown}"),
        "additional property is not allowed",
    ))
}

pub(super) fn validate_array(
    schema: &Map<String, Value>,
    value: &Value,
    path: &str,
    depth: usize,
) -> Result<(), SchemaProblem> {
    let Some(item_schema) = schema.get("items") else {
        return Ok(());
    };
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
    Ok(())
}
