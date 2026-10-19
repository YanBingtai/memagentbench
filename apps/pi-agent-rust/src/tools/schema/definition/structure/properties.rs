use serde_json::{Map, Value};

use super::super::validate_schema_node;
use crate::tools::schema::SchemaProblem;

pub(in crate::tools::schema::definition) fn properties<'a>(
    object: &'a Map<String, Value>,
    path: &str,
) -> Result<Option<&'a Map<String, Value>>, SchemaProblem> {
    match object.get("properties") {
        Some(value) => value.as_object().map(Some).ok_or_else(|| {
            SchemaProblem::new(format!("{path}.properties"), "properties must be an object")
        }),
        None => Ok(None),
    }
}

pub(in crate::tools::schema::definition) fn validate_properties(
    properties: Option<&Map<String, Value>>,
    path: &str,
    depth: usize,
) -> Result<(), SchemaProblem> {
    let Some(properties) = properties else {
        return Ok(());
    };
    for (name, property_schema) in properties {
        validate_schema_node(
            property_schema,
            &format!("{path}.properties.{name}"),
            false,
            depth.saturating_add(1),
        )?;
    }
    Ok(())
}
