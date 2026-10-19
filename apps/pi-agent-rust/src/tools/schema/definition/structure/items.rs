use serde_json::{Map, Value};

use super::super::validate_schema_node;
use crate::tools::schema::SchemaProblem;

pub(in crate::tools::schema::definition) fn validate_items(
    object: &Map<String, Value>,
    type_name: &str,
    path: &str,
    depth: usize,
) -> Result<(), SchemaProblem> {
    let Some(items) = object.get("items") else {
        return Ok(());
    };
    if type_name != "array" {
        return Err(SchemaProblem::new(
            path,
            "items is only supported for array schemas",
        ));
    }
    validate_schema_node(
        items,
        &format!("{path}.items"),
        false,
        depth.saturating_add(1),
    )
}
