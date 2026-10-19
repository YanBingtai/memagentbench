use serde_json::{Map, Value};

use crate::tools::schema::SchemaProblem;

pub(in crate::tools::schema::definition) fn validate_additional_properties(
    object: &Map<String, Value>,
    type_name: &str,
    path: &str,
) -> Result<(), SchemaProblem> {
    let Some(additional) = object.get("additionalProperties") else {
        return Ok(());
    };
    if type_name != "object" {
        return Err(SchemaProblem::new(
            path,
            "additionalProperties is only supported for object schemas",
        ));
    }
    if !additional.is_boolean() {
        return Err(SchemaProblem::new(
            format!("{path}.additionalProperties"),
            "additionalProperties must be a boolean",
        ));
    }
    Ok(())
}
