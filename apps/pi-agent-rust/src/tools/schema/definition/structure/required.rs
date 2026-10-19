use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::tools::schema::SchemaProblem;

pub(in crate::tools::schema::definition) fn validate_required(
    object: &Map<String, Value>,
    properties: Option<&Map<String, Value>>,
    type_name: &str,
    path: &str,
) -> Result<(), SchemaProblem> {
    let Some(required) = object.get("required") else {
        return Ok(());
    };
    if type_name != "object" {
        return Err(SchemaProblem::new(
            path,
            "required is only supported for object schemas",
        ));
    }
    let required = required.as_array().ok_or_else(|| {
        SchemaProblem::new(
            format!("{path}.required"),
            "required must be an array of strings",
        )
    })?;
    let mut seen = BTreeSet::new();
    for item in required {
        let name = item.as_str().ok_or_else(|| {
            SchemaProblem::new(
                format!("{path}.required"),
                "required must contain only strings",
            )
        })?;
        if !seen.insert(name) {
            return Err(SchemaProblem::new(
                format!("{path}.required"),
                format!("duplicate required property {name:?}"),
            ));
        }
        if !properties.is_some_and(|properties| properties.contains_key(name)) {
            return Err(SchemaProblem::new(
                format!("{path}.required"),
                format!("required property {name:?} is not declared"),
            ));
        }
    }
    Ok(())
}
