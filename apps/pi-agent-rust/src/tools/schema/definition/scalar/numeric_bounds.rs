use serde_json::{Map, Value};

use crate::tools::schema::SchemaProblem;

pub(in crate::tools::schema::definition) fn validate_numeric_bounds(
    object: &Map<String, Value>,
    type_name: &str,
    path: &str,
) -> Result<(), SchemaProblem> {
    for keyword in ["minimum", "maximum"] {
        let Some(value) = object.get(keyword) else {
            continue;
        };
        if !matches!(type_name, "integer" | "number") || !value.is_number() {
            return Err(SchemaProblem::new(
                format!("{path}.{keyword}"),
                format!("{keyword} must be a number on a numeric schema"),
            ));
        }
        if type_name == "integer" && value.as_i64().is_none() && value.as_u64().is_none() {
            return Err(SchemaProblem::new(
                format!("{path}.{keyword}"),
                format!("{keyword} must be an integer on an integer schema"),
            ));
        }
    }
    if let (Some(minimum), Some(maximum)) = (object.get("minimum"), object.get("maximum")) {
        if minimum.as_f64() > maximum.as_f64() {
            return Err(SchemaProblem::new(path, "minimum must not exceed maximum"));
        }
    }
    Ok(())
}
