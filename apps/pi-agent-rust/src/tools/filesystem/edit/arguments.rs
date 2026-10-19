//! JSON arguments and validation for `edit_file`.

use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

use crate::tools::errors::ToolError;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EditFileArguments {
    pub(super) path: PathBuf,
    pub(super) old_text: String,
    pub(super) new_text: String,
}

pub(super) fn from_value(arguments: Value) -> Result<EditFileArguments, ToolError> {
    serde_json::from_value(arguments).map_err(|source| ToolError::InvalidArguments {
        name: "edit_file".to_string(),
        source,
    })
}

pub(super) fn validate(arguments: &Value) -> Result<(), ToolError> {
    let parsed = from_value(arguments.clone())?;
    if parsed.old_text.is_empty() {
        return Err(ToolError::EditTargetEmpty { path: parsed.path });
    }
    Ok(())
}
