//! JSON arguments and validation for `read_file`.

use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

use crate::tools::errors::ToolError;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReadFileArguments {
    pub(super) path: PathBuf,
    pub(super) offset: Option<usize>,
    pub(super) limit: Option<usize>,
}

pub(super) fn from_value(arguments: Value) -> Result<ReadFileArguments, ToolError> {
    serde_json::from_value(arguments).map_err(|source| ToolError::InvalidArguments {
        name: "read_file".to_string(),
        source,
    })
}

pub(super) fn validate(arguments: &Value) -> Result<(), ToolError> {
    let parsed = from_value(arguments.clone())?;
    if parsed.offset == Some(0) {
        return Err(ToolError::InvalidRange {
            tool: "read_file".to_string(),
            reason: "offset must be greater than zero".to_string(),
        });
    }
    if parsed.limit == Some(0) {
        return Err(ToolError::InvalidRange {
            tool: "read_file".to_string(),
            reason: "limit must be greater than zero".to_string(),
        });
    }
    Ok(())
}
