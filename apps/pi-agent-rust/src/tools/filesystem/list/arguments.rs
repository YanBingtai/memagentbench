//! JSON arguments and validation for `list_files`.

use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

use crate::tools::{
    errors::ToolError,
    limits::{MAX_LIST_DEPTH, MAX_LIST_ENTRIES},
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ListFilesArguments {
    pub(super) path: Option<PathBuf>,
    pub(super) max_depth: Option<usize>,
    pub(super) max_entries: Option<usize>,
}

pub(super) fn from_value(arguments: Value) -> Result<ListFilesArguments, ToolError> {
    serde_json::from_value(arguments).map_err(|source| ToolError::InvalidArguments {
        name: "list_files".to_string(),
        source,
    })
}

pub(super) fn validate(arguments: &Value) -> Result<(), ToolError> {
    let parsed = from_value(arguments.clone())?;
    validate_limits(parsed.max_depth, parsed.max_entries)
}

pub(super) fn validate_limits(
    max_depth: Option<usize>,
    max_entries: Option<usize>,
) -> Result<(), ToolError> {
    if max_depth.is_some_and(|depth| depth > MAX_LIST_DEPTH) {
        return Err(ToolError::InvalidArgumentValue {
            name: "list_files".to_string(),
            reason: format!("max_depth must be at most {MAX_LIST_DEPTH}"),
        });
    }
    if max_entries.is_some_and(|entries| entries == 0 || entries > MAX_LIST_ENTRIES) {
        return Err(ToolError::InvalidArgumentValue {
            name: "list_files".to_string(),
            reason: format!("max_entries must be between 1 and {MAX_LIST_ENTRIES}"),
        });
    }
    Ok(())
}
