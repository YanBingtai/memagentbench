use crate::model::error::ModelError;

use super::append_bounded;

pub(super) fn merge_metadata(
    target: &mut String,
    value: &str,
    field: &'static str,
) -> Result<(), ModelError> {
    if value.is_empty() {
        return Ok(());
    }
    if target.is_empty() {
        target.push_str(value);
    } else if target != value {
        return Err(ModelError::StreamProtocol(format!(
            "conflicting {field}: existing {:?}, received {:?}",
            target, value
        )));
    }
    Ok(())
}

pub(super) fn append_fragment(
    target: &mut String,
    value: &str,
    total: &mut usize,
    limit: usize,
    label: &'static str,
) -> Result<(), ModelError> {
    if value.is_empty() || value == target {
        return Ok(());
    }
    let fragment = value.strip_prefix(target.as_str()).unwrap_or(value);
    append_bounded(target, fragment, total, limit, label)
}
