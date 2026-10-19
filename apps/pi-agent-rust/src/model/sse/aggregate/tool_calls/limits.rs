use crate::model::error::ModelError;

/// Append a fragment while charging its bytes against the stream limit.
pub(super) fn append_bounded(
    target: &mut String,
    value: &str,
    total: &mut usize,
    limit: usize,
    label: &'static str,
) -> Result<(), ModelError> {
    *total = total
        .checked_add(value.len())
        .ok_or(ModelError::StreamLimit { limit: label })?;
    if *total > limit {
        return Err(ModelError::StreamLimit { limit: label });
    }
    target.push_str(value);
    Ok(())
}
