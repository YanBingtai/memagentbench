//! Aggregation and validation of streamed tool-call fragments.

mod apply;
mod finish;
mod limits;
mod merge;
mod types;

pub(crate) use types::StreamToolAccumulator;

pub(super) use apply::apply_delta;
pub(super) use finish::finish;

use crate::model::error::ModelError;

pub(super) fn append_bounded(
    target: &mut String,
    value: &str,
    total: &mut usize,
    limit: usize,
    label: &'static str,
) -> Result<(), ModelError> {
    limits::append_bounded(target, value, total, limit, label)
}
