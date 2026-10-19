//! Child-agent progress and result metadata helpers.

mod details;
mod status;
mod text;
mod updates;

pub(crate) use details::child_details;
#[cfg(test)]
pub(crate) use status::child_error_status;
pub(crate) use status::child_error_status_with_outcome;
pub(crate) use text::truncate_output;
pub(crate) use updates::subagent_progress;
