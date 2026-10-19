//! Bounded JSONL framing and crash-tail repair.

mod read;
mod repair;

pub(crate) use read::{read_record_line, read_records};
pub(crate) use repair::{truncate_incomplete_tail, unix_timestamp_ms};
