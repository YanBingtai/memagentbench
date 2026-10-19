//! Directory-entry handling for the bounded `search_files` traversal.

mod collect;
mod limits;
mod paths;
mod read;

use std::path::Path;

use super::report::SearchReport;

pub(super) use collect::collect_entries;
pub(super) use read::read_children;

pub(super) struct SearchOptions<'a> {
    pub(super) workspace: &'a Path,
    pub(super) prefix: &'a Path,
    pub(super) query: &'a str,
    pub(super) max_results: usize,
    pub(super) max_file_bytes: usize,
}

pub(super) struct SearchState<'a> {
    pub(super) scanned_files: &'a mut usize,
    pub(super) report: &'a mut SearchReport,
}
