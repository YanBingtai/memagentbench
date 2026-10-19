//! Per-file scan and result-limit handling for directory traversal.

use std::path::Path;

use crate::tools::limits::MAX_SEARCH_FILES;

use super::super::super::matcher::{search_file, FileSearchOutcome};
use super::{SearchOptions, SearchState};

/// Inspect one regular file and report whether traversal should stop.
pub(super) async fn inspect_file(
    path: &Path,
    relative: &Path,
    options: &SearchOptions<'_>,
    state: &mut SearchState<'_>,
) -> bool {
    if !reserve_file(state) {
        return true;
    }

    let remaining_results = options
        .max_results
        .saturating_sub(state.report.matches.len());
    match search_file(
        path,
        relative,
        options.prefix,
        options.query,
        remaining_results,
        options.max_file_bytes,
    )
    .await
    {
        FileSearchOutcome::Skipped => state.report.skipped_files += 1,
        FileSearchOutcome::Matches(matches) => state.report.matches.extend(matches),
        FileSearchOutcome::ReachedResultLimit(matches) => {
            state.report.matches.extend(matches);
            state.report.truncated = true;
            return true;
        }
    }
    false
}

fn reserve_file(state: &mut SearchState<'_>) -> bool {
    if *state.scanned_files == MAX_SEARCH_FILES {
        state.report.truncated = true;
        state.report.file_limit_reached = true;
        return false;
    }
    *state.scanned_files += 1;
    true
}
