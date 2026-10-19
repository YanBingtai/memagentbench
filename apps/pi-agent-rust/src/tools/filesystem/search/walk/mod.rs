//! Deterministic, workspace-bounded directory traversal for `search_files`.

mod entries;
mod report;

use std::path::{Path, PathBuf};

use crate::tools::errors::ToolError;

use self::entries::{collect_entries, read_children, SearchOptions, SearchState};

pub(super) use self::report::SearchReport;

pub(super) async fn search_directory(
    workspace: &Path,
    root: &Path,
    prefix: &Path,
    query: &str,
    max_results: usize,
    max_file_bytes: usize,
) -> Result<SearchReport, ToolError> {
    let mut pending = vec![(root.to_path_buf(), PathBuf::new())];
    let mut report = SearchReport::default();
    let mut scanned_files = 0usize;
    let options = SearchOptions {
        workspace,
        prefix,
        query,
        max_results,
        max_file_bytes,
    };

    while let Some((directory, relative_directory)) = pending.pop() {
        let children = read_children(&directory).await?;
        let mut state = SearchState {
            scanned_files: &mut scanned_files,
            report: &mut report,
        };
        let child_directories =
            collect_entries(&options, &relative_directory, children, &mut state).await?;

        if report.truncated {
            return Ok(report);
        }
        pending.extend(child_directories.into_iter().rev());
    }

    Ok(report)
}
