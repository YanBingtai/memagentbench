//! Bounded text-file matching for `search_files`.

use std::path::{Path, PathBuf};

use crate::tools::limits::{truncate_output, MAX_SEARCH_LINE_BYTES};

/// Result of inspecting one regular file.
pub(super) enum FileSearchOutcome {
    Skipped,
    Matches(Vec<String>),
    ReachedResultLimit(Vec<String>),
}

pub(super) async fn search_file(
    path: &Path,
    relative: &Path,
    prefix: &Path,
    query: &str,
    max_results: usize,
    max_file_bytes: usize,
) -> FileSearchOutcome {
    let metadata = match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata,
        Err(_) => return FileSearchOutcome::Skipped,
    };
    if metadata.len() > max_file_bytes as u64 {
        return FileSearchOutcome::Skipped;
    }

    let text = match tokio::fs::read_to_string(path).await {
        Ok(text) => text,
        Err(_) => return FileSearchOutcome::Skipped,
    };

    let display_path = display_path(prefix, relative);
    let mut matches = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if !line.contains(query) {
            continue;
        }
        matches.push(format_match(&display_path, line_index + 1, line));
        if matches.len() == max_results {
            return FileSearchOutcome::ReachedResultLimit(matches);
        }
    }

    FileSearchOutcome::Matches(matches)
}

fn display_path(prefix: &Path, relative: &Path) -> PathBuf {
    if prefix.as_os_str().is_empty() {
        relative.to_path_buf()
    } else {
        prefix.join(relative)
    }
}

fn format_match(path: &Path, line_number: usize, line: &str) -> String {
    format!(
        "{}:{}:{}",
        path.display(),
        line_number,
        truncate_output(line.to_string(), MAX_SEARCH_LINE_BYTES)
    )
}
