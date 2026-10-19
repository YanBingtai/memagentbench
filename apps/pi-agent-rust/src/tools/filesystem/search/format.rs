//! Bounded output formatting for `search_files`.

use crate::tools::limits::{truncate_output, MAX_SEARCH_FILES};

use super::walk::SearchReport;

pub(super) fn format_report(
    report: SearchReport,
    max_results: usize,
    max_output_bytes: usize,
) -> String {
    let SearchReport {
        matches,
        skipped_files,
        truncated,
        file_limit_reached,
    } = report;
    let mut output = if matches.is_empty() {
        "(no matches)".to_string()
    } else {
        matches.join("\n")
    };
    if skipped_files > 0 {
        output.push_str(&format!(
            "\n[skipped {skipped_files} unreadable, binary, or oversized files]"
        ));
    }
    if truncated {
        if file_limit_reached {
            output.push_str(&format!(
                "\n[search truncated after scanning {MAX_SEARCH_FILES} files]"
            ));
        } else {
            output.push_str(&format!(
                "\n[search truncated after {max_results} matching lines]"
            ));
        }
    }
    truncate_output(output, max_output_bytes)
}
