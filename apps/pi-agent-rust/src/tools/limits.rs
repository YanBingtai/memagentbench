//! Shared size and traversal limits for built-in tools.

pub(crate) const DEFAULT_TOOL_UPDATE_BYTES: usize = 65_536;
pub(crate) const DEFAULT_LIST_MAX_DEPTH: usize = 2;
pub(crate) const DEFAULT_LIST_MAX_ENTRIES: usize = 200;
pub(crate) const DEFAULT_TOOL_OUTPUT_BYTES: usize = 65_536;
pub(crate) const MAX_LIST_DEPTH: usize = 32;
pub(crate) const MAX_LIST_ENTRIES: usize = 1_000;
pub(crate) const DEFAULT_SEARCH_MAX_RESULTS: usize = 100;
pub(crate) const MAX_SEARCH_RESULTS: usize = 1_000;
pub(crate) const MAX_SEARCH_FILES: usize = 10_000;
pub(crate) const MAX_SEARCH_LINE_BYTES: usize = 1_024;
pub(crate) const BRAVE_SEARCH_ENDPOINT: &str = "https://api.search.brave.com/res/v1/web/search";
pub(crate) const MAX_WEB_QUERY_CHARS: usize = 600;
pub(crate) const DEFAULT_WEB_SEARCH_RESULTS: usize = 5;
pub(crate) const MAX_WEB_SEARCH_RESULTS: usize = 20;
pub(crate) const MAX_WEB_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
pub(crate) const DEFAULT_SHELL_TIMEOUT_SECS: u64 = 120;
pub(crate) const MAX_SHELL_TIMEOUT_SECS: u64 = 600;
pub(crate) const MAX_SHELL_COMMAND_BYTES: usize = 64 * 1024;
pub(crate) const MAX_TOOL_ARGUMENT_BYTES: usize = 1_048_576;
pub(crate) const MAX_TOOL_ARGUMENT_DEPTH: usize = 64;
pub(crate) const MAX_SCHEMA_DEPTH: usize = 32;
pub(crate) fn truncate_output(text: String, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text;
    }
    if max_bytes == 0 {
        return String::new();
    }

    const MARKER: &str = "…";
    let content_limit = max_bytes.saturating_sub(MARKER.len());
    let mut end = content_limit.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }

    format!("{}{}", &text[..end], MARKER)
}
