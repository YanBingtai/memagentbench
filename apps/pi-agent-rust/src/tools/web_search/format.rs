//! Bounded, provider-neutral formatting for web search results.

use serde_json::json;

use super::super::{limits::truncate_output, ToolResult};
use super::provider::BraveSearchResponse;

pub(super) fn results(
    response: BraveSearchResponse,
    query: &str,
    count: usize,
    max_output_bytes: usize,
) -> ToolResult {
    let results = response
        .web
        .map(|web| web.results)
        .unwrap_or_default()
        .into_iter()
        .take(count)
        .collect::<Vec<_>>();
    let mut content = String::new();
    let mut details = Vec::with_capacity(results.len());
    for (index, result) in results.iter().enumerate() {
        if index > 0 {
            content.push('\n');
        }
        content.push_str(&format!("[{}] {}\n{}", index + 1, result.title, result.url));
        if let Some(description) = result
            .description
            .as_deref()
            .filter(|value| !value.is_empty())
        {
            content.push('\n');
            content.push_str(description);
        }
        details.push(json!({
        "title": result.title,
        "url": result.url,
        "description": result.description,
        }));
    }
    if content.is_empty() {
        content.push_str("(no results)");
    }
    ToolResult::text(truncate_output(content, max_output_bytes))
        .with_details(json!({
            "provider": "brave",
            "query": query,
            "result_count": details.len(),
            "results": details,
        }))
        .bounded(max_output_bytes)
}
