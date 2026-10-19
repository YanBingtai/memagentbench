//! Request arguments and validation for the web search tool.

use serde::Deserialize;
use serde_json::Value;

use super::super::{errors::ToolError, limits::*};

/// Arguments accepted by the web search tool.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WebSearchArguments {
    pub(super) query: String,
    pub(super) count: Option<usize>,
    pub(super) country: Option<String>,
    pub(super) search_lang: Option<String>,
    pub(super) freshness: Option<String>,
}

pub(super) fn parse(arguments: Value) -> Result<WebSearchArguments, ToolError> {
    let arguments: WebSearchArguments =
        serde_json::from_value(arguments).map_err(|source| ToolError::InvalidArguments {
            name: "web_search".to_string(),
            source,
        })?;
    validate(&arguments)?;
    Ok(arguments)
}

pub(super) fn validate(arguments: &WebSearchArguments) -> Result<(), ToolError> {
    let query = arguments.query.trim();
    if query.is_empty() {
        return Err(ToolError::InvalidArgumentValue {
            name: "web_search".to_string(),
            reason: "query must not be empty".to_string(),
        });
    }
    if query.chars().count() > MAX_WEB_QUERY_CHARS {
        return Err(ToolError::InvalidArgumentValue {
            name: "web_search".to_string(),
            reason: format!("query must not exceed {MAX_WEB_QUERY_CHARS} characters"),
        });
    }
    if arguments
        .count
        .is_some_and(|count| !(1..=MAX_WEB_SEARCH_RESULTS).contains(&count))
    {
        return Err(ToolError::InvalidArgumentValue {
            name: "web_search".to_string(),
            reason: format!("count must be between 1 and {MAX_WEB_SEARCH_RESULTS}"),
        });
    }
    if arguments.country.as_deref().is_some_and(|country| {
        country.len() != 2
            || !country
                .chars()
                .all(|character| character.is_ascii_alphabetic())
    }) {
        return Err(ToolError::InvalidArgumentValue {
            name: "web_search".to_string(),
            reason: "country must be a two-letter code".to_string(),
        });
    }
    if arguments.search_lang.as_deref().is_some_and(|language| {
        !(2..=16).contains(&language.len())
            || !language
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '-')
    }) {
        return Err(ToolError::InvalidArgumentValue {
            name: "web_search".to_string(),
            reason: "search_lang must contain 2-16 letters, numbers, or '-'".to_string(),
        });
    }
    if arguments
        .freshness
        .as_deref()
        .is_some_and(|freshness| !matches!(freshness, "pd" | "pw" | "pm" | "py"))
    {
        return Err(ToolError::InvalidArgumentValue {
            name: "web_search".to_string(),
            reason: "freshness must be one of pd, pw, pm, or py".to_string(),
        });
    }
    Ok(())
}
