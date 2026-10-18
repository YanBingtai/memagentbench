use std::{env, time::Duration};

mod format;
mod provider;

use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{agent_loop::CancellationToken, protocol::ToolDefinition};
use pi_agent_runtime::RiskClass;

use super::{
    errors::ToolError, limits::*, registry::Tool, ToolConcurrency, ToolExecutionContext,
    ToolFuture, ToolReplayPolicy, ToolResult, ToolResultFuture,
};

#[derive(Clone)]
pub struct WebSearchTool {
    client: Client,
    endpoint: String,
    api_key: String,
    timeout: Duration,
    max_output_bytes: usize,
}

impl std::fmt::Debug for WebSearchTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WebSearchTool")
            .field("endpoint", &self.endpoint)
            .field("timeout", &self.timeout)
            .field("max_output_bytes", &self.max_output_bytes)
            .finish_non_exhaustive()
    }
}

impl WebSearchTool {
    pub fn new(api_key: impl Into<String>, timeout: Duration, max_output_bytes: usize) -> Self {
        Self {
            client: Client::new(),
            endpoint: BRAVE_SEARCH_ENDPOINT.to_string(),
            api_key: api_key.into(),
            timeout: if timeout.is_zero() {
                Duration::from_secs(15)
            } else {
                timeout
            },
            max_output_bytes: if max_output_bytes == 0 {
                DEFAULT_TOOL_OUTPUT_BYTES
            } else {
                max_output_bytes
            },
        }
    }

    /// Build the default Brave-backed tool from the process environment.
    pub fn from_env(max_output_bytes: usize) -> Option<Self> {
        let api_key = env::var("BRAVE_SEARCH_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())?;
        let timeout = env::var("BRAVE_SEARCH_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|seconds| *seconds > 0)
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(15));
        Some(Self::new(api_key, timeout, max_output_bytes))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    async fn search(
        &self,
        arguments: Value,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        let arguments: WebSearchArguments =
            serde_json::from_value(arguments).map_err(|source| ToolError::InvalidArguments {
                name: "web_search".to_string(),
                source,
            })?;
        validate_web_search_arguments(&arguments)?;

        if context.is_cancelled() {
            return Err(ToolError::WebSearch {
                message: "search cancelled by the caller".to_string(),
            });
        }
        if self.api_key.trim().is_empty() {
            return Err(ToolError::WebSearch {
                message: "BRAVE_SEARCH_API_KEY is not configured".to_string(),
            });
        }

        let query = arguments.query.trim().to_string();
        let count = arguments.count.unwrap_or(DEFAULT_WEB_SEARCH_RESULTS);
        let cancellation = context.cancellation_token();
        let response = provider::request(provider::SearchRequest {
            client: &self.client,
            endpoint: &self.endpoint,
            api_key: &self.api_key,
            timeout: self.timeout,
            query: &query,
            count,
            country: arguments.country.as_deref(),
            search_lang: arguments.search_lang.as_deref(),
            freshness: arguments.freshness.as_deref(),
            cancellation: &cancellation,
        })
        .await?;
        Ok(format::results(
            response,
            &query,
            count,
            self.max_output_bytes,
        ))
    }
}

impl Tool for WebSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "web_search",
            "Search the public web and return bounded titles, URLs, and snippets.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_WEB_QUERY_CHARS,
                        "description": "Search query; maximum 600 characters"
                    },
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_WEB_SEARCH_RESULTS,
                        "description": "Number of results to return"
                    },
                    "country": {
                        "type": "string",
                        "minLength": 2,
                        "maxLength": 2,
                        "description": "Optional two-letter country code"
                    },
                    "search_lang": {
                        "type": "string",
                        "minLength": 2,
                        "maxLength": 16,
                        "description": "Optional search language code"
                    },
                    "freshness": {
                        "type": "string",
                        "enum": ["pd", "pw", "pm", "py"],
                        "description": "Optional freshness filter: day, week, month, or year"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let arguments: WebSearchArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "web_search".to_string(),
                    source,
                }
            })?;
        validate_web_search_arguments(&arguments)
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        let context = ToolExecutionContext::new(CancellationToken::new())
            .with_output_limit(self.max_output_bytes);
        Box::pin(async move { Ok(self.search(arguments, context).await?.content) })
    }

    fn execute_result<'a>(&'a self, arguments: Value) -> ToolResultFuture<'a> {
        let context = ToolExecutionContext::new(CancellationToken::new())
            .with_output_limit(self.max_output_bytes);
        Box::pin(self.search(arguments, context))
    }

    fn execute_result_with_context<'a>(
        &'a self,
        arguments: Value,
        context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(self.search(arguments, context))
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Safe
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

/// Arguments accepted by the web search tool.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WebSearchArguments {
    query: String,
    count: Option<usize>,
    country: Option<String>,
    search_lang: Option<String>,
    freshness: Option<String>,
}

fn validate_web_search_arguments(arguments: &WebSearchArguments) -> Result<(), ToolError> {
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
