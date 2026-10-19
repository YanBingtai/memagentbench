//! Brave request execution and bounded result formatting.

use serde_json::Value;

use super::{arguments, format, provider, WebSearchTool};
use crate::tools::{
    errors::ToolError, limits::DEFAULT_WEB_SEARCH_RESULTS, ToolExecutionContext, ToolResult,
};

impl WebSearchTool {
    pub(super) async fn search(
        &self,
        arguments: Value,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        let arguments = arguments::parse(arguments)?;

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
