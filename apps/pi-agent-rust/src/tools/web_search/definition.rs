//! Stable model definition and tool lifecycle for web search.

use serde_json::{json, Value};

use crate::{agent_loop::CancellationToken, protocol::ToolDefinition};
use pi_agent_runtime::RiskClass;

use super::{arguments, WebSearchTool};
use crate::tools::{
    limits::*, registry::Tool, ToolConcurrency, ToolExecutionContext, ToolFuture, ToolReplayPolicy,
    ToolResultFuture,
};

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

    fn validate_arguments(&self, arguments: &Value) -> Result<(), crate::tools::errors::ToolError> {
        arguments::parse(arguments.clone()).map(|_| ())
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
