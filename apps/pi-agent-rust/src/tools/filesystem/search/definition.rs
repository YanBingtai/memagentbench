//! Model-facing definition and argument validation for `search_files`.

use std::path::PathBuf;

use pi_agent_runtime::RiskClass;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{protocol::ToolDefinition, tools::limits::*};

use super::SearchFilesTool;
use crate::tools::{
    errors::ToolError, registry::Tool, ToolConcurrency, ToolFuture, ToolReplayPolicy,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SearchFilesArguments {
    pub(super) query: String,
    pub(super) path: Option<PathBuf>,
    pub(super) max_results: Option<usize>,
}

pub(super) fn parse(arguments: Value) -> Result<SearchFilesArguments, ToolError> {
    serde_json::from_value(arguments).map_err(|source| ToolError::InvalidArguments {
        name: "search_files".to_string(),
        source,
    })
}

fn validate(arguments: &SearchFilesArguments) -> Result<(), ToolError> {
    if arguments.query.is_empty() {
        return Err(ToolError::InvalidArgumentValue {
            name: "search_files".to_string(),
            reason: "query must not be empty".to_string(),
        });
    }
    if arguments
        .max_results
        .is_some_and(|results| results == 0 || results > MAX_SEARCH_RESULTS)
    {
        return Err(ToolError::InvalidArgumentValue {
            name: "search_files".to_string(),
            reason: format!("max_results must be between 1 and {MAX_SEARCH_RESULTS}"),
        });
    }
    Ok(())
}

impl Tool for SearchFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "search_files",
            "Search UTF-8 text files for a literal substring inside the configured workspace.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Literal, case-sensitive substring to find"
                    },
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory; defaults to the workspace root"
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_RESULTS,
                        "description": "Maximum matching lines to return"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let parsed = parse(arguments.clone())?;
        validate(&parsed)
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

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move { self.search(arguments).await })
    }
}
