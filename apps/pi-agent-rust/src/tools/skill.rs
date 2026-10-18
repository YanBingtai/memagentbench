use serde::Deserialize;
use serde_json::{json, Value};

use crate::{protocol::ToolDefinition, skills::SkillCatalog};
use pi_agent_runtime::RiskClass;

use super::{errors::ToolError, registry::Tool, ToolConcurrency, ToolFuture, ToolReplayPolicy};

#[derive(Debug, Clone)]
pub struct ReadSkillTool {
    catalog: SkillCatalog,
    max_output_bytes: usize,
}

impl ReadSkillTool {
    pub fn new(catalog: SkillCatalog, max_output_bytes: usize) -> Self {
        Self {
            catalog,
            max_output_bytes,
        }
    }
}

impl Tool for ReadSkillTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "read_skill",
            "Read the full instructions for a registered skill by its exact name.",
            json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact skill name from the available skills list",
                        "minLength": 1
                    }
                },
                "required": ["name"],
                "additionalProperties": false
            }),
        )
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Safe
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        serde_json::from_value::<ReadSkillArguments>(arguments.clone())
            .map_err(|source| ToolError::InvalidArguments {
                name: "read_skill".to_string(),
                source,
            })
            .and_then(|parsed| {
                if parsed.name.trim().is_empty() {
                    return Err(ToolError::InvalidArgumentValue {
                        name: "read_skill".to_string(),
                        reason: "name must not be empty".to_string(),
                    });
                }
                Ok(())
            })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: ReadSkillArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "read_skill".to_string(),
                        source,
                    }
                })?;
            if self.catalog.get(&arguments.name).is_none() {
                return Err(ToolError::UnknownSkill {
                    name: arguments.name,
                });
            }

            let content = self
                .catalog
                .format_invocation(&arguments.name, None)
                .map_err(|error| ToolError::SkillInvocation {
                    name: arguments.name.clone(),
                    message: error.to_string(),
                })?;
            if content.len() > self.max_output_bytes {
                return Err(ToolError::SkillTooLarge {
                    name: arguments.name,
                    size: content.len(),
                    limit: self.max_output_bytes,
                });
            }

            Ok(content)
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadSkillArguments {
    name: String,
}
