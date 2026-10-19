use std::{collections::BTreeMap, sync::Arc};

use super::super::super::{limits::DEFAULT_TOOL_OUTPUT_BYTES, schema::validate_schema_definition};
use super::super::{
    validation::validate_tool_name, RegisteredTool, RegistryError, Tool, ToolCapabilities,
    ToolGateway, ToolRegistry,
};

impl Default for ToolRegistry {
    fn default() -> Self {
        Self {
            tools: BTreeMap::new(),
            max_tool_output_bytes: DEFAULT_TOOL_OUTPUT_BYTES,
        }
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the execution facade for this registry.
    pub fn gateway(&self) -> ToolGateway {
        ToolGateway::new(self.clone())
    }

    /// Set the maximum content/details size returned by registered tools.
    pub fn with_output_limit(mut self, max_bytes: usize) -> Self {
        self.max_tool_output_bytes = max_bytes;
        self
    }

    pub fn register<T>(&mut self, tool: T) -> Result<(), RegistryError>
    where
        T: Tool + 'static,
    {
        let definition = tool.definition();
        if definition.definition_type != "function" {
            return Err(RegistryError::InvalidDefinition {
                name: definition.function.name.clone(),
                reason: "definition type must be function".to_string(),
            });
        }
        if let Err(reason) = validate_tool_name(&definition.function.name) {
            return Err(RegistryError::InvalidDefinition {
                name: definition.function.name.clone(),
                reason,
            });
        }
        validate_schema_definition(&definition).map_err(|reason| {
            RegistryError::InvalidDefinition {
                name: definition.function.name.clone(),
                reason,
            }
        })?;
        let name = definition.function.name.clone();
        if self.tools.contains_key(&name) {
            return Err(RegistryError::DuplicateTool { name });
        }
        let capabilities = ToolCapabilities {
            concurrency: tool.concurrency(),
            replay_policy: tool.replay_policy(),
            exposure: tool.exposure(),
            risk_class: tool.risk_class(),
        };

        self.tools.insert(
            name,
            RegisteredTool {
                tool: Arc::new(tool),
                definition,
                capabilities,
            },
        );
        Ok(())
    }
}
