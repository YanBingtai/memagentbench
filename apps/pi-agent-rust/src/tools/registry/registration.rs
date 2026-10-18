//! Tool registration, discovery, and capability metadata.

use super::super::{
    limits::*, schema::validate_schema_definition, EditFileTool, ListFilesTool, ReadFileTool,
    ReadSkillTool, SearchFilesTool, ShellTool, ToolContext, ToolExposure, ToolReplayPolicy,
    WriteFileTool,
};
use super::validation::validate_tool_name;
use super::*;

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

    pub fn with_read_file(context: ToolContext) -> Result<Self, RegistryError> {
        let mut registry = Self::new().with_output_limit(context.max_tool_output_bytes);
        registry.register(ReadFileTool::new(context))?;
        Ok(registry)
    }

    pub fn register_write_file(&mut self, context: ToolContext) -> Result<(), RegistryError> {
        self.register(WriteFileTool::new(context))
    }

    pub fn register_edit_file(&mut self, context: ToolContext) -> Result<(), RegistryError> {
        self.register(EditFileTool::new(context))
    }

    pub fn register_shell(&mut self, context: ToolContext) -> Result<(), RegistryError> {
        // This compatibility helper is intentionally explicit in its
        // implementation: library callers that need isolation should build a
        // `ShellTool` with `with_sandbox_mode` and register it directly. The
        // CLI does that and defaults to managed mode.
        self.register(ShellTool::new(context))
    }

    pub fn register_list_files(&mut self, context: ToolContext) -> Result<(), RegistryError> {
        self.register(ListFilesTool::new(context))
    }

    pub fn register_search_files(&mut self, context: ToolContext) -> Result<(), RegistryError> {
        self.register(SearchFilesTool::new(context))
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

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|registered| registered.definition.clone())
            .collect()
    }

    /// Return only tools that may be advertised in a model request.
    pub fn model_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .filter(|registered| registered.capabilities.exposure == ToolExposure::Direct)
            .map(|registered| registered.definition.clone())
            .collect()
    }

    /// Return a registry without one capability.
    ///
    /// This is used when constructing a child agent so delegation cannot
    /// recurse accidentally through a cloned parent registry.
    pub fn without(&self, name: &str) -> Self {
        let mut registry = self.clone();
        registry.tools.remove(name);
        registry
    }

    /// Return the concurrency contract for one registered tool.
    pub fn concurrency(&self, name: &str) -> Option<ToolConcurrency> {
        self.tools
            .get(name)
            .map(|registered| registered.capabilities.concurrency)
    }

    /// Return the recovery policy for one registered tool.
    pub fn replay_policy(&self, name: &str) -> Option<ToolReplayPolicy> {
        self.tools
            .get(name)
            .map(|registered| registered.capabilities.replay_policy)
    }

    pub fn exposure(&self, name: &str) -> Option<ToolExposure> {
        self.tools
            .get(name)
            .map(|registered| registered.capabilities.exposure)
    }

    /// Return the declared runtime risk for one registered tool.
    pub fn risk_class(&self, name: &str) -> Option<RiskClass> {
        self.tools
            .get(name)
            .map(|registered| registered.capabilities.risk_class)
    }
    /// Register a name-based reader for the supplied skills.
    ///
    /// The tool accepts only catalog names and never arbitrary filesystem
    /// paths, so a model cannot use it to escape the configured skill roots.
    pub fn register_skill_reader(
        &mut self,
        catalog: SkillCatalog,
        max_output_bytes: usize,
    ) -> Result<(), RegistryError> {
        self.register(ReadSkillTool::new(catalog, max_output_bytes))
    }
}
