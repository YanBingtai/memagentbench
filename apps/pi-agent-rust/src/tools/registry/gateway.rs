//! Registry-backed capability facade used by the agent loop.

use super::*;

impl ToolGateway {
    pub fn new(registry: ToolRegistry) -> Self {
        Self { registry }
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    pub fn into_registry(self) -> ToolRegistry {
        self.registry
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.registry.definitions()
    }

    pub fn model_definitions(&self) -> Vec<ToolDefinition> {
        self.registry.model_definitions()
    }

    pub fn capabilities(&self, name: &str) -> Option<ToolCapabilities> {
        self.registry
            .tools
            .get(name)
            .map(|registered| registered.capabilities)
    }

    pub fn concurrency(&self, name: &str) -> Option<ToolConcurrency> {
        self.registry.concurrency(name)
    }

    pub fn replay_policy(&self, name: &str) -> Option<ToolReplayPolicy> {
        self.registry.replay_policy(name)
    }

    pub fn exposure(&self, name: &str) -> Option<ToolExposure> {
        self.registry.exposure(name)
    }

    pub fn risk_class(&self, name: &str) -> Option<RiskClass> {
        self.registry.risk_class(name)
    }

    /// Prepare a call produced by the model. Hidden tools are rejected here.
    pub fn prepare_model_call(
        &self,
        call: &ToolCall,
        arguments: Value,
    ) -> Result<PreparedToolCall, ToolError> {
        self.registry.prepare_model_call(call, arguments)
    }

    /// Prepare a trusted in-process call, including hidden capabilities.
    pub fn prepare_trusted_call(
        &self,
        call: &ToolCall,
        arguments: Value,
    ) -> Result<PreparedToolCall, ToolError> {
        self.registry.prepare_parsed_call(call, arguments)
    }

    pub async fn execute_prepared(
        &self,
        prepared: PreparedToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.registry.execute_prepared_result(prepared).await
    }

    pub async fn execute_prepared_result(
        &self,
        prepared: PreparedToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.execute_prepared(prepared).await
    }

    pub async fn execute_prepared_with_context(
        &self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        self.registry
            .execute_prepared_result_with_context(prepared, context)
            .await
    }

    pub async fn execute_prepared_result_with_context(
        &self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        self.execute_prepared_with_context(prepared, context).await
    }

    pub fn normalize_result(&self, result: ToolResult) -> ToolResult {
        self.registry.normalize_result(result)
    }

    pub fn output_limit(&self) -> usize {
        self.registry.output_limit()
    }

    pub fn without(&self, name: &str) -> Self {
        Self::new(self.registry.without(name))
    }

    /// Ensure the model can load full instructions for the supplied skills.
    ///
    /// The CLI may register this tool before constructing the agent. Library
    /// callers that only attach a catalog get the same capability without
    /// creating a duplicate registration.
    pub(crate) fn ensure_skill_reader(
        &mut self,
        catalog: SkillCatalog,
        max_output_bytes: usize,
    ) -> Result<(), RegistryError> {
        if self.registry.exposure("read_skill").is_none() {
            self.registry
                .register_skill_reader(catalog, max_output_bytes)?;
        }
        Ok(())
    }
}
