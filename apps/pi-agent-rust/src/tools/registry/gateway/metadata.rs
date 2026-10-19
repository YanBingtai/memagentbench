//! Gateway accessors for registry metadata and capability contracts.

use pi_agent_runtime::RiskClass;

use crate::protocol::ToolDefinition;
use crate::tools::contracts::{
    ToolCapabilities, ToolConcurrency, ToolExposure, ToolReplayPolicy, ToolResult,
};

use super::super::{ToolGateway, ToolRegistry};

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

    pub fn normalize_result(&self, result: ToolResult) -> ToolResult {
        self.registry.normalize_result(result)
    }

    pub fn output_limit(&self) -> usize {
        self.registry.output_limit()
    }

    pub fn without(&self, name: &str) -> Self {
        Self::new(self.registry.without(name))
    }
}
