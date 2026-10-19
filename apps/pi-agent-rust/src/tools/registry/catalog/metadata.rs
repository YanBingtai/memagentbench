use pi_agent_runtime::RiskClass;

use crate::protocol::ToolDefinition;

use super::super::{ToolConcurrency, ToolExposure, ToolRegistry, ToolReplayPolicy};

impl ToolRegistry {
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
}
