//! Provider-specific construction from the application configuration.

use std::time::Duration;

use crate::{
    config::Config,
    model::{ChatOptions, ModelClient, ToolChoice},
    tools::ToolRegistry,
};

use super::super::{AgentError, AgentLoop};

impl AgentLoop<ModelClient> {
    /// Build an agent from the application's configuration and tool registry.
    pub fn from_config(config: &Config, tools: ToolRegistry) -> Result<Self, AgentError> {
        let client = ModelClient::new(
            &config.base_url,
            Duration::from_secs(config.request_timeout_secs),
        )?;
        let tools = tools.with_output_limit(config.max_tool_output_bytes);
        let agent = AgentLoop::new_with_policy(
            client,
            tools,
            config.model.clone(),
            config.api_key(),
            ChatOptions {
                temperature: config.temperature,
                enable_thinking: config.enable_thinking,
            },
            config.run_policy(),
        )?;
        Ok(match &config.system_prompt {
            Some(prompt) => agent.with_system_prompt(prompt.clone()),
            None => agent,
        })
    }

    /// Configure how the OpenAI-compatible provider selects advertised tools.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.client = self.client.with_tool_choice(tool_choice);
        self
    }
}
