use pi_agent_runtime::RiskClass;

use crate::{
    model::{ChatOptions, ModelClient},
    skills::SkillCatalog,
    tools::ToolRegistry,
};

use super::{SubagentTool, DEFAULT_MAX_OUTPUT_BYTES};

impl SubagentTool {
    // Keep the public constructor explicit: each dependency is an intentional
    // capability boundary for the child agent. A settings struct can be added
    // later without silently changing this compatibility surface.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: ModelClient,
        model: impl Into<String>,
        api_key: Option<String>,
        options: ChatOptions,
        tools: ToolRegistry,
        policy: crate::config::RunPolicy,
        max_runtime_secs: u64,
        max_output_bytes: usize,
    ) -> Self {
        let max_runtime_secs = max_runtime_secs.max(1).min(policy.tool_timeout_secs.max(1));
        let max_output_bytes = if max_output_bytes == 0 {
            DEFAULT_MAX_OUTPUT_BYTES
        } else {
            max_output_bytes.min(DEFAULT_MAX_OUTPUT_BYTES)
        };
        let mut child_tools = tools.without("spawn_subagent");
        for definition in child_tools.definitions() {
            let name = definition.function.name;
            if child_tools.risk_class(&name) != Some(RiskClass::ReadOnly) {
                child_tools = child_tools.without(&name);
            }
        }

        Self {
            client,
            model: model.into(),
            api_key,
            options,
            tools: child_tools,
            policy,
            max_runtime_secs,
            max_output_bytes,
            skills: None,
        }
    }

    /// Give child agents the same model-visible skill catalog as their parent.
    pub fn with_skills(mut self, skills: SkillCatalog) -> Self {
        if !skills.is_empty() {
            if self.tools.exposure("read_skill").is_none() {
                self.tools
                    .register_skill_reader(skills.clone(), self.max_output_bytes)
                    .expect("the built-in read_skill tool must remain valid");
            }
            self.skills = Some(skills);
        } else {
            self.skills = None;
        }
        self
    }
}
