use crate::skills::SkillCatalog;

use super::super::super::AgentLoop;

impl<M> AgentLoop<M> {
    /// Attach a stable, ephemeral system prompt to every model request.
    ///
    /// The prompt is inserted before session messages and is never appended to
    /// the durable transcript. Empty prompts are ignored so callers can pass
    /// optional configuration without creating blank system messages.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();
        self.system_prompt = (!prompt.trim().is_empty()).then_some(prompt);
        self
    }

    /// Attach skills whose metadata should be visible to the model.
    ///
    /// The catalog index is added as an ephemeral system message for model
    /// requests. It is not written to the durable session transcript, so a
    /// later run can rebuild it from the current skill roots.
    pub fn with_skills(mut self, skills: SkillCatalog) -> Self {
        if !skills.is_empty() {
            let max_output_bytes = self.tools.output_limit();
            self.tools
                .ensure_skill_reader(skills.clone(), max_output_bytes)
                .expect("the built-in read_skill tool must remain valid");
            self.skills = Some(skills);
        } else {
            self.skills = None;
        }
        self
    }

    /// Set the current delegation depth used in child-agent lineage metadata.
    pub fn with_delegation_depth(mut self, depth: u16) -> Self {
        self.delegation_depth = depth;
        self
    }
}
