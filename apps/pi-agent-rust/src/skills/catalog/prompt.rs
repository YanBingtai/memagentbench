use super::SkillCatalog;
use crate::skills::model::SkillError;

impl SkillCatalog {
    /// Format the model-visible skill index. Full instructions are loaded only on explicit use.
    pub fn format_for_system_prompt(&self) -> String {
        let visible: Vec<_> = self
            .skills
            .values()
            .filter(|skill| !skill.disable_model_invocation)
            .collect();
        if visible.is_empty() {
            return String::new();
        }

        let mut lines = vec![
            "The following skills provide specialized instructions for specific tasks.".to_string(),
            "Read the full skill file when the task matches its description.".to_string(),
            "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.".to_string(),
            String::new(),
            "<available_skills>".to_string(),
        ];
        for skill in visible {
            lines.push("  <skill>".to_string());
            lines.push(format!("    <name>{}</name>", escape_xml(&skill.name)));
            lines.push(format!(
                "    <description>{}</description>",
                escape_xml(&skill.description)
            ));
            lines.push(format!(
                "    <location>{}</location>",
                escape_xml(&skill.display_location())
            ));
            lines.push("  </skill>".to_string());
        }
        lines.push("</available_skills>".to_string());
        lines.join("\n")
    }

    /// Format the full instructions for one explicit skill invocation.
    pub fn format_invocation(
        &self,
        name: &str,
        additional_instructions: Option<&str>,
    ) -> Result<String, SkillError> {
        let skill = self.get(name).ok_or_else(|| SkillError::UnknownSkill {
            name: name.to_string(),
        })?;
        let mut prompt = format!(
            "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
            escape_xml(&skill.name),
            escape_xml(&skill.display_location()),
            escape_xml(&skill.skill_directory()),
            skill.content
        );
        if let Some(additional) = additional_instructions.filter(|text| !text.is_empty()) {
            prompt.push_str("\n\n");
            prompt.push_str(additional);
        }
        Ok(prompt)
    }
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
