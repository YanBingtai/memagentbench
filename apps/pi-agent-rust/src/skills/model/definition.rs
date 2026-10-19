use std::path::{Path, PathBuf};

use super::{validation, SkillError};
use crate::skills::document::{parse_frontmatter, split_frontmatter, SkillDocumentError};

/// One validated instruction bundle available to the agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    /// Stable lowercase identifier used for explicit lookup.
    pub name: String,
    /// Short description used when the model decides which skill is relevant.
    pub description: String,
    /// Full instructions loaded when the skill is invoked.
    pub content: String,
    /// Source file used to resolve relative references in `content`.
    pub file_path: PathBuf,
    /// Keep the skill available for explicit lookup but hide it from model discovery.
    pub disable_model_invocation: bool,
}

impl Skill {
    /// Construct a skill after validating metadata that affects model routing.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        content: impl Into<String>,
        file_path: impl Into<PathBuf>,
    ) -> Result<Self, SkillError> {
        let name = name.into();
        validation::name(&name)?;

        let description = description.into();
        validation::description(&description)?;

        let file_path = file_path.into();
        if file_path.as_os_str().is_empty() {
            return Err(SkillError::EmptyPath);
        }

        Ok(Self {
            name,
            description,
            content: content.into(),
            file_path,
            disable_model_invocation: false,
        })
    }

    /// Parse one `SKILL.md` document with YAML frontmatter.
    ///
    /// The parser accepts the three fields used by the runtime contract and
    /// ignores unrelated metadata so skills can carry additional application
    /// fields. Full directory discovery and diagnostics are separate concerns.
    pub fn from_markdown(
        contents: &str,
        default_name: impl Into<String>,
        file_path: impl Into<PathBuf>,
    ) -> Result<Self, SkillDocumentError> {
        let (frontmatter, body) = split_frontmatter(contents)?;
        let metadata = parse_frontmatter(&frontmatter)?;
        let name = metadata.name.unwrap_or_else(|| default_name.into());
        let description = metadata
            .description
            .ok_or(SkillDocumentError::MissingField("description"))?;
        let mut skill = Self::new(name, description, body, file_path)
            .map_err(SkillDocumentError::InvalidSkill)?;
        skill.disable_model_invocation = metadata.disable_model_invocation.unwrap_or(false);
        Ok(skill)
    }

    /// Hide this skill from the model-visible catalog while retaining explicit lookup.
    pub fn with_model_invocation_disabled(mut self, disabled: bool) -> Self {
        self.disable_model_invocation = disabled;
        self
    }

    pub(crate) fn display_location(&self) -> String {
        self.file_path.to_string_lossy().into_owned()
    }

    pub(crate) fn skill_directory(&self) -> String {
        self.file_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_string_lossy()
            .into_owned()
    }
}
