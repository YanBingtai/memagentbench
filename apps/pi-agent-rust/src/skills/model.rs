//! Validated skill metadata and instruction bundles.

use std::{
    fmt,
    path::{Path, PathBuf},
};

use super::{
    document::{parse_frontmatter, split_frontmatter, SkillDocumentError},
    limits::*,
};

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
        validate_name(&name)?;

        let description = description.into();
        validate_description(&description)?;

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

/// Errors returned while validating or looking up skills.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillError {
    EmptyName,
    InvalidName { name: String, reason: &'static str },
    NameTooLong { name: String, limit: usize },
    EmptyDescription,
    DescriptionTooLong { limit: usize },
    EmptyPath,
    DuplicateName { name: String },
    UnknownSkill { name: String },
}

impl fmt::Display for SkillError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => write!(formatter, "skill name must not be empty"),
            Self::InvalidName { name, reason } => {
                write!(formatter, "invalid skill name {name:?}: {reason}")
            }
            Self::NameTooLong { name, limit } => {
                write!(formatter, "skill name {name:?} exceeds {limit} characters")
            }
            Self::EmptyDescription => write!(formatter, "skill description must not be empty"),
            Self::DescriptionTooLong { limit } => {
                write!(formatter, "skill description exceeds {limit} characters")
            }
            Self::EmptyPath => write!(formatter, "skill file path must not be empty"),
            Self::DuplicateName { name } => {
                write!(formatter, "skill {name:?} is already registered")
            }
            Self::UnknownSkill { name } => write!(formatter, "unknown skill {name:?}"),
        }
    }
}

impl std::error::Error for SkillError {}

fn validate_name(name: &str) -> Result<(), SkillError> {
    if name.is_empty() {
        return Err(SkillError::EmptyName);
    }
    if name.chars().count() > MAX_NAME_CHARS {
        return Err(SkillError::NameTooLong {
            name: name.to_string(),
            limit: MAX_NAME_CHARS,
        });
    }
    if name.starts_with('-') || name.ends_with('-') || name.contains("--") {
        return Err(SkillError::InvalidName {
            name: name.to_string(),
            reason: "must not start, end, or contain consecutive hyphens",
        });
    }
    if !name.chars().all(|character| {
        character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-'
    }) {
        return Err(SkillError::InvalidName {
            name: name.to_string(),
            reason: "must contain only lowercase letters, digits, and hyphens",
        });
    }
    Ok(())
}

fn validate_description(description: &str) -> Result<(), SkillError> {
    if description.trim().is_empty() {
        return Err(SkillError::EmptyDescription);
    }
    if description.chars().count() > MAX_DESCRIPTION_CHARS {
        return Err(SkillError::DescriptionTooLong {
            limit: MAX_DESCRIPTION_CHARS,
        });
    }
    Ok(())
}
