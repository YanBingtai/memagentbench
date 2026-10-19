use std::fmt;

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
