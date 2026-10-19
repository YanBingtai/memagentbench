//! SKILL.md frontmatter parsing and targeted repair.

mod frontmatter;
mod repair;

use super::model::SkillError;
use thiserror::Error;

pub(crate) use frontmatter::{parse_frontmatter, split_frontmatter};

/// Errors returned while parsing the metadata block of one `SKILL.md` file.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SkillDocumentError {
    #[error("SKILL.md is missing YAML frontmatter delimited by ---")]
    MissingFrontmatter,
    #[error("SKILL.md frontmatter is not terminated by ---")]
    UnterminatedFrontmatter,
    #[error("invalid YAML frontmatter: {message}")]
    InvalidYaml { message: String },
    #[error("frontmatter is missing required field {0:?}")]
    MissingField(&'static str),
    #[error(transparent)]
    InvalidSkill(#[from] SkillError),
}
