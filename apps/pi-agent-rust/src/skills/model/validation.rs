use super::SkillError;
use crate::skills::limits::{MAX_DESCRIPTION_CHARS, MAX_NAME_CHARS};

pub(super) fn name(name: &str) -> Result<(), SkillError> {
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

pub(super) fn description(description: &str) -> Result<(), SkillError> {
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
