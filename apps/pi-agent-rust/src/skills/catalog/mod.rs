//! Deterministic skill collection used by routing and prompt assembly.

use std::collections::BTreeMap;

use super::model::{Skill, SkillError};

mod prompt;

/// Deterministic collection of skills used by routing and prompt assembly.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SkillCatalog {
    skills: BTreeMap<String, Skill>,
}

impl SkillCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one skill. Duplicate names are rejected instead of silently shadowing a skill.
    pub fn insert(&mut self, skill: Skill) -> Result<(), SkillError> {
        if self.skills.contains_key(&skill.name) {
            return Err(SkillError::DuplicateName {
                name: skill.name.clone(),
            });
        }
        self.skills.insert(skill.name.clone(), skill);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Skill> {
        self.skills.values()
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}
