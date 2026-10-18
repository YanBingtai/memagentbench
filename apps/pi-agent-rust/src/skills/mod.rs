//! Skill domain, document parsing, discovery, and catalog facade.

mod catalog;
mod document;
mod limits;
mod loader;
mod model;

pub use catalog::SkillCatalog;
pub use document::SkillDocumentError;
pub use loader::{
    SkillDiagnostic, SkillDiagnosticError, SkillLoadReport, SkillLoader, SkillLoaderError,
};
pub use model::{Skill, SkillError};

#[cfg(test)]
mod tests;
