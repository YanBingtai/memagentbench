use std::path::{Path, PathBuf};

use thiserror::Error;

use super::super::{
    catalog::SkillCatalog, document::SkillDocumentError, limits::DEFAULT_MAX_SKILL_FILE_BYTES,
};

/// A non-fatal problem found while discovering one skill document.
///
/// Loading a directory should not discard valid skills because one unrelated
/// file is malformed. Callers can inspect these diagnostics and decide
/// whether they are warnings or startup errors for their application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillDiagnostic {
    pub path: PathBuf,
    pub error: SkillDiagnosticError,
}

/// Errors that can be reported for one file or directory entry during loading.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SkillDiagnosticError {
    #[error("could not read directory: {message}")]
    ReadDirectory { message: String },
    #[error("could not inspect directory entry: {message}")]
    InspectEntry { message: String },
    #[error("could not read skill file: {message}")]
    ReadFile { message: String },
    #[error("skill file is too large ({size} bytes, limit {limit} bytes)")]
    FileTooLarge { size: u64, limit: u64 },
    #[error("skill document is invalid: {0}")]
    Parse(SkillDocumentError),
    #[error("skill name {name:?} is already registered")]
    DuplicateName { name: String },
}

/// Fatal errors for the root supplied to [`SkillLoader`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SkillLoaderError {
    #[error("could not inspect skill root {path:?}: {message}")]
    InspectRoot { path: PathBuf, message: String },
    #[error("skill root {path:?} is not a directory")]
    RootNotDirectory { path: PathBuf },
}

/// Result of loading a skill root, including valid skills and non-fatal issues.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SkillLoadReport {
    pub catalog: SkillCatalog,
    pub diagnostics: Vec<SkillDiagnostic>,
}

impl SkillLoadReport {
    pub fn catalog(&self) -> &SkillCatalog {
        &self.catalog
    }

    pub fn diagnostics(&self) -> &[SkillDiagnostic] {
        &self.diagnostics
    }
}

/// Discover `SKILL.md` files below one root in deterministic order.
///
/// A directory containing `SKILL.md` is treated as one skill and is not
/// traversed further. Hidden directories and `node_modules` are skipped, and
/// direct `*.md` files in the root are accepted as a compatibility format.
/// Frontmatter is parsed as YAML; the body remains a separate instruction
/// string and is never interpreted by the metadata parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillLoader {
    pub(super) root: PathBuf,
    pub(super) max_file_bytes: u64,
}

impl SkillLoader {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            max_file_bytes: DEFAULT_MAX_SKILL_FILE_BYTES,
        }
    }

    /// Set the maximum number of bytes read from one discovered skill file.
    ///
    /// The default is one mebibyte. A caller that owns the surrounding
    /// configuration can lower or raise the limit without changing discovery
    /// or parsing behavior.
    pub fn with_max_file_bytes(mut self, max_file_bytes: u64) -> Self {
        self.max_file_bytes = max_file_bytes;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}
