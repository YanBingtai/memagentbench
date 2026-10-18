//! Deterministic filesystem discovery and diagnostics.

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use thiserror::Error;

use super::{
    catalog::SkillCatalog, document::SkillDocumentError, limits::DEFAULT_MAX_SKILL_FILE_BYTES,
    model::Skill,
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
    root: PathBuf,
    max_file_bytes: u64,
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

    pub fn load_dir(root: impl Into<PathBuf>) -> Result<SkillLoadReport, SkillLoaderError> {
        Self::new(root).load()
    }

    pub fn load(&self) -> Result<SkillLoadReport, SkillLoaderError> {
        let root = fs::canonicalize(&self.root).map_err(|error| SkillLoaderError::InspectRoot {
            path: self.root.clone(),
            message: error.to_string(),
        })?;
        if !root.is_dir() {
            return Err(SkillLoaderError::RootNotDirectory { path: root });
        }

        let mut report = SkillLoadReport::default();
        let mut files = Vec::new();
        discover_directory(&root, true, &mut files, &mut report.diagnostics);
        files.sort();

        for path in files {
            let contents = match read_skill_file(&path, self.max_file_bytes) {
                Ok(contents) => contents,
                Err(error @ SkillDiagnosticError::FileTooLarge { .. }) => {
                    report.diagnostics.push(SkillDiagnostic { path, error });
                    continue;
                }
                Err(error) => {
                    report.diagnostics.push(SkillDiagnostic { path, error });
                    continue;
                }
            };

            let default_name = default_skill_name(&path);
            let skill = match Skill::from_markdown(&contents, default_name, path.clone()) {
                Ok(skill) => skill,
                Err(error) => {
                    report.diagnostics.push(SkillDiagnostic {
                        path,
                        error: SkillDiagnosticError::Parse(error),
                    });
                    continue;
                }
            };
            let name = skill.name.clone();
            if report.catalog.insert(skill).is_err() {
                report.diagnostics.push(SkillDiagnostic {
                    path,
                    error: SkillDiagnosticError::DuplicateName { name },
                });
            }
        }

        Ok(report)
    }
}

fn read_skill_file(path: &Path, max_file_bytes: u64) -> Result<String, SkillDiagnosticError> {
    let file = fs::File::open(path).map_err(|error| SkillDiagnosticError::ReadFile {
        message: error.to_string(),
    })?;
    let metadata = file
        .metadata()
        .map_err(|error| SkillDiagnosticError::ReadFile {
            message: error.to_string(),
        })?;
    if metadata.len() > max_file_bytes {
        return Err(SkillDiagnosticError::FileTooLarge {
            size: metadata.len(),
            limit: max_file_bytes,
        });
    }

    let mut bytes = Vec::new();
    file.take(max_file_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| SkillDiagnosticError::ReadFile {
            message: error.to_string(),
        })?;
    if bytes.len() as u64 > max_file_bytes {
        return Err(SkillDiagnosticError::FileTooLarge {
            size: bytes.len() as u64,
            limit: max_file_bytes,
        });
    }

    String::from_utf8(bytes).map_err(|error| SkillDiagnosticError::ReadFile {
        message: error.to_string(),
    })
}

#[derive(Debug)]
struct DirectoryEntry {
    path: PathBuf,
    name: String,
    file_type: fs::FileType,
}

fn discover_directory(
    directory: &Path,
    is_root: bool,
    files: &mut Vec<PathBuf>,
    diagnostics: &mut Vec<SkillDiagnostic>,
) {
    let entries = match read_directory_entries(directory, diagnostics) {
        Some(entries) => entries,
        None => return,
    };

    if let Some(skill_file) = entries
        .iter()
        .find(|entry| entry.name == "SKILL.md" && entry.file_type.is_file())
    {
        files.push(skill_file.path.clone());
        return;
    }

    if is_root {
        files.extend(entries.iter().filter_map(|entry| {
            let is_markdown = Path::new(&entry.name)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("md"));
            (entry.file_type.is_file() && is_markdown && entry.name != "SKILL.md")
                .then(|| entry.path.clone())
        }));
    }

    for entry in entries {
        if !entry.file_type.is_dir()
            || entry.file_type.is_symlink()
            || is_ignored_directory(&entry.name)
        {
            continue;
        }
        discover_directory(&entry.path, false, files, diagnostics);
    }
}

fn read_directory_entries(
    directory: &Path,
    diagnostics: &mut Vec<SkillDiagnostic>,
) -> Option<Vec<DirectoryEntry>> {
    let read_dir = match fs::read_dir(directory) {
        Ok(read_dir) => read_dir,
        Err(error) => {
            diagnostics.push(SkillDiagnostic {
                path: directory.to_path_buf(),
                error: SkillDiagnosticError::ReadDirectory {
                    message: error.to_string(),
                },
            });
            return None;
        }
    };

    let mut entries = Vec::new();
    for result in read_dir {
        let entry = match result {
            Ok(entry) => entry,
            Err(error) => {
                diagnostics.push(SkillDiagnostic {
                    path: directory.to_path_buf(),
                    error: SkillDiagnosticError::InspectEntry {
                        message: error.to_string(),
                    },
                });
                continue;
            }
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(error) => {
                diagnostics.push(SkillDiagnostic {
                    path,
                    error: SkillDiagnosticError::InspectEntry {
                        message: error.to_string(),
                    },
                });
                continue;
            }
        };
        entries.push(DirectoryEntry {
            name: entry.file_name().to_string_lossy().into_owned(),
            path,
            file_type,
        });
    }
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    Some(entries)
}

fn default_skill_name(path: &Path) -> String {
    if path.file_name().is_some_and(|name| name == "SKILL.md") {
        path.parent()
            .and_then(Path::file_name)
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    } else {
        path.file_stem()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

fn is_ignored_directory(name: &str) -> bool {
    name == "node_modules" || name.starts_with('.')
}
