use std::{
    fs,
    path::{Path, PathBuf},
};

use super::{SkillDiagnostic, SkillDiagnosticError};

#[derive(Debug)]
struct DirectoryEntry {
    path: PathBuf,
    name: String,
    file_type: fs::FileType,
}

pub(super) fn discover_directory(
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

fn is_ignored_directory(name: &str) -> bool {
    name == "node_modules" || name.starts_with('.')
}
