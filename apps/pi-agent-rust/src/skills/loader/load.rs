use std::fs;

use super::{
    discovery::discover_directory, parse::default_skill_name, read::read_skill_file,
    SkillDiagnostic, SkillDiagnosticError, SkillLoadReport, SkillLoader, SkillLoaderError,
};
use crate::skills::model::Skill;

impl SkillLoader {
    pub fn load_dir(
        root: impl Into<std::path::PathBuf>,
    ) -> Result<SkillLoadReport, SkillLoaderError> {
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
