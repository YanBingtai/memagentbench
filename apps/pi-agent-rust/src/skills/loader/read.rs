use std::{fs, io::Read, path::Path};

use super::SkillDiagnosticError;

pub(super) fn read_skill_file(
    path: &Path,
    max_file_bytes: u64,
) -> Result<String, SkillDiagnosticError> {
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
