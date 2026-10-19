use std::path::Path;

pub(super) fn default_skill_name(path: &Path) -> String {
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
