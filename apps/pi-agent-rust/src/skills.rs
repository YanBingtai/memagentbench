//! Domain types for reusable agent skills.
//!
//! A skill is a named instruction bundle. This module deliberately does not
//! know how a skill was discovered on disk; loading `SKILL.md` files is a
//! separate boundary that we will add later. Keeping the catalog independent
//! makes it usable by a CLI, TUI, or service and lets tests inject skills
//! without touching the filesystem.

use std::{
    collections::BTreeMap,
    fmt, fs,
    io::Read,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use thiserror::Error;

const MAX_NAME_CHARS: usize = 64;
const MAX_DESCRIPTION_CHARS: usize = 1_024;
const DEFAULT_MAX_SKILL_FILE_BYTES: u64 = 1_048_576;

/// One validated instruction bundle available to the agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    /// Stable lowercase identifier used for explicit lookup.
    pub name: String,
    /// Short description used when the model decides which skill is relevant.
    pub description: String,
    /// Full instructions loaded when the skill is invoked.
    pub content: String,
    /// Source file used to resolve relative references in `content`.
    pub file_path: PathBuf,
    /// Keep the skill available for explicit lookup but hide it from model discovery.
    pub disable_model_invocation: bool,
}

impl Skill {
    /// Construct a skill after validating metadata that affects model routing.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        content: impl Into<String>,
        file_path: impl Into<PathBuf>,
    ) -> Result<Self, SkillError> {
        let name = name.into();
        validate_name(&name)?;

        let description = description.into();
        validate_description(&description)?;

        let file_path = file_path.into();
        if file_path.as_os_str().is_empty() {
            return Err(SkillError::EmptyPath);
        }

        Ok(Self {
            name,
            description,
            content: content.into(),
            file_path,
            disable_model_invocation: false,
        })
    }

    /// Parse one `SKILL.md` document with YAML frontmatter.
    ///
    /// The parser accepts the three fields used by the runtime contract and
    /// ignores unrelated metadata so skills can carry additional application
    /// fields. Full directory discovery and diagnostics are separate concerns.
    pub fn from_markdown(
        contents: &str,
        default_name: impl Into<String>,
        file_path: impl Into<PathBuf>,
    ) -> Result<Self, SkillDocumentError> {
        let (frontmatter, body) = split_frontmatter(contents)?;
        let metadata = parse_frontmatter(&frontmatter)?;
        let name = metadata.name.unwrap_or_else(|| default_name.into());
        let description = metadata
            .description
            .ok_or(SkillDocumentError::MissingField("description"))?;
        let mut skill = Self::new(name, description, body, file_path)
            .map_err(SkillDocumentError::InvalidSkill)?;
        skill.disable_model_invocation = metadata.disable_model_invocation.unwrap_or(false);
        Ok(skill)
    }

    /// Hide this skill from the model-visible catalog while retaining explicit lookup.
    pub fn with_model_invocation_disabled(mut self, disabled: bool) -> Self {
        self.disable_model_invocation = disabled;
        self
    }

    fn display_location(&self) -> String {
        self.file_path.to_string_lossy().into_owned()
    }

    fn skill_directory(&self) -> String {
        self.file_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_string_lossy()
            .into_owned()
    }
}

/// Errors returned while validating or looking up skills.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillError {
    EmptyName,
    InvalidName { name: String, reason: &'static str },
    NameTooLong { name: String, limit: usize },
    EmptyDescription,
    DescriptionTooLong { limit: usize },
    EmptyPath,
    DuplicateName { name: String },
    UnknownSkill { name: String },
}

impl fmt::Display for SkillError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => write!(formatter, "skill name must not be empty"),
            Self::InvalidName { name, reason } => {
                write!(formatter, "invalid skill name {name:?}: {reason}")
            }
            Self::NameTooLong { name, limit } => {
                write!(formatter, "skill name {name:?} exceeds {limit} characters")
            }
            Self::EmptyDescription => write!(formatter, "skill description must not be empty"),
            Self::DescriptionTooLong { limit } => {
                write!(formatter, "skill description exceeds {limit} characters")
            }
            Self::EmptyPath => write!(formatter, "skill file path must not be empty"),
            Self::DuplicateName { name } => {
                write!(formatter, "skill {name:?} is already registered")
            }
            Self::UnknownSkill { name } => write!(formatter, "unknown skill {name:?}"),
        }
    }
}

impl std::error::Error for SkillError {}

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

#[derive(Debug, Default, Deserialize)]
struct FrontmatterMetadata {
    name: Option<String>,
    description: Option<String>,
    #[serde(rename = "disable-model-invocation")]
    disable_model_invocation: Option<bool>,
}

fn split_frontmatter(contents: &str) -> Result<(String, String), SkillDocumentError> {
    let contents = contents.strip_prefix('\u{feff}').unwrap_or(contents);
    let mut lines = contents.split_inclusive('\n');
    let opening = lines.next().map(strip_line_ending);
    if opening != Some("---") {
        return Err(SkillDocumentError::MissingFrontmatter);
    }

    let mut metadata = Vec::new();
    for line in lines.by_ref() {
        if strip_line_ending(line) == "---" {
            return Ok((metadata.join("\n"), lines.collect::<String>()));
        }
        metadata.push(strip_line_ending(line));
    }
    Err(SkillDocumentError::UnterminatedFrontmatter)
}

fn strip_line_ending(line: &str) -> &str {
    let line = line.strip_suffix('\n').unwrap_or(line);
    line.strip_suffix('\r').unwrap_or(line)
}

fn parse_frontmatter(contents: &str) -> Result<FrontmatterMetadata, SkillDocumentError> {
    let parsed: Result<FrontmatterMetadata, serde_yaml::Error> =
        match serde_yaml::from_str(contents) {
            Ok(metadata) => Ok(metadata),
            Err(original_error) => match repair_frontmatter_scalar_fields(contents) {
                Some(repaired) => serde_yaml::from_str(&repaired).map_err(|_| original_error),
                None => Err(original_error),
            },
        };

    parsed.map_err(|error| SkillDocumentError::InvalidYaml {
        message: error.to_string(),
    })
}

/// Repair only the prose scalars that Codex accepts from third-party skills.
///
/// YAML treats a colon followed by whitespace as a mapping separator. Skill
/// authors commonly write descriptions such as `Build for AWS: ECS` without
/// quoting them, so the strict parser rejects an otherwise useful document.
/// We quote those scalar values and retry YAML parsing; all other syntax keeps
/// the original parser error.
fn repair_frontmatter_scalar_fields(frontmatter: &str) -> Option<String> {
    let mut changed = false;
    let mut block_scalar_indent = None;
    let mut repaired_lines = Vec::new();

    for line in frontmatter.lines() {
        let indent = line
            .chars()
            .take_while(|character| *character == ' ')
            .count();
        if let Some(block_indent) = block_scalar_indent {
            if line.trim().is_empty() || indent > block_indent {
                repaired_lines.push(line.to_string());
                continue;
            }
            block_scalar_indent = None;
        }

        let Some((key, value)) = line.split_once(':') else {
            repaired_lines.push(line.to_string());
            continue;
        };
        if key.trim().is_empty()
            || value
                .chars()
                .next()
                .is_none_or(|character| !character.is_whitespace())
        {
            repaired_lines.push(line.to_string());
            continue;
        }

        let trimmed_start = value.trim_start();
        let leading_whitespace = &value[..value.len() - trimmed_start.len()];
        let mut scalar = trimmed_start;
        let mut comment = "";
        for (index, character) in trimmed_start.char_indices() {
            if character == '#'
                && (index == 0
                    || trimmed_start[..index]
                        .chars()
                        .next_back()
                        .is_some_and(char::is_whitespace))
            {
                let comment_start = trimmed_start[..index].trim_end().len();
                scalar = &trimmed_start[..comment_start];
                comment = &trimmed_start[comment_start..];
                break;
            }
        }

        let scalar = scalar.trim_end();
        let Some(first_character) = scalar.chars().next() else {
            repaired_lines.push(line.to_string());
            continue;
        };
        if matches!(first_character, '|' | '>') {
            block_scalar_indent = Some(indent);
            repaired_lines.push(line.to_string());
            continue;
        }
        if matches!(first_character, '\'' | '"') {
            repaired_lines.push(line.to_string());
            continue;
        }

        let mut characters = scalar.chars().peekable();
        let mut has_colon_separator = false;
        while let Some(character) = characters.next() {
            if character == ':'
                && characters
                    .peek()
                    .is_some_and(|next_character| next_character.is_whitespace())
            {
                has_colon_separator = true;
                break;
            }
        }
        let invalid_flow_like_scalar = matches!(first_character, '[' | '{' | '@' | '`')
            && serde_yaml::from_str::<serde_yaml::Value>(scalar).is_err();
        if !has_colon_separator && !invalid_flow_like_scalar {
            repaired_lines.push(line.to_string());
            continue;
        }

        let quoted_scalar = format!("'{}'", scalar.replace('\'', "''"));
        repaired_lines.push(format!(
            "{key}:{leading_whitespace}{quoted_scalar}{comment}"
        ));
        changed = true;
    }

    changed.then(|| repaired_lines.join("\n"))
}

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

    /// Format the model-visible skill index. Full instructions are loaded only on explicit use.
    pub fn format_for_system_prompt(&self) -> String {
        let visible: Vec<&Skill> = self
            .skills
            .values()
            .filter(|skill| !skill.disable_model_invocation)
            .collect();
        if visible.is_empty() {
            return String::new();
        }

        let mut lines = vec![
            "The following skills provide specialized instructions for specific tasks.".to_string(),
            "Read the full skill file when the task matches its description.".to_string(),
            "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.".to_string(),
            String::new(),
            "<available_skills>".to_string(),
        ];
        for skill in visible {
            lines.push("  <skill>".to_string());
            lines.push(format!("    <name>{}</name>", escape_xml(&skill.name)));
            lines.push(format!(
                "    <description>{}</description>",
                escape_xml(&skill.description)
            ));
            lines.push(format!(
                "    <location>{}</location>",
                escape_xml(&skill.display_location())
            ));
            lines.push("  </skill>".to_string());
        }
        lines.push("</available_skills>".to_string());
        lines.join("\n")
    }

    /// Format the full instructions for one explicit skill invocation.
    pub fn format_invocation(
        &self,
        name: &str,
        additional_instructions: Option<&str>,
    ) -> Result<String, SkillError> {
        let skill = self.get(name).ok_or_else(|| SkillError::UnknownSkill {
            name: name.to_string(),
        })?;
        let mut prompt = format!(
            "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
            escape_xml(&skill.name),
            escape_xml(&skill.display_location()),
            escape_xml(&skill.skill_directory()),
            skill.content
        );
        if let Some(additional) = additional_instructions.filter(|text| !text.is_empty()) {
            prompt.push_str("\n\n");
            prompt.push_str(additional);
        }
        Ok(prompt)
    }
}

fn validate_name(name: &str) -> Result<(), SkillError> {
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

fn validate_description(description: &str) -> Result<(), SkillError> {
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

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(name: &str, description: &str) -> Skill {
        Skill::new(
            name,
            description,
            "Follow these instructions.",
            format!("/skills/{name}/SKILL.md"),
        )
        .expect("fixture skill should be valid")
    }

    fn write_skill(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("fixture directory should be created");
        }
        std::fs::write(path, contents).expect("fixture skill should be written");
    }

    #[test]
    fn validates_names_and_descriptions_before_registration() {
        assert!(matches!(
            Skill::new("Bad_Name", "description", "content", "/tmp/SKILL.md"),
            Err(SkillError::InvalidName { .. })
        ));
        assert!(matches!(
            Skill::new("valid", "  ", "content", "/tmp/SKILL.md"),
            Err(SkillError::EmptyDescription)
        ));
    }

    #[test]
    fn parses_bom_crlf_frontmatter_and_preserves_skill_body() {
        let document = "\u{feff}---\r\nname: research\r\ndescription: 'Build A:B evidence'\r\ndisable-model-invocation: true\r\n---\r\nUse primary sources.\r\nKeep citations.\r\n";

        let skill = Skill::from_markdown(document, "fallback", "/skills/research/SKILL.md")
            .expect("valid SKILL.md should parse");

        assert_eq!(skill.name, "research");
        assert_eq!(skill.description, "Build A:B evidence");
        assert!(skill.disable_model_invocation);
        assert_eq!(skill.content, "Use primary sources.\r\nKeep citations.\r\n");
    }

    #[test]
    fn uses_default_name_when_frontmatter_omits_name() {
        let skill = Skill::from_markdown(
            "---\ndescription: A skill without an explicit name\n---\nInstructions",
            "directory-name",
            "/skills/directory-name/SKILL.md",
        )
        .expect("description-only frontmatter should use the loader name");

        assert_eq!(skill.name, "directory-name");
        assert_eq!(skill.content, "Instructions");
        assert!(!skill.disable_model_invocation);
    }

    #[test]
    fn rejects_invalid_yaml_and_unterminated_frontmatter() {
        assert!(matches!(
            Skill::from_markdown(
                "---\nname: first\nname: second\ndescription: test\n---\nbody",
                "fallback",
                "/skills/test/SKILL.md"
            ),
            Err(SkillDocumentError::InvalidYaml { .. })
        ));
        assert!(matches!(
            Skill::from_markdown(
                "---\ndescription: test\ndisable-model-invocation: yes\n---\nbody",
                "fallback",
                "/skills/test/SKILL.md"
            ),
            Err(SkillDocumentError::InvalidYaml { .. })
        ));
        assert!(matches!(
            Skill::from_markdown(
                "---\ndescription: test\nbody",
                "fallback",
                "/skills/test/SKILL.md"
            ),
            Err(SkillDocumentError::UnterminatedFrontmatter)
        ));
    }

    #[test]
    fn rejects_unclosed_quoted_scalars() {
        let error = Skill::from_markdown(
            "---\ndescription: 'unfinished\n---\nbody",
            "fallback",
            "/skills/test/SKILL.md",
        )
        .expect_err("unclosed quotes should not be accepted");

        assert!(matches!(error, SkillDocumentError::InvalidYaml { .. }));
    }

    #[test]
    fn parses_multiline_yaml_and_ignores_application_metadata() {
        let document = "---\nname: research\ndescription: |\n  Gather evidence from\n  primary sources.\nmetadata:\n  owner: team-research\ndisable-model-invocation: false\n---\nInstructions\n";

        let skill = Skill::from_markdown(document, "fallback", "/skills/research/SKILL.md")
            .expect("multiline YAML frontmatter should parse");

        assert_eq!(
            skill.description,
            "Gather evidence from\nprimary sources.\n"
        );
        assert_eq!(skill.content, "Instructions\n");
        assert!(!skill.disable_model_invocation);
    }

    #[test]
    fn repairs_common_unquoted_prose_before_yaml_parsing() {
        let document = "---\nname: research\ndescription: Build for AWS: ECS\nargument-hint: <duration: e.g. 7d>\n---\nInstructions\n";

        let skill = Skill::from_markdown(document, "fallback", "/skills/research/SKILL.md")
            .expect("common unquoted prose should be repaired");

        assert_eq!(skill.description, "Build for AWS: ECS");
        assert_eq!(skill.content, "Instructions\n");
    }

    #[test]
    fn repairs_scalar_fields_without_touching_block_scalar_content() {
        let document = "---\nname: research\ndescription: |-\n  Build for AWS: ECS\nargument-hint: <duration: e.g. 7d>\n---\nInstructions\n";

        let skill = Skill::from_markdown(document, "fallback", "/skills/research/SKILL.md")
            .expect("block scalar content should remain valid");

        assert_eq!(skill.description, "Build for AWS: ECS");
        assert_eq!(skill.content, "Instructions\n");
    }

    #[test]
    fn catalog_rejects_duplicates_and_keeps_deterministic_order() {
        let mut catalog = SkillCatalog::new();
        catalog.insert(skill("z-last", "Z")).unwrap();
        catalog.insert(skill("a-first", "A")).unwrap();

        assert_eq!(
            catalog
                .iter()
                .map(|skill| skill.name.as_str())
                .collect::<Vec<_>>(),
            ["a-first", "z-last"]
        );
        assert!(matches!(
            catalog.insert(skill("a-first", "duplicate")),
            Err(SkillError::DuplicateName { .. })
        ));
    }

    #[test]
    fn system_prompt_hides_disabled_skills_and_escapes_metadata() {
        let mut catalog = SkillCatalog::new();
        catalog
            .insert(
                Skill::new(
                    "safe",
                    "Use <safe> & careful",
                    "safe instructions",
                    "/skills/safe/SKILL.md",
                )
                .unwrap(),
            )
            .unwrap();
        catalog
            .insert(skill("hidden", "not model visible").with_model_invocation_disabled(true))
            .unwrap();

        let prompt = catalog.format_for_system_prompt();
        assert!(prompt.contains("<name>safe</name>"));
        assert!(prompt.contains("Use &lt;safe&gt; &amp; careful"));
        assert!(!prompt.contains("hidden"));
    }

    #[test]
    fn invocation_includes_reference_directory_and_extra_instructions() {
        let mut catalog = SkillCatalog::new();
        catalog.insert(skill("research", "Research tasks")).unwrap();

        let prompt = catalog
            .format_invocation("research", Some("Focus on primary sources."))
            .unwrap();
        assert!(prompt.contains("References are relative to /skills/research."));
        assert!(prompt.contains("Follow these instructions."));
        assert!(prompt.ends_with("Focus on primary sources."));
    }

    #[test]
    fn unknown_invocation_is_typed_error() {
        let catalog = SkillCatalog::new();
        assert!(matches!(
            catalog.format_invocation("missing", None),
            Err(SkillError::UnknownSkill { .. })
        ));
    }

    #[test]
    fn loader_discovers_nested_skills_and_root_markdown_in_order() {
        let root = tempfile::tempdir().expect("temporary root should be created");
        write_skill(
            &root.path().join("legacy.md"),
            "---\nname: legacy\ndescription: Root compatibility skill\n---\nLegacy body",
        );
        write_skill(
            &root.path().join("nested/research/SKILL.md"),
            "---\nname: research\ndescription: Research skill\n---\nResearch body",
        );

        let report = SkillLoader::load_dir(root.path()).expect("skill root should load");

        assert_eq!(report.catalog().len(), 2);
        assert_eq!(
            report
                .catalog()
                .iter()
                .map(|skill| skill.name.as_str())
                .collect::<Vec<_>>(),
            ["legacy", "research"]
        );
        assert!(report.diagnostics().is_empty());
    }

    #[test]
    fn loader_stops_at_skill_marker_and_skips_hidden_directories() {
        let root = tempfile::tempdir().expect("temporary root should be created");
        write_skill(
            &root.path().join("group/SKILL.md"),
            "---\nname: group\ndescription: Group skill\n---\nGroup body",
        );
        write_skill(
            &root.path().join("group/nested/SKILL.md"),
            "---\nname: nested\ndescription: Should not be discovered\n---\nNested body",
        );
        write_skill(
            &root.path().join(".hidden/SKILL.md"),
            "---\nname: hidden\ndescription: Should not be discovered\n---\nHidden body",
        );
        write_skill(
            &root.path().join("node_modules/dependency/SKILL.md"),
            "---\nname: dependency\ndescription: Should not be discovered\n---\nDependency body",
        );

        let report = SkillLoader::load_dir(root.path()).expect("skill root should load");

        assert_eq!(report.catalog().len(), 1);
        assert!(report.catalog().get("group").is_some());
        assert!(report.catalog().get("nested").is_none());
        assert!(report.catalog().get("hidden").is_none());
        assert!(report.catalog().get("dependency").is_none());
    }

    #[test]
    fn loader_keeps_valid_skills_and_reports_parse_and_duplicate_errors() {
        let root = tempfile::tempdir().expect("temporary root should be created");
        write_skill(
            &root.path().join("a/SKILL.md"),
            "---\nname: shared\ndescription: First skill\n---\nFirst body",
        );
        write_skill(
            &root.path().join("b/SKILL.md"),
            "---\nname: shared\ndescription: Duplicate skill\n---\nSecond body",
        );
        write_skill(
            &root.path().join("broken/SKILL.md"),
            "---\nname: broken\n---\nMissing description",
        );

        let report = SkillLoader::load_dir(root.path()).expect("skill root should load");

        assert_eq!(report.catalog().len(), 1);
        assert_eq!(
            report.catalog().get("shared").unwrap().content,
            "First body"
        );
        assert_eq!(report.diagnostics().len(), 2);
        assert!(report.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic.error,
            SkillDiagnosticError::DuplicateName { ref name } if name == "shared"
        )));
        assert!(report.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic.error,
            SkillDiagnosticError::Parse(SkillDocumentError::MissingField("description"))
        )));
    }

    #[test]
    fn loader_reports_oversized_files_without_reading_them_into_the_catalog() {
        let root = tempfile::tempdir().expect("temporary root should be created");
        let path = root.path().join("large/SKILL.md");
        write_skill(
            &path,
            &format!(
                "---\nname: large\ndescription: Large skill\n---\n{}",
                "x".repeat(64)
            ),
        );

        let report = SkillLoader::new(root.path())
            .with_max_file_bytes(32)
            .load()
            .expect("skill root should load");

        assert!(report.catalog().is_empty());
        assert!(matches!(
            report.diagnostics().first().map(|diagnostic| &diagnostic.error),
            Some(SkillDiagnosticError::FileTooLarge { size, limit })
                if *size > *limit && *limit == 32
        ));
    }

    #[test]
    fn loader_rejects_a_file_as_root() {
        let root = tempfile::tempdir().expect("temporary root should be created");
        let file = root.path().join("not-a-directory");
        std::fs::write(&file, "content").expect("fixture file should be written");

        let error = SkillLoader::load_dir(&file).expect_err("file root should be rejected");

        assert!(matches!(
            error,
            SkillLoaderError::RootNotDirectory { path } if path == file
        ));
    }
}
