//! SKILL.md frontmatter parsing and targeted repair.

use serde::Deserialize;
use thiserror::Error;

use super::model::SkillError;

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

#[derive(Debug, Default, Deserialize)]
pub(crate) struct FrontmatterMetadata {
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    #[serde(rename = "disable-model-invocation")]
    pub(crate) disable_model_invocation: Option<bool>,
}

pub(crate) fn split_frontmatter(contents: &str) -> Result<(String, String), SkillDocumentError> {
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

pub(crate) fn parse_frontmatter(contents: &str) -> Result<FrontmatterMetadata, SkillDocumentError> {
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
