use serde::Deserialize;

use super::{repair::repair_frontmatter_scalar_fields, SkillDocumentError};

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
