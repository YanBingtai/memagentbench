//! Convert provider hits into bounded, model-facing projections.
//!
//! Providers may return text, structured JSON, or references to large
//! artifacts. Text and structured values are rendered inline; large or binary
//! values are represented by a bounded JSON reference so the model can still
//! identify the supporting artifact without loading its payload.

use std::{error::Error, fmt};

use pi_agent_memory::{ContentRef, MemoryHit};
use pi_agent_runtime::MemoryProjection;

/// Default maximum number of Unicode scalar values copied into one projection.
pub const DEFAULT_MAX_CHARS: usize = 8 * 1024;

/// A bounded renderer for provider-side projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionRenderer {
    max_chars: usize,
}

impl Default for ProjectionRenderer {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_MAX_CHARS,
        }
    }
}

impl ProjectionRenderer {
    pub fn new(max_chars: usize) -> Result<Self, ProjectionError> {
        if max_chars == 0 {
            return Err(ProjectionError::InvalidLimit);
        }
        Ok(Self { max_chars })
    }

    pub const fn max_chars(self) -> usize {
        self.max_chars
    }

    /// Renders a hit into a bounded model-facing projection.
    ///
    /// `Ok(None)` means the hit did not include a projection. Binary content is
    /// represented by its artifact metadata rather than being dropped.
    pub fn render(&self, hit: &MemoryHit) -> Result<Option<MemoryProjection>, ProjectionError> {
        let Some(content) = hit.projection.as_ref() else {
            return Ok(None);
        };

        let (rendered, preserve_identity) = match content {
            ContentRef::Text { text } => (text.clone(), false),
            ContentRef::Json { value } => (
                serde_json::to_string(value)
                    .map_err(|error| ProjectionError::Json(error.to_string()))?,
                false,
            ),
            ContentRef::Blob {
                uri,
                media_type,
                sha256,
                size_bytes,
            } => (
                render_blob_reference(uri, media_type, sha256, *size_bytes)?,
                true,
            ),
        };
        if rendered.trim().is_empty() {
            return Err(ProjectionError::EmptyContent {
                item_id: hit.item_id,
            });
        }
        if hit.provenance.trim().is_empty() || hit.method_id.trim().is_empty() {
            return Err(ProjectionError::EmptyProvenance {
                item_id: hit.item_id,
            });
        }

        let content = if preserve_identity && rendered.chars().count() > self.max_chars {
            return Err(ProjectionError::ProjectionTooLarge {
                item_id: hit.item_id,
                actual: rendered.chars().count(),
                maximum: self.max_chars,
            });
        } else {
            truncate_chars(&rendered, self.max_chars)
        };
        let evidence_refs = hit
            .evidence_refs
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| ProjectionError::Json(error.to_string()))?;
        let evidence_bytes = if evidence_refs.is_empty() {
            0
        } else {
            serde_json::to_vec(&evidence_refs)
                .map_err(|error| ProjectionError::Json(error.to_string()))?
                .len()
        };
        let token_cost = estimate_tokens_with_metadata(&content, evidence_bytes);
        let provenance = if hit.snapshot_id.as_deref().unwrap_or_default().is_empty() {
            format!("{}:{}", hit.method_id, hit.provenance)
        } else {
            format!(
                "{}:{}@{}",
                hit.method_id,
                hit.provenance,
                hit.snapshot_id.as_deref().unwrap_or_default()
            )
        };

        Ok(Some(MemoryProjection {
            projection_id: uuid::Uuid::new_v4(),
            source_memory_id: hit.item_id,
            source_version: hit.version,
            content,
            evidence_refs,
            provenance,
            token_cost,
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionError {
    InvalidLimit,
    EmptyContent {
        item_id: uuid::Uuid,
    },
    EmptyProvenance {
        item_id: uuid::Uuid,
    },
    ProjectionTooLarge {
        item_id: uuid::Uuid,
        actual: usize,
        maximum: usize,
    },
    Json(String),
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimit => write!(formatter, "projection character limit must be positive"),
            Self::EmptyContent { item_id } => {
                write!(formatter, "memory hit {item_id} has empty projection")
            }
            Self::EmptyProvenance { item_id } => {
                write!(formatter, "memory hit {item_id} has empty provenance")
            }
            Self::ProjectionTooLarge {
                item_id,
                actual,
                maximum,
            } => write!(
                formatter,
                "memory hit {item_id} reference projection has {actual} characters, maximum is {maximum}"
            ),
            Self::Json(error) => write!(formatter, "memory JSON projection failed: {error}"),
        }
    }
}

impl Error for ProjectionError {}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn estimate_tokens_with_metadata(content: &str, evidence_bytes: usize) -> u32 {
    let chars = content.chars().count().saturating_add(evidence_bytes);
    let tokens = chars.saturating_add(3) / 4;
    u32::try_from(tokens.max(1)).unwrap_or(u32::MAX)
}

/// Renders binary content as metadata rather than copying the artifact bytes
/// into the model request. JSON keeps URI and digest values unambiguous when
/// they contain whitespace or punctuation.
fn render_blob_reference(
    uri: &str,
    media_type: &str,
    sha256: &str,
    size_bytes: u64,
) -> Result<String, ProjectionError> {
    #[derive(serde::Serialize)]
    struct BlobReference<'a> {
        #[serde(rename = "type")]
        content_type: &'static str,
        uri: &'a str,
        media_type: &'a str,
        sha256: &'a str,
        size_bytes: u64,
    }

    serde_json::to_string(&BlobReference {
        content_type: "blob",
        uri,
        media_type,
        sha256,
        size_bytes,
    })
    .map_err(|error| ProjectionError::Json(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_memory::{EvidenceSpan, MemoryHit};
    use uuid::Uuid;

    fn hit(projection: Option<ContentRef>) -> MemoryHit {
        MemoryHit {
            item_id: Uuid::from_u128(7),
            source_record_ids: Vec::new(),
            version: 2,
            method_id: "text-memory".to_string(),
            rank: 0,
            score: Some(0.9),
            evidence: vec![EvidenceSpan::Text {
                start_char: 0,
                end_char: 3,
            }],
            evidence_refs: Vec::new(),
            projection,
            provenance: "record-7".to_string(),
            snapshot_id: Some("snapshot-1".to_string()),
            latency_ms: 2,
        }
    }

    #[test]
    fn renders_text_and_json_with_bounded_token_cost() {
        let renderer = ProjectionRenderer::new(64).unwrap();
        let rendered = renderer
            .render(&hit(Some(ContentRef::Text {
                text: "abcdef".to_string(),
            })))
            .unwrap()
            .unwrap();
        assert_eq!(rendered.content, "abcdef");
        assert_eq!(rendered.token_cost, 2);
        assert!(rendered.provenance.contains("snapshot-1"));

        let json = renderer
            .render(&hit(Some(ContentRef::Json {
                value: serde_json::json!({"ok": true}),
            })))
            .unwrap()
            .unwrap();
        assert_eq!(json.content, r#"{"ok":true}"#);

        let truncating = ProjectionRenderer::new(5).unwrap();
        assert_eq!(
            truncating
                .render(&hit(Some(ContentRef::Text {
                    text: "abcdef".to_string(),
                })))
                .unwrap()
                .unwrap()
                .content,
            "abcde"
        );
    }

    #[test]
    fn binary_content_becomes_a_bounded_artifact_reference() {
        let hit = hit(Some(ContentRef::Blob {
            uri: "cas://video".to_string(),
            media_type: "video/mp4".to_string(),
            sha256: "digest".to_string(),
            size_bytes: 10,
        }));
        let rendered = ProjectionRenderer::default()
            .render(&hit)
            .unwrap()
            .expect("blob content should produce a reference projection");

        assert!(rendered.content.chars().count() <= DEFAULT_MAX_CHARS);
        assert!(rendered.content.contains(r#""type":"blob""#));
        assert!(rendered.content.contains(r#""uri":"cas://video""#));
        assert!(rendered.content.contains(r#""media_type":"video/mp4""#));
        assert!(rendered.content.contains(r#""sha256":"digest""#));
        assert!(rendered.content.contains(r#""size_bytes":10"#));
        assert_eq!(rendered.provenance, "text-memory:record-7@snapshot-1");
    }

    #[test]
    fn binary_reference_is_never_truncated_into_invalid_json() {
        let hit = hit(Some(ContentRef::Blob {
            uri: "cas://video".to_string(),
            media_type: "video/mp4".to_string(),
            sha256: "digest".to_string(),
            size_bytes: 10,
        }));

        assert!(matches!(
            ProjectionRenderer::new(8).unwrap().render(&hit),
            Err(ProjectionError::ProjectionTooLarge { maximum: 8, .. })
        ));
    }
}
