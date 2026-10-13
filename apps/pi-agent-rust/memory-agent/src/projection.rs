//! Convert provider hits into bounded, model-facing projections.
//!
//! Providers may return text, structured JSON, or references to large
//! artifacts. Only the first two can be assembled into the current model
//! request without another modality-specific renderer, so unsupported content
//! is skipped explicitly.

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

    /// Renders a hit when its provider supplied inline text or JSON.
    ///
    /// `Ok(None)` means the hit is valid but requires a modality-specific
    /// renderer (for example video or trajectory) before it can reach a text
    /// model. It is deliberately not treated as a provider error.
    pub fn render(&self, hit: &MemoryHit) -> Result<Option<MemoryProjection>, ProjectionError> {
        let Some(content) = hit.projection.as_ref() else {
            return Ok(None);
        };

        let rendered = match content {
            ContentRef::Text { text } => text.clone(),
            ContentRef::Json { value } => serde_json::to_string(value)
                .map_err(|error| ProjectionError::Json(error.to_string()))?,
            ContentRef::Blob { .. } => return Ok(None),
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

        let content = truncate_chars(&rendered, self.max_chars);
        let token_cost = estimate_tokens(&content);
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
            provenance,
            token_cost,
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionError {
    InvalidLimit,
    EmptyContent { item_id: uuid::Uuid },
    EmptyProvenance { item_id: uuid::Uuid },
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
            Self::Json(error) => write!(formatter, "memory JSON projection failed: {error}"),
        }
    }
}

impl Error for ProjectionError {}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn estimate_tokens(value: &str) -> u32 {
    let chars = value.chars().count();
    let tokens = chars.saturating_add(3) / 4;
    u32::try_from(tokens.max(1)).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_memory::{EvidenceSpan, MemoryHit};
    use uuid::Uuid;

    fn hit(projection: Option<ContentRef>) -> MemoryHit {
        MemoryHit {
            item_id: Uuid::from_u128(7),
            version: 2,
            method_id: "text-memory".to_string(),
            rank: 0,
            score: Some(0.9),
            evidence: vec![EvidenceSpan::Text {
                start_char: 0,
                end_char: 3,
            }],
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
    fn binary_content_is_skipped_until_a_modality_renderer_exists() {
        let hit = hit(Some(ContentRef::Blob {
            uri: "cas://video".to_string(),
            media_type: "video/mp4".to_string(),
            sha256: "digest".to_string(),
            size_bytes: 10,
        }));
        assert!(ProjectionRenderer::default()
            .render(&hit)
            .unwrap()
            .is_none());
    }
}
