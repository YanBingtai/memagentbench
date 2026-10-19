//! Brave Search response DTOs.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct BraveSearchResponse {
    pub(crate) web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct BraveWebResults {
    #[serde(default)]
    pub(crate) results: Vec<BraveSearchResult>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct BraveSearchResult {
    pub(crate) title: String,
    pub(crate) url: String,
    pub(crate) description: Option<String>,
}
