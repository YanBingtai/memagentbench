//! Brave Search HTTP request construction and bounded response decoding.

use std::time::Duration;

use reqwest::Client;

use crate::agent_loop::CancellationToken;

use super::response::BraveSearchResponse;
use crate::tools::{limits::*, ToolError};

pub(crate) struct SearchRequest<'a> {
    pub(crate) client: &'a Client,
    pub(crate) endpoint: &'a str,
    pub(crate) api_key: &'a str,
    pub(crate) timeout: Duration,
    pub(crate) query: &'a str,
    pub(crate) count: usize,
    pub(crate) country: Option<&'a str>,
    pub(crate) search_lang: Option<&'a str>,
    pub(crate) freshness: Option<&'a str>,
    pub(crate) cancellation: &'a CancellationToken,
}

pub(crate) async fn request(options: SearchRequest<'_>) -> Result<BraveSearchResponse, ToolError> {
    let SearchRequest {
        client,
        endpoint,
        api_key,
        timeout,
        query,
        count,
        country,
        search_lang,
        freshness,
        cancellation,
    } = options;
    let mut query_params = vec![
        ("q", query.to_string()),
        ("count", count.to_string()),
        ("safesearch", "moderate".to_string()),
    ];
    if let Some(country) = country {
        query_params.push(("country", country.to_ascii_uppercase()));
    }
    if let Some(search_lang) = search_lang {
        query_params.push(("search_lang", search_lang.to_string()));
    }
    if let Some(freshness) = freshness {
        query_params.push(("freshness", freshness.to_string()));
    }

    let request = client
        .get(endpoint)
        .header("X-Subscription-Token", api_key)
        .header("Accept", "application/json")
        .query(&query_params)
        .timeout(timeout);
    let response = tokio::select! {
        result = request.send() => result.map_err(|error| ToolError::WebSearch {
            message: error.to_string(),
        })?,
        _ = cancellation.cancelled() => {
            return Err(ToolError::WebSearch {
                message: "search cancelled by the caller".to_string(),
            });
        }
    };

    if response
        .content_length()
        .is_some_and(|length| length > MAX_WEB_RESPONSE_BYTES as u64)
    {
        return Err(ToolError::WebSearch {
            message: format!("search response exceeds the {MAX_WEB_RESPONSE_BYTES}-byte limit"),
        });
    }
    let status = response.status();
    let body = tokio::select! {
        result = response.bytes() => result.map_err(|error| ToolError::WebSearch {
            message: error.to_string(),
        })?,
        _ = cancellation.cancelled() => {
            return Err(ToolError::WebSearch {
                message: "search cancelled by the caller".to_string(),
            });
        }
    };
    if body.len() > MAX_WEB_RESPONSE_BYTES {
        return Err(ToolError::WebSearch {
            message: format!("search response exceeds the {MAX_WEB_RESPONSE_BYTES}-byte limit"),
        });
    }
    if !status.is_success() {
        let message = String::from_utf8_lossy(&body);
        return Err(ToolError::WebSearch {
            message: format!(
                "HTTP {status}: {}",
                truncate_output(message.into_owned(), 512)
            ),
        });
    }

    let response: BraveSearchResponse =
        serde_json::from_slice(&body).map_err(|error| ToolError::WebSearch {
            message: format!("invalid provider response: {error}"),
        })?;
    Ok(response)
}
