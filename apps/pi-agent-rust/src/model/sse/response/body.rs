use crate::model::error::ModelError;

/// Read a response body without allowing an unbounded provider payload.
///
/// The caller chooses a smaller limit for error responses. A known
/// `Content-Length` is rejected before reading, and chunked responses are
/// bounded as they arrive.
pub(crate) async fn read_bounded_body(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<String, ModelError> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(ModelError::ResponseLimit { limit });
    }

    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(ModelError::Http)? {
        if body.len().saturating_add(chunk.len()) > limit {
            return Err(ModelError::ResponseLimit { limit });
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).map_err(|_| ModelError::ResponseEncoding)
}
