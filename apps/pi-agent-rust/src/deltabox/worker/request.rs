use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{invalid_request, WORKER_ARG_BYTES};
use crate::deltabox::MAX_WORKER_REQUEST_BYTES;
use crate::ToolError;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerRequest {
    pub(crate) name: String,
    pub(crate) arguments: Value,
    pub(crate) max_file_bytes: usize,
    pub(crate) max_output_bytes: usize,
    pub(crate) lease: PathBuf,
    pub(crate) timeout_ms: u64,
}

pub(crate) fn parse_request(chunks: &[String]) -> Result<WorkerRequest, ToolError> {
    if chunks.iter().map(String::len).sum::<usize>() > MAX_WORKER_REQUEST_BYTES {
        return Err(invalid_request("DeltaBox worker request is too large"));
    }

    let request: WorkerRequest = serde_json::from_str(&chunks.concat()).map_err(invalid_request)?;
    if request.max_output_bytes == 0 || request.timeout_ms == 0 {
        return Err(invalid_request("DeltaBox worker limits must be positive"));
    }
    Ok(request)
}

pub(crate) fn chunk_request(mut request: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    while !request.is_empty() {
        let mut end = request.len().min(WORKER_ARG_BYTES);
        while !request.is_char_boundary(end) {
            end -= 1;
        }
        chunks.push(request[..end].to_string());
        request = &request[end..];
    }
    chunks
}
