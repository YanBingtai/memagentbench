use std::{fs, path::Path, time::Duration};

use crate::ToolError;

use super::invalid_request;

pub(super) fn is_live(path: &Path) -> Result<bool, ToolError> {
    match fs::metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(invalid_request(format!(
            "cannot read cancellation lease: {error}"
        ))),
    }
}

pub(super) async fn wait_for_release(path: &Path) -> Result<(), ToolError> {
    loop {
        tokio::time::sleep(Duration::from_millis(10)).await;
        if !is_live(path)? {
            return Ok(());
        }
    }
}
