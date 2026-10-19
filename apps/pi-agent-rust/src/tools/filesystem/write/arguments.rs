//! JSON arguments accepted by the `write_file` tool.

use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WriteFileArguments {
    pub(super) path: PathBuf,
    pub(super) content: String,
}
