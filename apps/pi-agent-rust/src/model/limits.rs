//! Bounded model transport and stream limits.

pub(crate) const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
pub(crate) const MAX_ERROR_BODY_BYTES: usize = 8 * 1024;
pub(crate) const MAX_RESPONSE_BODY_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_STREAM_EVENT_BYTES: usize = 64 * 1024;
pub(crate) const MAX_STREAM_CONTENT_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const MAX_STREAM_TOOL_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const MAX_STREAM_TOOL_COUNT: usize = 128;
pub(crate) const MAX_STREAM_TOOL_INDEX: usize = 1024;
