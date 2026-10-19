//! Compatibility facade for bounded non-streaming model responses.
//!
//! Each concern lives in a small module while this facade keeps the original
//! `model::sse::response` API available to the model client.

mod body;
mod error;
mod normalize;
mod schema;
mod validation;

pub(crate) use body::read_bounded_body;
#[allow(unused_imports)]
pub(crate) use error::{api_error_message, truncate_for_error};
pub(crate) use normalize::normalize_inline_thinking;
pub(crate) use schema::{ChatCompletionResponse, Choice};
pub(crate) use validation::validate_non_streaming_message;
