//! Brave Search provider boundary.

mod request;
mod response;

pub(super) use request::{request, SearchRequest};
pub(super) use response::BraveSearchResponse;
