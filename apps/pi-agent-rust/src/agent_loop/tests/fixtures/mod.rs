use super::super::*;

mod helpers;
mod http;
mod models;
mod tools;

pub(super) use helpers::*;
pub(super) use http::*;
pub(super) use models::*;
pub(super) use tools::*;

pub(crate) fn test_config(base_url: String) -> Config {
    Config {
        base_url,
        model: "test-model".to_string(),
        max_steps: 2,
        request_timeout_secs: 5,
        ..Config::default()
    }
}
