//! OpenAI-compatible HTTP client implementation.

mod chat_model;
mod completion;
mod configuration;
mod streaming;
mod usage;

pub use configuration::ModelClient;
