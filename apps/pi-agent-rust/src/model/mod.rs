//! Provider-neutral model API and OpenAI-compatible transport facade.

mod api;
mod client;
mod error;
mod limits;
mod options;
mod sse;

pub use api::{
    ChatFuture, ChatModel, ChatRequest, DeltaCallback, ModelUsage, ReasoningDeltaCallback,
    ToolCallDelta, ToolCallDeltaCallback,
};
pub use client::ModelClient;
pub use error::ModelError;
pub use options::{ChatOptions, ToolChoice};

#[cfg(test)]
mod tests;
