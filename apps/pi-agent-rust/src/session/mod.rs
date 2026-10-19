//! Durable JSONL session storage facade.

mod error;
mod jsonl;
mod limits;
mod lock;
mod repair;
mod store;
mod types;

pub use error::SessionError;
pub use types::{SessionRecord, SessionStore};

#[cfg(test)]
mod tests;
