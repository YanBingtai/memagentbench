//! Focused model transport, projection, and protocol tests.
//!
//! Each submodule follows one responsibility so failures point directly at
//! the provider contract under test without changing production visibility.

mod client;
mod errors;
mod projection;
mod request;
mod streaming;
