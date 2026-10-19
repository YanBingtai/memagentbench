//! Constructors and fluent configuration for [`AgentLoop`].
//!
//! The facade keeps the public builder surface in one place while the
//! implementations are grouped by responsibility: provider construction,
//! runtime options, and asynchronous hooks.

mod config;
mod hooks;
mod options;
