//! Bounded literal-text search inside the configured workspace.

mod definition;
mod execution;
mod format;
mod matcher;
mod walk;

use super::super::ToolContext;

/// Search UTF-8 text files inside the configured workspace.
#[derive(Debug, Clone)]
pub struct SearchFilesTool {
    context: ToolContext,
}

impl SearchFilesTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}
