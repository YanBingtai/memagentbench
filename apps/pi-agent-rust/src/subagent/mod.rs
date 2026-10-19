//! A bounded, read-only child-agent tool for delegated, isolated work.
//!
//! A child receives a fresh transcript and an explicitly cloned tool set. The
//! parent gets only a bounded summary, so intermediate child messages do not
//! pollute the parent session. The child is deliberately non-recursive and
//! read-only: delegation and every tool above `ReadOnly` are removed at
//! construction time.

mod arguments;
mod config;
mod execution;
mod progress;
#[cfg(test)]
mod tests;
mod tool;

#[cfg(test)]
use crate::tools::Tool;
use crate::{
    config::RunPolicy,
    model::{ChatOptions, ModelClient},
    skills::SkillCatalog,
    tools::ToolRegistry,
};

const DEFAULT_MAX_OUTPUT_BYTES: usize = 32 * 1024;

/// Executes one delegated task with a fresh AgentLoop.
pub struct SubagentTool {
    client: ModelClient,
    model: String,
    api_key: Option<String>,
    options: ChatOptions,
    tools: ToolRegistry,
    policy: RunPolicy,
    max_runtime_secs: u64,
    max_output_bytes: usize,
    skills: Option<SkillCatalog>,
}
