//! Constructors and compatibility helpers for built-in tools.

use super::super::{
    EditFileTool, ListFilesTool, ReadFileTool, ReadSkillTool, SearchFilesTool, ShellTool,
    ToolContext, WriteFileTool,
};
use crate::skills::SkillCatalog;

use super::{RegistryError, ToolRegistry};

impl ToolRegistry {
    pub fn with_read_file(context: ToolContext) -> Result<Self, RegistryError> {
        let mut registry = Self::new().with_output_limit(context.max_tool_output_bytes);
        registry.register(ReadFileTool::new(context))?;
        Ok(registry)
    }

    pub fn register_write_file(&mut self, context: ToolContext) -> Result<(), RegistryError> {
        self.register(WriteFileTool::new(context))
    }

    pub fn register_edit_file(&mut self, context: ToolContext) -> Result<(), RegistryError> {
        self.register(EditFileTool::new(context))
    }

    pub fn register_shell(&mut self, context: ToolContext) -> Result<(), RegistryError> {
        // This compatibility helper is intentionally explicit in its
        // implementation: library callers that need isolation should build a
        // `ShellTool` with `with_sandbox_mode` and register it directly. The
        // CLI does that and defaults to managed mode.
        self.register(ShellTool::new(context))
    }

    pub fn register_list_files(&mut self, context: ToolContext) -> Result<(), RegistryError> {
        self.register(ListFilesTool::new(context))
    }

    pub fn register_search_files(&mut self, context: ToolContext) -> Result<(), RegistryError> {
        self.register(SearchFilesTool::new(context))
    }

    /// Register a name-based reader for the supplied skills.
    ///
    /// The tool accepts only catalog names and never arbitrary filesystem
    /// paths, so a model cannot use it to escape the configured skill roots.
    pub fn register_skill_reader(
        &mut self,
        catalog: SkillCatalog,
        max_output_bytes: usize,
    ) -> Result<(), RegistryError> {
        self.register(ReadSkillTool::new(catalog, max_output_bytes))
    }
}
