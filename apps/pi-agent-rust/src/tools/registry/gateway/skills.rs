//! Skill-reader setup for gateway consumers.

use crate::skills::SkillCatalog;
use crate::tools::errors::RegistryError;

use super::super::ToolGateway;

impl ToolGateway {
    /// Ensure the model can load full instructions for the supplied skills.
    ///
    /// The CLI may register this tool before constructing the agent. Library
    /// callers that only attach a catalog get the same capability without
    /// creating a duplicate registration.
    pub(crate) fn ensure_skill_reader(
        &mut self,
        catalog: SkillCatalog,
        max_output_bytes: usize,
    ) -> Result<(), RegistryError> {
        if self.registry.exposure("read_skill").is_none() {
            self.registry
                .register_skill_reader(catalog, max_output_bytes)?;
        }
        Ok(())
    }
}
