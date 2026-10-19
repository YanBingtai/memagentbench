use uuid::Uuid;

use super::ToolExecutionContext;

impl ToolExecutionContext {
    /// Attach the identity of the parent agent call that owns this execution.
    ///
    /// Tools may use this for child-agent lineage and telemetry. It is
    /// metadata only; runtime admission and authorization remain outside the
    /// tool context.
    pub fn with_parent_identity(
        mut self,
        run_id: Uuid,
        turn_id: Uuid,
        step: usize,
        tool_call_id: impl Into<String>,
    ) -> Self {
        self.parent_run_id = Some(run_id);
        self.parent_turn_id = Some(turn_id);
        self.parent_step = Some(step);
        self.parent_tool_call_id = Some(tool_call_id.into());
        self
    }

    pub fn parent_run_id(&self) -> Option<Uuid> {
        self.parent_run_id
    }

    pub fn parent_turn_id(&self) -> Option<Uuid> {
        self.parent_turn_id
    }

    pub fn parent_step(&self) -> Option<usize> {
        self.parent_step
    }

    pub fn parent_tool_call_id(&self) -> Option<&str> {
        self.parent_tool_call_id.as_deref()
    }

    /// Set the current agent's delegation depth for child-agent lineage.
    pub fn with_delegation_depth(mut self, depth: u16) -> Self {
        self.delegation_depth = depth;
        self
    }

    pub fn delegation_depth(&self) -> u16 {
        self.delegation_depth
    }
}
