use std::{
    sync::{Arc, Mutex as StdMutex},
    time::Duration,
};

use crate::{
    config::RunPolicy,
    model::{ChatModel, ChatOptions},
    tools::ToolRegistry,
};

use super::super::super::{validate_policy, AgentError, AgentLoop, ControlState};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub fn new(
        client: M,
        tools: ToolRegistry,
        model: impl Into<String>,
        api_key: Option<String>,
        options: ChatOptions,
        max_steps: usize,
    ) -> Result<Self, AgentError> {
        Self::new_with_policy(
            client,
            tools,
            model,
            api_key,
            options,
            RunPolicy {
                max_steps,
                ..RunPolicy::default()
            },
        )
    }

    /// Build an agent with explicit runtime limits and deadlines.
    pub fn new_with_policy(
        client: M,
        tools: ToolRegistry,
        model: impl Into<String>,
        api_key: Option<String>,
        options: ChatOptions,
        policy: RunPolicy,
    ) -> Result<Self, AgentError> {
        validate_policy(policy)?;

        Ok(Self {
            client,
            tools: tools.gateway(),
            model: model.into(),
            api_key,
            options,
            policy,
            allow_irreversible_tools: false,
            run_timeout: policy.run_timeout_secs.map(Duration::from_secs),
            session: None,
            runtime_ledger_file: None,
            runtime_recovery: false,
            system_prompt: None,
            skills: None,
            event_sink: None,
            before_tool_hook: None,
            after_tool_hook: None,
            memory_hook: None,
            context_compaction_hook: None,
            after_run_hook: None,
            control: Arc::new(StdMutex::new(ControlState::default())),
            delegation_depth: 0,
        })
    }
}
