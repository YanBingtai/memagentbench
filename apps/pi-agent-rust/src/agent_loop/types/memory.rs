//! Memory hook contracts and static memory context limits.

use std::{future::Future, pin::Pin, sync::Arc};

use pi_agent_runtime::{MemoryHookContext, MemoryProposal, MemoryRevision, PolicyRevision};

use crate::message::Message;

/// Input supplied to a memory hook before one model request.
///
/// The hook receives an owned, already-truncated context so it can perform
/// asynchronous retrieval without borrowing the agent loop across an await.
#[derive(Debug, Clone)]
pub struct MemoryHookRequest {
    pub context: MemoryHookContext,
    pub messages: Vec<Message>,
}

pub type MemoryHookFuture =
    Pin<Box<dyn Future<Output = Result<Option<MemoryProposal>, String>> + Send>>;

pub(crate) type MemoryProposalHook =
    Arc<dyn Fn(MemoryHookRequest) -> MemoryHookFuture + Send + Sync>;

/// Static limits and revisions used to build each runtime memory context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryHookConfig {
    pub namespace: String,
    pub memory_revision: MemoryRevision,
    pub policy_revision: PolicyRevision,
    pub max_tokens: u32,
    pub max_candidates: u32,
    pub max_projection_bytes: u32,
}

impl Default for MemoryHookConfig {
    fn default() -> Self {
        Self {
            namespace: "default".to_owned(),
            memory_revision: MemoryRevision::INITIAL,
            policy_revision: PolicyRevision::INITIAL,
            max_tokens: 1_024,
            max_candidates: 16,
            max_projection_bytes: 32 * 1024,
        }
    }
}
