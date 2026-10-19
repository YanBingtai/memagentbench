//! Context compaction hook contracts.

use std::{future::Future, pin::Pin, sync::Arc};

use crate::{events::RunContext, message::Message};

/// Why the agent asked a caller-provided compactor to transform context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextCompactionReason {
    ByteBudget,
}

/// Input supplied to the optional context compaction hook.
///
/// The returned messages are request-only context. They are not written to
/// the session automatically; the durable transcript remains the source of
/// truth for recovery and auditing.
#[derive(Debug, Clone)]
pub struct ContextCompactionRequest {
    pub context: RunContext,
    pub step: usize,
    pub reason: ContextCompactionReason,
    pub messages: Vec<Message>,
    pub max_context_messages: usize,
    pub max_context_bytes: usize,
}

pub type ContextCompactionFuture =
    Pin<Box<dyn Future<Output = Result<Vec<Message>, String>> + Send>>;

pub(crate) type ContextCompactionHook =
    Arc<dyn Fn(ContextCompactionRequest) -> ContextCompactionFuture + Send + Sync>;
