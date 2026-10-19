pub mod agent_loop;
pub mod config;
pub mod deltabox;
pub mod events;
pub mod message;
pub mod model;
pub mod session;
pub mod skills;
pub mod subagent;
pub mod tools;

/// Stable protocol surface shared by the model, agent loop, tools, and
/// session layers.
///
/// The implementations still live in the legacy modules for now. Keeping
/// this facade separate lets callers depend on message and event contracts
/// while those implementations are moved into the future `agent-protocol`
/// crate.
pub mod protocol {
    pub use super::events::{AgentEvent, MemoryTrace, RunContext, RunOutcome};
    pub use super::message::{
        FunctionCall, Message, ToolCall, ToolDefinition, ToolFunction, ToolResultMetadata,
    };
}

pub use agent_loop::{
    AfterRunFuture, AfterRunRequest, AgentError, AgentLoop, CancellationToken,
    ContextCompactionFuture, ContextCompactionReason, ContextCompactionRequest, EventSink,
    MemoryHookConfig, MemoryHookFuture, MemoryHookRequest, PendingRuntimeAction, RunResult,
    RuntimeReconcileOutcome,
};
pub use config::{Config, ConfigError, RunPolicy};
pub use events::{AgentEvent, RunContext, RunOutcome};
pub use message::{
    FunctionCall, Message, ToolCall, ToolDefinition, ToolFunction, ToolResultMetadata,
};
pub use model::{
    ChatFuture, ChatModel, ChatOptions, ChatRequest, ModelClient, ModelError, ToolCallDelta,
    ToolCallDeltaCallback, ToolChoice,
};
pub use session::{SessionError, SessionRecord, SessionStore};
pub use skills::{
    Skill, SkillCatalog, SkillDiagnostic, SkillDiagnosticError, SkillDocumentError, SkillError,
    SkillLoadReport, SkillLoader, SkillLoaderError,
};
pub use subagent::SubagentTool;
pub use tools::{
    EditFileTool, ListFilesTool, PreparedToolCall, ReadFileTool, RegistryError, SearchFilesTool,
    ShellTool, Tool, ToolCapabilities, ToolConcurrency, ToolContext, ToolError,
    ToolExecutionContext, ToolExposure, ToolFuture, ToolGateway, ToolRegistry, ToolReplayPolicy,
    ToolResult, ToolResultFuture, ToolUpdate, ToolUpdateSink, WebSearchTool, WriteFileTool,
};
