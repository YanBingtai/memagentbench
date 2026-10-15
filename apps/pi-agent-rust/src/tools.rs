use std::{
    cmp::Ordering as CompareOrdering,
    collections::{BTreeMap, BTreeSet},
    future::Future,
    path::{Component, Path, PathBuf},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use pi_agent_runtime::RiskClass;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use uuid::Uuid;

use crate::{
    agent_loop::CancellationToken,
    config::Config,
    message::{ToolCall, ToolDefinition},
    skills::SkillCatalog,
};

pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + 'a>>;
pub type ToolResultFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ToolResult, ToolError>> + Send + 'a>>;

/// An incremental update emitted while a tool is running.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolUpdate {
    pub content: String,
    #[serde(default)]
    pub details: Value,
}

impl ToolUpdate {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            details: Value::Null,
        }
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }

    /// Bound a progress update before it reaches an event sink.
    pub fn bounded(self, max_bytes: usize) -> Self {
        let result = ToolResult::text(self.content)
            .with_details(self.details)
            .bounded(max_bytes);
        Self {
            content: result.content,
            details: result.details,
        }
    }
}

pub type ToolUpdateSink = Arc<dyn Fn(ToolUpdate) + Send + Sync>;

/// Cooperative cancellation and progress reporting for one tool call.
#[derive(Clone)]
pub struct ToolExecutionContext {
    cancellation: CancellationToken,
    update_sink: Option<ToolUpdateSink>,
    max_update_bytes: usize,
    update_gate: Arc<UpdateGate>,
    parent_run_id: Option<Uuid>,
    parent_turn_id: Option<Uuid>,
    parent_step: Option<usize>,
    parent_tool_call_id: Option<String>,
    delegation_depth: u16,
}

const DEFAULT_TOOL_UPDATE_BYTES: usize = 65_536;

#[derive(Debug, Default)]
struct UpdateGate {
    open: AtomicBool,
}

struct UpdateLease {
    gate: Arc<UpdateGate>,
}

impl Drop for UpdateLease {
    fn drop(&mut self) {
        self.gate.open.store(false, Ordering::Release);
    }
}

impl ToolExecutionContext {
    pub fn new(cancellation: CancellationToken) -> Self {
        Self {
            cancellation,
            update_sink: None,
            max_update_bytes: DEFAULT_TOOL_UPDATE_BYTES,
            update_gate: Arc::new(UpdateGate {
                open: AtomicBool::new(true),
            }),
            parent_run_id: None,
            parent_turn_id: None,
            parent_step: None,
            parent_tool_call_id: None,
            delegation_depth: 0,
        }
    }

    /// Set the maximum content/details size for progress updates.
    pub fn with_output_limit(mut self, max_bytes: usize) -> Self {
        self.max_update_bytes = max_bytes;
        self
    }

    pub fn with_update_sink<F>(mut self, sink: F) -> Self
    where
        F: Fn(ToolUpdate) + Send + Sync + 'static,
    {
        self.update_sink = Some(Arc::new(sink));
        self
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Clone the run-wide cancellation signal for a delegated operation.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

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

    /// Close the progress channel after a tool reaches a terminal state.
    ///
    /// A tool may retain a cloned context for cleanup work. Once this method
    /// is called, those late updates are ignored instead of appearing after
    /// the corresponding `ToolFinished` event.
    pub fn close_updates(&self) {
        self.update_gate.open.store(false, Ordering::Release);
    }

    pub fn update(&self, update: ToolUpdate) {
        if !self.update_gate.open.load(Ordering::Acquire) {
            return;
        }
        if let Some(sink) = &self.update_sink {
            sink(update.bounded(self.max_update_bytes));
        }
    }

    fn lease_updates(&self) -> UpdateLease {
        UpdateLease {
            gate: Arc::clone(&self.update_gate),
        }
    }
}

/// Structured output from a tool execution.
///
/// `content` is the compatibility text sent back to the model. `details` is
/// kept for event consumers and runtime auditing, while `terminate` lets a
/// tool finish the current agent run without asking the model for another
/// turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub content: String,
    #[serde(default)]
    pub details: Value,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub terminate: bool,
}

impl ToolResult {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            details: Value::Null,
            is_error: false,
            terminate: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            is_error: true,
            ..Self::text(content)
        }
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }

    pub fn terminating(mut self) -> Self {
        self.terminate = true;
        self
    }

    /// Bound content and structured details before they cross the agent boundary.
    pub fn bounded(mut self, max_bytes: usize) -> Self {
        self.content = truncate_output(self.content, max_bytes);
        let details_bytes = serde_json::to_vec(&self.details)
            .map(|value| value.len())
            .unwrap_or(usize::MAX);
        if details_bytes > max_bytes {
            let preserved_status = self
                .details
                .get("status")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .filter(|status| {
                    matches!(
                        status.as_str(),
                        "cancelled"
                            | "timed_out"
                            | "unknown"
                            | "rejected"
                            | "truncated"
                            | "max_steps"
                            | "reconciled"
                    )
                });
            let summary = match preserved_status.as_deref() {
                Some(status) => json!({
                    "status": status,
                    "details_truncated": true,
                    "original_bytes": details_bytes,
                    "limit_bytes": max_bytes,
                }),
                None => json!({
                    "status": "details_truncated",
                    "original_bytes": details_bytes,
                    "limit_bytes": max_bytes,
                }),
            };
            self.details = if max_bytes > 0
                && serde_json::to_vec(&summary)
                    .map(|value| value.len() <= max_bytes)
                    .unwrap_or(false)
            {
                summary
            } else if let Some(status) = preserved_status.as_deref() {
                let minimal = json!({"status": status});
                if serde_json::to_vec(&minimal)
                    .map(|value| value.len() <= max_bytes)
                    .unwrap_or(false)
                {
                    minimal
                } else {
                    Value::Null
                }
            } else {
                Value::Null
            };
        }
        self
    }
}

/// Concurrency contract used by the agent scheduler.
///
/// Tools are exclusive by default because future tools may mutate files,
/// sessions, or external systems. A tool opts into parallel execution only
/// when its implementation is safe to run concurrently with another call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolConcurrency {
    Exclusive,
    Parallel,
}

/// Declares whether an interrupted invocation may be replayed automatically.
///
/// `Safe` is reserved for tools whose only observable effect is reading
/// already-existing state. Mutating tools remain `Never` unless their
/// implementation provides an idempotency contract that the runtime can
/// enforce during recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolReplayPolicy {
    Never,
    Safe,
}

/// Controls whether a registered tool is advertised to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolExposure {
    /// Include the tool in the model-visible function list.
    Direct,
    /// Keep the tool executable for trusted callers without advertising it.
    Hidden,
}

/// Immutable capability metadata for one registered tool.
///
/// The model-facing registry owns discovery and schema snapshots. This small
/// value is the execution-side contract that an agent loop or adapter can
/// inspect before it submits an action to the runtime. It deliberately does
/// not contain sandbox or runtime implementation details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolCapabilities {
    pub concurrency: ToolConcurrency,
    pub replay_policy: ToolReplayPolicy,
    pub exposure: ToolExposure,
    pub risk_class: RiskClass,
}

/// Shared limits and filesystem root available to built-in tools.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub workspace: PathBuf,
    pub max_file_bytes: usize,
    pub max_tool_output_bytes: usize,
}

const DEFAULT_LIST_MAX_DEPTH: usize = 2;
const DEFAULT_LIST_MAX_ENTRIES: usize = 200;
const DEFAULT_TOOL_OUTPUT_BYTES: usize = 65_536;
const MAX_LIST_DEPTH: usize = 32;
const MAX_LIST_ENTRIES: usize = 1_000;
const DEFAULT_SEARCH_MAX_RESULTS: usize = 100;
const MAX_SEARCH_RESULTS: usize = 1_000;
const MAX_SEARCH_FILES: usize = 10_000;
const MAX_SEARCH_LINE_BYTES: usize = 1_024;
const MAX_TOOL_ARGUMENT_BYTES: usize = 1_048_576;
const MAX_TOOL_ARGUMENT_DEPTH: usize = 64;
const MAX_SCHEMA_DEPTH: usize = 32;

impl ToolContext {
    pub fn from_config(config: &Config) -> Self {
        Self {
            workspace: config.workspace.clone(),
            max_file_bytes: config.max_file_bytes,
            max_tool_output_bytes: config.max_tool_output_bytes,
        }
    }
}

/// Errors produced while looking up, validating, or executing a tool.
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("invalid tool call {call_id}: {reason}")]
    InvalidCall { call_id: String, reason: String },

    #[error("unknown tool: {name}")]
    UnknownTool { name: String },

    #[error("tool is not available to the model: {name}")]
    ToolNotModelVisible { name: String },

    #[error("invalid arguments for tool {name}: {source}")]
    InvalidArguments {
        name: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid argument for tool {name}: {reason}")]
    InvalidArgumentValue { name: String, reason: String },

    #[error("arguments for tool {name} violate its schema at {path}: {reason}")]
    SchemaViolation {
        name: String,
        path: String,
        reason: String,
    },

    #[error("arguments for tool {name} exceed the {limit}-byte limit ({size} bytes)")]
    ArgumentsTooLarge {
        name: String,
        size: usize,
        limit: usize,
    },

    #[error("arguments for tool {name} exceed the depth limit of {limit}")]
    ArgumentsTooDeep { name: String, limit: usize },

    #[error("invalid range for tool {tool}: {reason}")]
    InvalidRange { tool: String, reason: String },

    #[error("subagent failed: {message}")]
    Subagent { message: String },

    #[error("invalid path for tool {tool}: {path}")]
    InvalidPath { tool: String, path: PathBuf },

    #[error("path escapes workspace for tool {tool}: {path}")]
    PathOutsideWorkspace { tool: String, path: PathBuf },

    #[error("failed to resolve workspace {path}: {source}")]
    ResolveWorkspace {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to resolve file {path}: {source}")]
    ResolveFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to inspect file {path}: {source}")]
    InspectFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("file is too large: {path} ({size} bytes, limit {limit} bytes)")]
    FileTooLarge {
        path: PathBuf,
        size: u64,
        limit: usize,
    },

    #[error("failed to read file {path}: {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to read directory {path}: {source}")]
    ReadDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to write file {path}: {source}")]
    WriteFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to replace file {path} from temporary file {temporary}: {source}")]
    RenameFile {
        path: PathBuf,
        temporary: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("file is not valid UTF-8: {path}")]
    InvalidUtf8 { path: PathBuf },

    #[error("edit target was not found in {path}")]
    EditTargetNotFound { path: PathBuf },

    #[error("edit target must not be empty in {path}")]
    EditTargetEmpty { path: PathBuf },

    #[error("edit target occurs {matches} times in {path}; expected exactly once")]
    EditTargetNotUnique { path: PathBuf, matches: usize },

    #[error("unknown skill: {name}")]
    UnknownSkill { name: String },

    #[error("skill output is too large: {name} ({size} bytes, limit {limit} bytes)")]
    SkillTooLarge {
        name: String,
        size: usize,
        limit: usize,
    },

    #[error("failed to format skill {name}: {message}")]
    SkillInvocation { name: String, message: String },
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("a tool named {name} is already registered")]
    DuplicateTool { name: String },

    #[error("invalid schema for tool {name}: {reason}")]
    InvalidDefinition { name: String, reason: String },
}

/// A dynamically dispatchable asynchronous tool.
pub trait Tool: Send + Sync {
    /// Return a stable, side-effect-free definition for the lifetime of the
    /// registered tool. The registry validates this definition at registration
    /// and providers receive the same logical schema on every request.
    fn definition(&self) -> ToolDefinition;
    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a>;

    /// Validate arguments before hooks, runtime admission, or execution.
    ///
    /// Built-in tools override this with typed deserialization and business
    /// rules. The registry applies the registered definition snapshot for the
    /// generic schema check, so custom tools do not need to re-read a dynamic
    /// definition here.
    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let _ = arguments;
        Ok(())
    }

    /// Execute with structured output while preserving the original string
    /// tool API for existing implementations.
    fn execute_result<'a>(&'a self, arguments: Value) -> ToolResultFuture<'a> {
        Box::pin(async move { self.execute(arguments).await.map(ToolResult::text) })
    }

    /// Execute with cooperative cancellation and progress reporting.
    ///
    /// Existing tools keep their original implementation automatically. A
    /// long-running tool can override this method and periodically inspect
    /// `context.is_cancelled()` or call `context.update(...)`.
    fn execute_result_with_context<'a>(
        &'a self,
        arguments: Value,
        _context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(async move { self.execute_result(arguments).await })
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    /// Declare the recovery policy for an interrupted invocation.
    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Never
    }

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Direct
    }

    /// Declares the side-effect class used by runtime admission.
    ///
    /// Custom tools are denied until they state their behavior explicitly.
    /// This keeps a newly registered tool from accidentally receiving the
    /// permissions intended for a read-only tool.
    fn risk_class(&self) -> RiskClass {
        RiskClass::Unknown
    }
}

/// A tool call whose JSON arguments have already been parsed and validated.
pub struct PreparedToolCall {
    tool: Arc<dyn Tool>,
    arguments: Value,
}

impl PreparedToolCall {
    pub fn arguments(&self) -> &Value {
        &self.arguments
    }
}

#[derive(Clone)]
struct RegisteredTool {
    tool: Arc<dyn Tool>,
    definition: ToolDefinition,
    capabilities: ToolCapabilities,
}

/// Registry used by the agent to advertise and execute tools.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: BTreeMap<String, RegisteredTool>,
    max_tool_output_bytes: usize,
}

/// Execution facade built from a [`ToolRegistry`].
///
/// Keeping this facade separate gives callers a clear boundary: the registry
/// advertises stable model definitions, while the gateway prepares and runs
/// already-validated calls. Runtime admission and sandbox execution remain
/// outside this module.
#[derive(Clone)]
pub struct ToolGateway {
    registry: ToolRegistry,
}

impl ToolGateway {
    pub fn new(registry: ToolRegistry) -> Self {
        Self { registry }
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    pub fn into_registry(self) -> ToolRegistry {
        self.registry
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.registry.definitions()
    }

    pub fn model_definitions(&self) -> Vec<ToolDefinition> {
        self.registry.model_definitions()
    }

    pub fn capabilities(&self, name: &str) -> Option<ToolCapabilities> {
        self.registry
            .tools
            .get(name)
            .map(|registered| registered.capabilities)
    }

    pub fn concurrency(&self, name: &str) -> Option<ToolConcurrency> {
        self.registry.concurrency(name)
    }

    pub fn replay_policy(&self, name: &str) -> Option<ToolReplayPolicy> {
        self.registry.replay_policy(name)
    }

    pub fn exposure(&self, name: &str) -> Option<ToolExposure> {
        self.registry.exposure(name)
    }

    pub fn risk_class(&self, name: &str) -> Option<RiskClass> {
        self.registry.risk_class(name)
    }

    /// Prepare a call produced by the model. Hidden tools are rejected here.
    pub fn prepare_model_call(
        &self,
        call: &ToolCall,
        arguments: Value,
    ) -> Result<PreparedToolCall, ToolError> {
        self.registry.prepare_model_call(call, arguments)
    }

    /// Prepare a trusted in-process call, including hidden capabilities.
    pub fn prepare_trusted_call(
        &self,
        call: &ToolCall,
        arguments: Value,
    ) -> Result<PreparedToolCall, ToolError> {
        self.registry.prepare_parsed_call(call, arguments)
    }

    pub async fn execute_prepared(
        &self,
        prepared: PreparedToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.registry.execute_prepared_result(prepared).await
    }

    pub async fn execute_prepared_result(
        &self,
        prepared: PreparedToolCall,
    ) -> Result<ToolResult, ToolError> {
        self.execute_prepared(prepared).await
    }

    pub async fn execute_prepared_with_context(
        &self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        self.registry
            .execute_prepared_result_with_context(prepared, context)
            .await
    }

    pub async fn execute_prepared_result_with_context(
        &self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        self.execute_prepared_with_context(prepared, context).await
    }

    pub fn normalize_result(&self, result: ToolResult) -> ToolResult {
        self.registry.normalize_result(result)
    }

    pub fn output_limit(&self) -> usize {
        self.registry.output_limit()
    }

    pub fn without(&self, name: &str) -> Self {
        Self::new(self.registry.without(name))
    }

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

impl Default for ToolRegistry {
    fn default() -> Self {
        Self {
            tools: BTreeMap::new(),
            max_tool_output_bytes: DEFAULT_TOOL_OUTPUT_BYTES,
        }
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the execution facade for this registry.
    pub fn gateway(&self) -> ToolGateway {
        ToolGateway::new(self.clone())
    }

    /// Set the maximum content/details size returned by registered tools.
    pub fn with_output_limit(mut self, max_bytes: usize) -> Self {
        self.max_tool_output_bytes = max_bytes;
        self
    }

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

    pub fn register_list_files(&mut self, context: ToolContext) -> Result<(), RegistryError> {
        self.register(ListFilesTool::new(context))
    }

    pub fn register_search_files(&mut self, context: ToolContext) -> Result<(), RegistryError> {
        self.register(SearchFilesTool::new(context))
    }

    pub fn register<T>(&mut self, tool: T) -> Result<(), RegistryError>
    where
        T: Tool + 'static,
    {
        let definition = tool.definition();
        if definition.definition_type != "function" {
            return Err(RegistryError::InvalidDefinition {
                name: definition.function.name.clone(),
                reason: "definition type must be function".to_string(),
            });
        }
        if let Err(reason) = validate_tool_name(&definition.function.name) {
            return Err(RegistryError::InvalidDefinition {
                name: definition.function.name.clone(),
                reason,
            });
        }
        validate_schema_definition(&definition).map_err(|reason| {
            RegistryError::InvalidDefinition {
                name: definition.function.name.clone(),
                reason,
            }
        })?;
        let name = definition.function.name.clone();
        if self.tools.contains_key(&name) {
            return Err(RegistryError::DuplicateTool { name });
        }
        let capabilities = ToolCapabilities {
            concurrency: tool.concurrency(),
            replay_policy: tool.replay_policy(),
            exposure: tool.exposure(),
            risk_class: tool.risk_class(),
        };

        self.tools.insert(
            name,
            RegisteredTool {
                tool: Arc::new(tool),
                definition,
                capabilities,
            },
        );
        Ok(())
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|registered| registered.definition.clone())
            .collect()
    }

    /// Return only tools that may be advertised in a model request.
    pub fn model_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .filter(|registered| registered.capabilities.exposure == ToolExposure::Direct)
            .map(|registered| registered.definition.clone())
            .collect()
    }

    /// Return a registry without one capability.
    ///
    /// This is used when constructing a child agent so delegation cannot
    /// recurse accidentally through a cloned parent registry.
    pub fn without(&self, name: &str) -> Self {
        let mut registry = self.clone();
        registry.tools.remove(name);
        registry
    }

    /// Return the concurrency contract for one registered tool.
    pub fn concurrency(&self, name: &str) -> Option<ToolConcurrency> {
        self.tools
            .get(name)
            .map(|registered| registered.capabilities.concurrency)
    }

    /// Return the recovery policy for one registered tool.
    pub fn replay_policy(&self, name: &str) -> Option<ToolReplayPolicy> {
        self.tools
            .get(name)
            .map(|registered| registered.capabilities.replay_policy)
    }

    pub fn exposure(&self, name: &str) -> Option<ToolExposure> {
        self.tools
            .get(name)
            .map(|registered| registered.capabilities.exposure)
    }

    /// Return the declared runtime risk for one registered tool.
    pub fn risk_class(&self, name: &str) -> Option<RiskClass> {
        self.tools
            .get(name)
            .map(|registered| registered.capabilities.risk_class)
    }

    pub async fn execute(&self, call: &ToolCall) -> Result<String, ToolError> {
        self.execute_result(call).await.map(|result| result.content)
    }

    pub async fn execute_result(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let prepared = self.prepare_call(call)?;
        self.execute_prepared_result(prepared).await
    }

    pub async fn execute_result_with_context(
        &self,
        call: &ToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        let prepared = self.prepare_call(call)?;
        self.execute_prepared_result_with_context(prepared, context)
            .await
    }

    pub async fn execute_prepared_result(
        &self,
        prepared: PreparedToolCall,
    ) -> Result<ToolResult, ToolError> {
        prepared
            .tool
            .execute_result(prepared.arguments)
            .await
            .map(|result| self.normalize_result(result))
    }

    pub async fn execute_prepared_result_with_context(
        &self,
        prepared: PreparedToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult, ToolError> {
        let _update_lease = context.lease_updates();
        let result = prepared
            .tool
            .execute_result_with_context(prepared.arguments, context)
            .await?;
        Ok(self.normalize_result(result))
    }

    /// Apply the registry-wide output bound after a tool or hook changes a result.
    pub fn normalize_result(&self, result: ToolResult) -> ToolResult {
        result.bounded(self.max_tool_output_bytes)
    }

    /// Return the registry-wide output limit used by tool results and updates.
    pub fn output_limit(&self) -> usize {
        self.max_tool_output_bytes
    }

    /// Validate one already-parsed call before hooks or runtime admission.
    pub fn validate_arguments(&self, call: &ToolCall, arguments: &Value) -> Result<(), ToolError> {
        self.prepare_parsed_call(call, arguments.clone())
            .map(|_| ())
    }

    /// Prepare a call after its JSON arguments have already been parsed.
    pub fn prepare_parsed_call(
        &self,
        call: &ToolCall,
        arguments: Value,
    ) -> Result<PreparedToolCall, ToolError> {
        validate_tool_call(call)?;
        if !arguments.is_object() {
            return Err(ToolError::InvalidCall {
                call_id: call.id.clone(),
                reason: "tool arguments must be a JSON object".to_string(),
            });
        }
        let registered =
            self.tools
                .get(&call.function.name)
                .ok_or_else(|| ToolError::UnknownTool {
                    name: call.function.name.clone(),
                })?;
        let tool = Arc::clone(&registered.tool);
        let argument_bytes = serde_json::to_vec(&arguments)
            .map_err(|source| ToolError::InvalidArguments {
                name: call.function.name.clone(),
                source,
            })?
            .len();
        if argument_bytes > MAX_TOOL_ARGUMENT_BYTES {
            return Err(ToolError::ArgumentsTooLarge {
                name: call.function.name.clone(),
                size: argument_bytes,
                limit: MAX_TOOL_ARGUMENT_BYTES,
            });
        }
        if argument_depth(&arguments, 0) > MAX_TOOL_ARGUMENT_DEPTH {
            return Err(ToolError::ArgumentsTooDeep {
                name: call.function.name.clone(),
                limit: MAX_TOOL_ARGUMENT_DEPTH,
            });
        }
        tool.validate_arguments(&arguments)?;
        validate_schema_arguments(&registered.definition, &arguments)?;
        Ok(PreparedToolCall { tool, arguments })
    }

    /// Prepare a call originating from the model.
    ///
    /// Trusted in-process callers may still use [`Self::prepare_parsed_call`]
    /// for hidden capabilities, but model output is restricted to the
    /// advertised tool surface.
    pub fn prepare_model_call(
        &self,
        call: &ToolCall,
        arguments: Value,
    ) -> Result<PreparedToolCall, ToolError> {
        if self.exposure(&call.function.name) == Some(ToolExposure::Hidden) {
            return Err(ToolError::ToolNotModelVisible {
                name: call.function.name.clone(),
            });
        }
        self.prepare_parsed_call(call, arguments)
    }

    fn prepare_call(&self, call: &ToolCall) -> Result<PreparedToolCall, ToolError> {
        validate_tool_call(call)?;
        if call.function.arguments.len() > MAX_TOOL_ARGUMENT_BYTES {
            return Err(ToolError::ArgumentsTooLarge {
                name: call.function.name.clone(),
                size: call.function.arguments.len(),
                limit: MAX_TOOL_ARGUMENT_BYTES,
            });
        }
        let arguments: Value =
            serde_json::from_str(&call.function.arguments).map_err(|source| {
                ToolError::InvalidArguments {
                    name: call.function.name.clone(),
                    source,
                }
            })?;
        self.prepare_parsed_call(call, arguments)
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

fn validate_tool_call(call: &ToolCall) -> Result<(), ToolError> {
    if call.id.trim().is_empty() {
        return Err(ToolError::InvalidCall {
            call_id: call.id.clone(),
            reason: "call id must not be empty".to_string(),
        });
    }
    if call.call_type != "function" {
        return Err(ToolError::InvalidCall {
            call_id: call.id.clone(),
            reason: format!("unsupported call type {:?}", call.call_type),
        });
    }
    if call.function.name.trim().is_empty() {
        return Err(ToolError::InvalidCall {
            call_id: call.id.clone(),
            reason: "function name must not be empty".to_string(),
        });
    }
    Ok(())
}

fn validate_tool_name(name: &str) -> Result<(), String> {
    if !(1..=64).contains(&name.len())
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        return Err(
            "function name must be 1-64 ASCII characters using only letters, numbers, '_' or '-'"
                .to_string(),
        );
    }
    Ok(())
}

#[derive(Debug)]
struct SchemaProblem {
    path: String,
    reason: String,
}

fn argument_depth(value: &Value, depth: usize) -> usize {
    match value {
        Value::Array(values) => values
            .iter()
            .map(|value| argument_depth(value, depth.saturating_add(1)))
            .max()
            .unwrap_or(depth),
        Value::Object(values) => values
            .values()
            .map(|value| argument_depth(value, depth.saturating_add(1)))
            .max()
            .unwrap_or(depth),
        _ => depth,
    }
}

/// Validate the intentionally small schema subset understood by the agent.
///
/// The provider still receives the original JSON Schema. Locally we enforce
/// the object/primitive fields used by built-in tools, together with required
/// fields, additionalProperties, enum, string length bounds, numeric bounds,
/// and array items.
/// Unsupported keywords are rejected at registration instead of being
/// silently ignored.
fn validate_schema_definition(definition: &ToolDefinition) -> Result<(), String> {
    validate_schema_node(&definition.function.parameters, "$.parameters", true, 0)
        .map_err(|problem| format!("{}: {}", problem.path, problem.reason))
}

fn validate_schema_node(
    schema: &Value,
    path: &str,
    root: bool,
    depth: usize,
) -> Result<(), SchemaProblem> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(SchemaProblem {
            path: path.to_string(),
            reason: format!("schema depth exceeds the limit of {MAX_SCHEMA_DEPTH}"),
        });
    }
    let object = schema.as_object().ok_or_else(|| SchemaProblem {
        path: path.to_string(),
        reason: "schema must be a JSON object".to_string(),
    })?;
    for keyword in object.keys() {
        if !matches!(
            keyword.as_str(),
            "type"
                | "properties"
                | "required"
                | "additionalProperties"
                | "enum"
                | "minimum"
                | "maximum"
                | "minLength"
                | "maxLength"
                | "items"
                | "description"
        ) {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: format!("unsupported schema keyword {keyword:?}"),
            });
        }
    }

    let type_name = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| SchemaProblem {
            path: path.to_string(),
            reason: "schema must declare a type".to_string(),
        })?;
    if root && type_name != "object" {
        return Err(SchemaProblem {
            path: path.to_string(),
            reason: "tool parameters must have type object".to_string(),
        });
    }
    if !matches!(
        type_name,
        "object" | "array" | "string" | "integer" | "number" | "boolean" | "null"
    ) {
        return Err(SchemaProblem {
            path: path.to_string(),
            reason: format!("unsupported schema type {type_name:?}"),
        });
    }

    if let Some(description) = object.get("description") {
        if !description.is_string() {
            return Err(SchemaProblem {
                path: format!("{path}.description"),
                reason: "description must be a string".to_string(),
            });
        }
    }

    if let Some(enum_values) = object.get("enum") {
        let enum_values = enum_values.as_array().ok_or_else(|| SchemaProblem {
            path: format!("{path}.enum"),
            reason: "enum must be a non-empty array".to_string(),
        })?;
        if enum_values.is_empty() {
            return Err(SchemaProblem {
                path: format!("{path}.enum"),
                reason: "enum must be a non-empty array".to_string(),
            });
        }
        for (index, value) in enum_values.iter().enumerate() {
            if !schema_type_matches_value(type_name, value) {
                return Err(SchemaProblem {
                    path: format!("{path}.enum[{index}]"),
                    reason: format!(
                        "enum value has type {}, expected {type_name}",
                        json_value_type(value)
                    ),
                });
            }
            if enum_values[..index].contains(value) {
                return Err(SchemaProblem {
                    path: format!("{path}.enum[{index}]"),
                    reason: "enum values must be unique".to_string(),
                });
            }
            if type_name == "string" {
                let length = value
                    .as_str()
                    .expect("string enum type was checked above")
                    .chars()
                    .count() as u64;
                if object
                    .get("minLength")
                    .and_then(Value::as_u64)
                    .is_some_and(|minimum| length < minimum)
                    || object
                        .get("maxLength")
                        .and_then(Value::as_u64)
                        .is_some_and(|maximum| length > maximum)
                {
                    return Err(SchemaProblem {
                        path: format!("{path}.enum[{index}]"),
                        reason: "enum value violates the string length bounds".to_string(),
                    });
                }
            }
        }
    }

    let properties = match object.get("properties") {
        Some(value) => Some(value.as_object().ok_or_else(|| SchemaProblem {
            path: format!("{path}.properties"),
            reason: "properties must be an object".to_string(),
        })?),
        None => None,
    };
    if properties.is_some() && type_name != "object" {
        return Err(SchemaProblem {
            path: path.to_string(),
            reason: "properties is only supported for object schemas".to_string(),
        });
    }
    if let Some(properties) = properties {
        for (name, property_schema) in properties {
            validate_schema_node(
                property_schema,
                &format!("{path}.properties.{name}"),
                false,
                depth.saturating_add(1),
            )?;
        }
    }

    if let Some(required) = object.get("required") {
        if type_name != "object" {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: "required is only supported for object schemas".to_string(),
            });
        }
        let required = required.as_array().ok_or_else(|| SchemaProblem {
            path: format!("{path}.required"),
            reason: "required must be an array of strings".to_string(),
        })?;
        let mut seen = BTreeSet::new();
        for item in required {
            let name = item.as_str().ok_or_else(|| SchemaProblem {
                path: format!("{path}.required"),
                reason: "required must contain only strings".to_string(),
            })?;
            if !seen.insert(name) {
                return Err(SchemaProblem {
                    path: format!("{path}.required"),
                    reason: format!("duplicate required property {name:?}"),
                });
            }
            if !properties.is_some_and(|properties| properties.contains_key(name)) {
                return Err(SchemaProblem {
                    path: format!("{path}.required"),
                    reason: format!("required property {name:?} is not declared"),
                });
            }
        }
    }

    if let Some(additional) = object.get("additionalProperties") {
        if type_name != "object" {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: "additionalProperties is only supported for object schemas".to_string(),
            });
        }
        if !additional.is_boolean() {
            return Err(SchemaProblem {
                path: format!("{path}.additionalProperties"),
                reason: "additionalProperties must be a boolean".to_string(),
            });
        }
    }

    if let Some(min_length) = object.get("minLength") {
        if type_name != "string" || min_length.as_u64().is_none() {
            return Err(SchemaProblem {
                path: format!("{path}.minLength"),
                reason: "minLength must be a non-negative integer on a string schema".to_string(),
            });
        }
    }
    if let Some(max_length) = object.get("maxLength") {
        if type_name != "string" || max_length.as_u64().is_none() {
            return Err(SchemaProblem {
                path: format!("{path}.maxLength"),
                reason: "maxLength must be a non-negative integer on a string schema".to_string(),
            });
        }
    }
    if let (Some(min_length), Some(max_length)) = (
        object.get("minLength").and_then(Value::as_u64),
        object.get("maxLength").and_then(Value::as_u64),
    ) {
        if min_length > max_length {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: "minLength must not exceed maxLength".to_string(),
            });
        }
    }

    for keyword in ["minimum", "maximum"] {
        if let Some(value) = object.get(keyword) {
            if !matches!(type_name, "integer" | "number") || !value.is_number() {
                return Err(SchemaProblem {
                    path: format!("{path}.{keyword}"),
                    reason: format!("{keyword} must be a number on a numeric schema"),
                });
            }
            if type_name == "integer" && value.as_i64().is_none() && value.as_u64().is_none() {
                return Err(SchemaProblem {
                    path: format!("{path}.{keyword}"),
                    reason: format!("{keyword} must be an integer on an integer schema"),
                });
            }
        }
    }
    if let (Some(minimum), Some(maximum)) = (object.get("minimum"), object.get("maximum")) {
        if minimum.as_f64() > maximum.as_f64() {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: "minimum must not exceed maximum".to_string(),
            });
        }
    }

    if let Some(items) = object.get("items") {
        if type_name != "array" {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: "items is only supported for array schemas".to_string(),
            });
        }
        validate_schema_node(
            items,
            &format!("{path}.items"),
            false,
            depth.saturating_add(1),
        )?;
    }

    Ok(())
}

fn validate_schema_arguments(
    definition: &ToolDefinition,
    arguments: &Value,
) -> Result<(), ToolError> {
    validate_schema_value(&definition.function.parameters, arguments, "arguments", 0).map_err(
        |problem| ToolError::SchemaViolation {
            name: definition.function.name.clone(),
            path: problem.path,
            reason: problem.reason,
        },
    )
}

fn validate_schema_value(
    schema: &Value,
    value: &Value,
    path: &str,
    depth: usize,
) -> Result<(), SchemaProblem> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(SchemaProblem {
            path: path.to_string(),
            reason: format!("schema depth exceeds the limit of {MAX_SCHEMA_DEPTH}"),
        });
    }
    let object = schema.as_object().ok_or_else(|| SchemaProblem {
        path: path.to_string(),
        reason: "schema must be a JSON object".to_string(),
    })?;
    let type_name = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| SchemaProblem {
            path: path.to_string(),
            reason: "schema must declare a type".to_string(),
        })?;
    let type_matches = match type_name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    };
    if !type_matches {
        return Err(SchemaProblem {
            path: path.to_string(),
            reason: format!("expected {type_name}, got {}", json_value_type(value)),
        });
    }
    if let Some(enum_values) = object.get("enum").and_then(Value::as_array) {
        if !enum_values.iter().any(|candidate| candidate == value) {
            return Err(SchemaProblem {
                path: path.to_string(),
                reason: "value is not one of the allowed enum values".to_string(),
            });
        }
    }

    match type_name {
        "object" => {
            let values = value.as_object().expect("object type was checked above");
            let properties = object
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            if let Some(required) = object.get("required").and_then(Value::as_array) {
                for required in required.iter().filter_map(Value::as_str) {
                    if !values.contains_key(required) {
                        return Err(SchemaProblem {
                            path: format!("{path}.{required}"),
                            reason: "required property is missing".to_string(),
                        });
                    }
                }
            }
            if object.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
                if let Some(unknown) = values.keys().find(|name| !properties.contains_key(*name)) {
                    return Err(SchemaProblem {
                        path: format!("{path}.{unknown}"),
                        reason: "additional property is not allowed".to_string(),
                    });
                }
            }
            for (name, property_schema) in properties {
                if let Some(property_value) = values.get(&name) {
                    validate_schema_value(
                        &property_schema,
                        property_value,
                        &format!("{path}.{name}"),
                        depth.saturating_add(1),
                    )?;
                }
            }
        }
        "array" => {
            if let Some(item_schema) = object.get("items") {
                for (index, item) in value
                    .as_array()
                    .expect("array type was checked above")
                    .iter()
                    .enumerate()
                {
                    validate_schema_value(
                        item_schema,
                        item,
                        &format!("{path}[{index}]"),
                        depth.saturating_add(1),
                    )?;
                }
            }
        }
        "string" => {
            let length = value
                .as_str()
                .expect("string type was checked above")
                .chars()
                .count() as u64;
            if let Some(min_length) = object.get("minLength").and_then(Value::as_u64) {
                if length < min_length {
                    return Err(SchemaProblem {
                        path: path.to_string(),
                        reason: format!("string length must be at least {min_length}"),
                    });
                }
            }
            if let Some(max_length) = object.get("maxLength").and_then(Value::as_u64) {
                if length > max_length {
                    return Err(SchemaProblem {
                        path: path.to_string(),
                        reason: format!("string length must be at most {max_length}"),
                    });
                }
            }
        }
        "integer" => {
            let integer = integer_value(value).expect("integer type was checked above");
            if let Some(minimum) = object.get("minimum").and_then(integer_value) {
                if compare_integers(integer, minimum) == CompareOrdering::Less {
                    return Err(SchemaProblem {
                        path: path.to_string(),
                        reason: format!("integer must be at least {minimum:?}"),
                    });
                }
            }
            if let Some(maximum) = object.get("maximum").and_then(integer_value) {
                if compare_integers(integer, maximum) == CompareOrdering::Greater {
                    return Err(SchemaProblem {
                        path: path.to_string(),
                        reason: format!("integer must be at most {maximum:?}"),
                    });
                }
            }
        }
        "number" => {
            let number = value.as_f64().expect("numeric type was checked above");
            if let Some(minimum) = object.get("minimum").and_then(Value::as_f64) {
                if number < minimum {
                    return Err(SchemaProblem {
                        path: path.to_string(),
                        reason: format!("number must be at least {minimum}"),
                    });
                }
            }
            if let Some(maximum) = object.get("maximum").and_then(Value::as_f64) {
                if number > maximum {
                    return Err(SchemaProblem {
                        path: path.to_string(),
                        reason: format!("number must be at most {maximum}"),
                    });
                }
            }
        }
        "boolean" | "null" => {}
        _ => unreachable!("schema definitions reject unsupported types"),
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegerValue {
    Signed(i64),
    Unsigned(u64),
}

fn integer_value(value: &Value) -> Option<IntegerValue> {
    value
        .as_i64()
        .map(IntegerValue::Signed)
        .or_else(|| value.as_u64().map(IntegerValue::Unsigned))
}

fn compare_integers(left: IntegerValue, right: IntegerValue) -> CompareOrdering {
    match (left, right) {
        (IntegerValue::Signed(left), IntegerValue::Signed(right)) => left.cmp(&right),
        (IntegerValue::Unsigned(left), IntegerValue::Unsigned(right)) => left.cmp(&right),
        (IntegerValue::Signed(left), IntegerValue::Unsigned(right)) => {
            if left < 0 {
                CompareOrdering::Less
            } else {
                (left as u64).cmp(&right)
            }
        }
        (IntegerValue::Unsigned(left), IntegerValue::Signed(right)) => {
            if right < 0 {
                CompareOrdering::Greater
            } else {
                left.cmp(&(right as u64))
            }
        }
    }
}

fn json_value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn schema_type_matches_value(type_name: &str, value: &Value) -> bool {
    match type_name {
        "integer" => integer_value(value).is_some(),
        "number" => value.is_number(),
        _ => json_value_type(value) == type_name,
    }
}

#[derive(Debug, Clone)]
pub struct ReadFileTool {
    context: ToolContext,
}

impl ReadFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "read_file",
            "Read a UTF-8 text file inside the configured workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path of the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-indexed line number to start reading from"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum number of lines to return"
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        )
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Safe
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let parsed: ReadFileArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "read_file".to_string(),
                    source,
                }
            })?;
        if parsed.offset == Some(0) {
            return Err(ToolError::InvalidRange {
                tool: "read_file".to_string(),
                reason: "offset must be greater than zero".to_string(),
            });
        }
        if parsed.limit == Some(0) {
            return Err(ToolError::InvalidRange {
                tool: "read_file".to_string(),
                reason: "limit must be greater than zero".to_string(),
            });
        }
        Ok(())
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: ReadFileArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "read_file".to_string(),
                        source,
                    }
                })?;
            let path =
                resolve_workspace_file(&self.context.workspace, &arguments.path, "read_file")
                    .await?;
            let metadata =
                tokio::fs::metadata(&path)
                    .await
                    .map_err(|source| ToolError::InspectFile {
                        path: path.clone(),
                        source,
                    })?;

            if !metadata.is_file() {
                return Err(ToolError::InvalidPath {
                    tool: "read_file".to_string(),
                    path,
                });
            }
            if metadata.len() > self.context.max_file_bytes as u64 {
                return Err(ToolError::FileTooLarge {
                    path,
                    size: metadata.len(),
                    limit: self.context.max_file_bytes,
                });
            }

            let bytes = tokio::fs::read(&path)
                .await
                .map_err(|source| ToolError::ReadFile {
                    path: path.clone(),
                    source,
                })?;
            let text = String::from_utf8(bytes)
                .map_err(|_| ToolError::InvalidUtf8 { path: path.clone() })?;
            let lines: Vec<&str> = text.split('\n').collect();
            let start = arguments.offset.unwrap_or(1);
            if start == 0 || start > lines.len() {
                return Err(ToolError::InvalidRange {
                    tool: "read_file".to_string(),
                    reason: format!(
                        "offset {start} is outside the file, which contains {} lines",
                        lines.len()
                    ),
                });
            }
            let start_index = start - 1;
            let end_index = match arguments.limit {
                Some(0) => {
                    return Err(ToolError::InvalidRange {
                        tool: "read_file".to_string(),
                        reason: "limit must be greater than zero".to_string(),
                    });
                }
                Some(limit) => start_index.saturating_add(limit).min(lines.len()),
                None => lines.len(),
            };
            let selected = lines[start_index..end_index].join("\n");

            Ok(truncate_output(
                selected,
                self.context.max_tool_output_bytes,
            ))
        })
    }
}

/// List workspace entries with bounded depth and output.
#[derive(Debug, Clone)]
pub struct ListFilesTool {
    context: ToolContext,
}

impl ListFilesTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for ListFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "list_files",
            "List files and directories inside the configured workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory; defaults to the workspace root"
                    },
                    "max_depth": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": MAX_LIST_DEPTH,
                        "description": "Number of directory levels below path to include"
                    },
                    "max_entries": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_LIST_ENTRIES,
                        "description": "Maximum number of entries to return"
                    }
                },
                "additionalProperties": false
            }),
        )
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let parsed: ListFilesArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "list_files".to_string(),
                    source,
                }
            })?;
        validate_list_limits(parsed.max_depth, parsed.max_entries)
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Safe
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: ListFilesArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "list_files".to_string(),
                        source,
                    }
                })?;
            let max_depth = arguments.max_depth.unwrap_or(DEFAULT_LIST_MAX_DEPTH);
            let max_entries = arguments.max_entries.unwrap_or(DEFAULT_LIST_MAX_ENTRIES);
            validate_list_limits(Some(max_depth), Some(max_entries))?;

            let requested = arguments.path.unwrap_or_else(|| PathBuf::from("."));
            let root =
                resolve_workspace_directory(&self.context.workspace, &requested, "list_files")
                    .await?;
            let prefix = display_relative_path(&requested);
            let (entries, truncated) =
                collect_directory_entries(&root, &root, &prefix, max_depth, max_entries).await?;

            let mut output = if entries.is_empty() {
                "(empty)".to_string()
            } else {
                entries.join("\n")
            };
            if truncated {
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(&format!("[truncated after {max_entries} entries]"));
            }
            Ok(truncate_output(output, self.context.max_tool_output_bytes))
        })
    }
}

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

impl Tool for SearchFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "search_files",
            "Search UTF-8 text files for a literal substring inside the configured workspace.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Literal, case-sensitive substring to find"
                    },
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory; defaults to the workspace root"
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_RESULTS,
                        "description": "Maximum matching lines to return"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let parsed: SearchFilesArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "search_files".to_string(),
                    source,
                }
            })?;
        if parsed.query.is_empty() {
            return Err(ToolError::InvalidArgumentValue {
                name: "search_files".to_string(),
                reason: "query must not be empty".to_string(),
            });
        }
        if parsed
            .max_results
            .is_some_and(|results| results == 0 || results > MAX_SEARCH_RESULTS)
        {
            return Err(ToolError::InvalidArgumentValue {
                name: "search_files".to_string(),
                reason: format!("max_results must be between 1 and {MAX_SEARCH_RESULTS}"),
            });
        }
        Ok(())
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Safe
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: SearchFilesArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "search_files".to_string(),
                        source,
                    }
                })?;
            let max_results = arguments.max_results.unwrap_or(DEFAULT_SEARCH_MAX_RESULTS);
            let requested = arguments.path.unwrap_or_else(|| PathBuf::from("."));
            let root =
                resolve_workspace_directory(&self.context.workspace, &requested, "search_files")
                    .await?;
            let prefix = display_relative_path(&requested);
            let report = search_directory(
                &root,
                &root,
                &prefix,
                &arguments.query,
                max_results,
                self.context.max_file_bytes,
            )
            .await?;

            let mut output = if report.matches.is_empty() {
                "(no matches)".to_string()
            } else {
                report.matches.join("\n")
            };
            if report.skipped_files > 0 {
                output.push_str(&format!(
                    "\n[skipped {} unreadable, binary, or oversized files]",
                    report.skipped_files
                ));
            }
            if report.truncated {
                if report.file_limit_reached {
                    output.push_str(&format!(
                        "\n[search truncated after scanning {MAX_SEARCH_FILES} files]"
                    ));
                } else {
                    output.push_str(&format!(
                        "\n[search truncated after {max_results} matching lines]"
                    ));
                }
            }
            Ok(truncate_output(output, self.context.max_tool_output_bytes))
        })
    }
}

/// Atomically write UTF-8 text inside the configured workspace.
#[derive(Debug, Clone)]
pub struct WriteFileTool {
    context: ToolContext,
}

impl WriteFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "write_file",
            "Atomically write UTF-8 text to a file inside the configured workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path of the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 file contents"
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        )
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let parsed: WriteFileArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "write_file".to_string(),
                    source,
                }
            })?;
        let size = parsed.content.len() as u64;
        if size > self.context.max_file_bytes as u64 {
            return Err(ToolError::FileTooLarge {
                path: parsed.path,
                size,
                limit: self.context.max_file_bytes,
            });
        }
        Ok(())
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::Reversible
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: WriteFileArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "write_file".to_string(),
                        source,
                    }
                })?;
            let size = arguments.content.len() as u64;
            if size > self.context.max_file_bytes as u64 {
                return Err(ToolError::FileTooLarge {
                    path: arguments.path.clone(),
                    size,
                    limit: self.context.max_file_bytes,
                });
            }

            atomic_write_workspace_file(
                &self.context.workspace,
                &arguments.path,
                arguments.content.as_bytes(),
            )
            .await?;

            Ok(format!(
                "Wrote {} bytes to {}",
                size,
                arguments.path.display()
            ))
        })
    }
}

/// Replace one exact text span in a UTF-8 workspace file.
#[derive(Debug, Clone)]
pub struct EditFileTool {
    context: ToolContext,
}

impl EditFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "edit_file",
            "Replace exactly one matching text span in a UTF-8 workspace file.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path of the file to edit"
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text that must occur once"
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text"
                    }
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": false
            }),
        )
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        let parsed: EditFileArguments =
            serde_json::from_value(arguments.clone()).map_err(|source| {
                ToolError::InvalidArguments {
                    name: "edit_file".to_string(),
                    source,
                }
            })?;
        if parsed.old_text.is_empty() {
            return Err(ToolError::EditTargetEmpty { path: parsed.path });
        }
        Ok(())
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::Reversible
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: EditFileArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "edit_file".to_string(),
                        source,
                    }
                })?;
            if arguments.old_text.is_empty() {
                return Err(ToolError::EditTargetEmpty {
                    path: arguments.path,
                });
            }

            let path =
                resolve_workspace_file(&self.context.workspace, &arguments.path, "edit_file")
                    .await?;
            let metadata =
                tokio::fs::metadata(&path)
                    .await
                    .map_err(|source| ToolError::InspectFile {
                        path: path.clone(),
                        source,
                    })?;
            if !metadata.is_file() {
                return Err(ToolError::InvalidPath {
                    tool: "edit_file".to_string(),
                    path,
                });
            }
            if metadata.len() > self.context.max_file_bytes as u64 {
                return Err(ToolError::FileTooLarge {
                    path,
                    size: metadata.len(),
                    limit: self.context.max_file_bytes,
                });
            }
            let bytes = tokio::fs::read(&path)
                .await
                .map_err(|source| ToolError::ReadFile {
                    path: path.clone(),
                    source,
                })?;
            let text = String::from_utf8(bytes)
                .map_err(|_| ToolError::InvalidUtf8 { path: path.clone() })?;
            let matches = text.match_indices(&arguments.old_text).count();
            match matches {
                0 => return Err(ToolError::EditTargetNotFound { path }),
                1 => {}
                matches => return Err(ToolError::EditTargetNotUnique { path, matches }),
            }

            let updated = text.replacen(&arguments.old_text, &arguments.new_text, 1);
            if updated.len() > self.context.max_file_bytes {
                return Err(ToolError::FileTooLarge {
                    path: arguments.path,
                    size: updated.len() as u64,
                    limit: self.context.max_file_bytes,
                });
            }
            atomic_write_workspace_file(
                &self.context.workspace,
                &arguments.path,
                updated.as_bytes(),
            )
            .await?;

            Ok(format!("Edited {}", arguments.path.display()))
        })
    }
}

/// Read the complete instructions for a registered skill by exact name.
#[derive(Debug, Clone)]
pub struct ReadSkillTool {
    catalog: SkillCatalog,
    max_output_bytes: usize,
}

impl ReadSkillTool {
    pub fn new(catalog: SkillCatalog, max_output_bytes: usize) -> Self {
        Self {
            catalog,
            max_output_bytes,
        }
    }
}

impl Tool for ReadSkillTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "read_skill",
            "Read the full instructions for a registered skill by its exact name.",
            json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact skill name from the available skills list",
                        "minLength": 1
                    }
                },
                "required": ["name"],
                "additionalProperties": false
            }),
        )
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Safe
    }

    fn validate_arguments(&self, arguments: &Value) -> Result<(), ToolError> {
        serde_json::from_value::<ReadSkillArguments>(arguments.clone())
            .map_err(|source| ToolError::InvalidArguments {
                name: "read_skill".to_string(),
                source,
            })
            .and_then(|parsed| {
                if parsed.name.trim().is_empty() {
                    return Err(ToolError::InvalidArgumentValue {
                        name: "read_skill".to_string(),
                        reason: "name must not be empty".to_string(),
                    });
                }
                Ok(())
            })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: ReadSkillArguments =
                serde_json::from_value(arguments).map_err(|source| {
                    ToolError::InvalidArguments {
                        name: "read_skill".to_string(),
                        source,
                    }
                })?;
            if self.catalog.get(&arguments.name).is_none() {
                return Err(ToolError::UnknownSkill {
                    name: arguments.name,
                });
            }

            let content = self
                .catalog
                .format_invocation(&arguments.name, None)
                .map_err(|error| ToolError::SkillInvocation {
                    name: arguments.name.clone(),
                    message: error.to_string(),
                })?;
            if content.len() > self.max_output_bytes {
                return Err(ToolError::SkillTooLarge {
                    name: arguments.name,
                    size: content.len(),
                    limit: self.max_output_bytes,
                });
            }

            Ok(content)
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileArguments {
    path: PathBuf,
    offset: Option<usize>,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ListFilesArguments {
    path: Option<PathBuf>,
    max_depth: Option<usize>,
    max_entries: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchFilesArguments {
    query: String,
    path: Option<PathBuf>,
    max_results: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteFileArguments {
    path: PathBuf,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EditFileArguments {
    path: PathBuf,
    old_text: String,
    new_text: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadSkillArguments {
    name: String,
}

fn validate_list_limits(
    max_depth: Option<usize>,
    max_entries: Option<usize>,
) -> Result<(), ToolError> {
    if max_depth.is_some_and(|depth| depth > MAX_LIST_DEPTH) {
        return Err(ToolError::InvalidArgumentValue {
            name: "list_files".to_string(),
            reason: format!("max_depth must be at most {MAX_LIST_DEPTH}"),
        });
    }
    if max_entries.is_some_and(|entries| entries == 0 || entries > MAX_LIST_ENTRIES) {
        return Err(ToolError::InvalidArgumentValue {
            name: "list_files".to_string(),
            reason: format!("max_entries must be between 1 and {MAX_LIST_ENTRIES}"),
        });
    }
    Ok(())
}

fn display_relative_path(path: &Path) -> PathBuf {
    path.components()
        .filter(|component| !matches!(component, Component::CurDir))
        .fold(PathBuf::new(), |mut result, component| {
            result.push(component.as_os_str());
            result
        })
}

#[derive(Debug, Default)]
struct SearchReport {
    matches: Vec<String>,
    skipped_files: usize,
    truncated: bool,
    file_limit_reached: bool,
}

async fn resolve_workspace_file(
    workspace: &Path,
    requested: &Path,
    tool: &str,
) -> Result<PathBuf, ToolError> {
    if requested.as_os_str().is_empty()
        || requested.is_absolute()
        || requested
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(ToolError::InvalidPath {
            tool: tool.to_string(),
            path: requested.to_path_buf(),
        });
    }

    let workspace =
        tokio::fs::canonicalize(workspace)
            .await
            .map_err(|source| ToolError::ResolveWorkspace {
                path: workspace.to_path_buf(),
                source,
            })?;
    let candidate = workspace.join(requested);
    let canonical = tokio::fs::canonicalize(&candidate)
        .await
        .map_err(|source| ToolError::ResolveFile {
            path: candidate.clone(),
            source,
        })?;

    if !canonical.starts_with(&workspace) {
        return Err(ToolError::PathOutsideWorkspace {
            tool: tool.to_string(),
            path: requested.to_path_buf(),
        });
    }

    Ok(canonical)
}

async fn resolve_workspace_directory(
    workspace: &Path,
    requested: &Path,
    tool: &str,
) -> Result<PathBuf, ToolError> {
    if requested.is_absolute()
        || requested
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(ToolError::InvalidPath {
            tool: tool.to_string(),
            path: requested.to_path_buf(),
        });
    }

    let workspace =
        tokio::fs::canonicalize(workspace)
            .await
            .map_err(|source| ToolError::ResolveWorkspace {
                path: workspace.to_path_buf(),
                source,
            })?;
    let candidate = if requested.as_os_str().is_empty() {
        workspace.clone()
    } else {
        workspace.join(requested)
    };
    let canonical = tokio::fs::canonicalize(&candidate)
        .await
        .map_err(|source| ToolError::ResolveFile {
            path: candidate.clone(),
            source,
        })?;
    if !canonical.starts_with(&workspace) {
        return Err(ToolError::PathOutsideWorkspace {
            tool: tool.to_string(),
            path: requested.to_path_buf(),
        });
    }
    let metadata =
        tokio::fs::metadata(&canonical)
            .await
            .map_err(|source| ToolError::InspectFile {
                path: canonical.clone(),
                source,
            })?;
    if !metadata.is_dir() {
        return Err(ToolError::InvalidPath {
            tool: tool.to_string(),
            path: requested.to_path_buf(),
        });
    }
    Ok(canonical)
}

async fn collect_directory_entries(
    workspace: &Path,
    root: &Path,
    prefix: &Path,
    max_depth: usize,
    max_entries: usize,
) -> Result<(Vec<String>, bool), ToolError> {
    let mut pending = vec![(root.to_path_buf(), PathBuf::new(), 0usize)];
    let mut entries = Vec::with_capacity(max_entries.min(64));

    while let Some((directory, relative_directory, depth)) = pending.pop() {
        let mut reader =
            tokio::fs::read_dir(&directory)
                .await
                .map_err(|source| ToolError::ReadDirectory {
                    path: directory.clone(),
                    source,
                })?;
        let mut children = Vec::new();
        while let Some(entry) =
            reader
                .next_entry()
                .await
                .map_err(|source| ToolError::ReadDirectory {
                    path: directory.clone(),
                    source,
                })?
        {
            children.push(entry);
        }
        children.sort_by_key(|entry| entry.file_name());

        let mut child_directories = Vec::new();
        for entry in children {
            if entries.len() == max_entries {
                return Ok((entries, true));
            }

            let name = entry.file_name();
            let relative = if relative_directory.as_os_str().is_empty() {
                PathBuf::from(&name)
            } else {
                relative_directory.join(&name)
            };
            let file_type = entry
                .file_type()
                .await
                .map_err(|source| ToolError::InspectFile {
                    path: entry.path(),
                    source,
                })?;
            let is_directory = file_type.is_dir();
            let mut display = if prefix.as_os_str().is_empty() {
                relative.clone()
            } else {
                prefix.join(&relative)
            }
            .to_string_lossy()
            .into_owned();
            if is_directory {
                display.push('/');
            }
            entries.push(display);

            if is_directory && depth < max_depth {
                let canonical = tokio::fs::canonicalize(entry.path())
                    .await
                    .map_err(|source| ToolError::InspectFile {
                        path: entry.path(),
                        source,
                    })?;
                if !canonical.starts_with(workspace) {
                    return Err(ToolError::PathOutsideWorkspace {
                        tool: "list_files".to_string(),
                        path: entry.path(),
                    });
                }
                child_directories.push((canonical, relative, depth + 1));
            }
        }

        for child in child_directories.into_iter().rev() {
            pending.push(child);
        }
    }

    Ok((entries, false))
}

async fn search_directory(
    workspace: &Path,
    root: &Path,
    prefix: &Path,
    query: &str,
    max_results: usize,
    max_file_bytes: usize,
) -> Result<SearchReport, ToolError> {
    let mut pending = vec![(root.to_path_buf(), PathBuf::new())];
    let mut report = SearchReport::default();
    let mut scanned_files = 0usize;

    while let Some((directory, relative_directory)) = pending.pop() {
        let mut reader =
            tokio::fs::read_dir(&directory)
                .await
                .map_err(|source| ToolError::ReadDirectory {
                    path: directory.clone(),
                    source,
                })?;
        let mut children = Vec::new();
        while let Some(entry) =
            reader
                .next_entry()
                .await
                .map_err(|source| ToolError::ReadDirectory {
                    path: directory.clone(),
                    source,
                })?
        {
            children.push(entry);
        }
        children.sort_by_key(|entry| entry.file_name());

        let mut child_directories = Vec::new();
        for entry in children {
            let name = entry.file_name();
            let relative = if relative_directory.as_os_str().is_empty() {
                PathBuf::from(&name)
            } else {
                relative_directory.join(&name)
            };
            let file_type = entry
                .file_type()
                .await
                .map_err(|source| ToolError::InspectFile {
                    path: entry.path(),
                    source,
                })?;

            if file_type.is_dir() {
                if should_skip_search_directory(&name) {
                    continue;
                }
                let canonical = tokio::fs::canonicalize(entry.path())
                    .await
                    .map_err(|source| ToolError::InspectFile {
                        path: entry.path(),
                        source,
                    })?;
                if !canonical.starts_with(workspace) {
                    return Err(ToolError::PathOutsideWorkspace {
                        tool: "search_files".to_string(),
                        path: entry.path(),
                    });
                }
                child_directories.push((canonical, relative));
                continue;
            }
            if !file_type.is_file() {
                continue;
            }

            if scanned_files == MAX_SEARCH_FILES {
                report.truncated = true;
                report.file_limit_reached = true;
                return Ok(report);
            }
            scanned_files += 1;

            let path = entry.path();
            let metadata = match tokio::fs::metadata(&path).await {
                Ok(metadata) => metadata,
                Err(_) => {
                    report.skipped_files += 1;
                    continue;
                }
            };
            if metadata.len() > max_file_bytes as u64 {
                report.skipped_files += 1;
                continue;
            }
            let text = match tokio::fs::read_to_string(&path).await {
                Ok(text) => text,
                Err(_) => {
                    report.skipped_files += 1;
                    continue;
                }
            };
            for (line_index, line) in text.lines().enumerate() {
                if !line.contains(query) {
                    continue;
                }
                let display_path = if prefix.as_os_str().is_empty() {
                    relative.clone()
                } else {
                    prefix.join(&relative)
                };
                report.matches.push(format!(
                    "{}:{}:{}",
                    display_path.display(),
                    line_index + 1,
                    truncate_output(line.to_string(), MAX_SEARCH_LINE_BYTES)
                ));
                if report.matches.len() == max_results {
                    report.truncated = true;
                    return Ok(report);
                }
            }
        }

        for child in child_directories.into_iter().rev() {
            pending.push(child);
        }
    }

    Ok(report)
}

fn should_skip_search_directory(name: &std::ffi::OsStr) -> bool {
    matches!(
        name.to_str(),
        Some(".git") | Some("target") | Some("node_modules")
    )
}

async fn resolve_workspace_write_path(
    workspace: &Path,
    requested: &Path,
) -> Result<(PathBuf, PathBuf), ToolError> {
    if requested.as_os_str().is_empty()
        || requested.is_absolute()
        || requested
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(ToolError::InvalidPath {
            tool: "write_file".to_string(),
            path: requested.to_path_buf(),
        });
    }

    let workspace =
        tokio::fs::canonicalize(workspace)
            .await
            .map_err(|source| ToolError::ResolveWorkspace {
                path: workspace.to_path_buf(),
                source,
            })?;
    let candidate = workspace.join(requested);
    let parent = candidate.parent().ok_or_else(|| ToolError::InvalidPath {
        tool: "write_file".to_string(),
        path: requested.to_path_buf(),
    })?;
    let parent = ensure_workspace_write_parent(&workspace, parent, requested).await?;

    let file_name = candidate
        .file_name()
        .ok_or_else(|| ToolError::InvalidPath {
            tool: "write_file".to_string(),
            path: requested.to_path_buf(),
        })?;
    let target = parent.join(file_name);
    match tokio::fs::symlink_metadata(&target).await {
        Ok(metadata) => {
            let canonical = tokio::fs::canonicalize(&target).await.map_err(|source| {
                ToolError::InspectFile {
                    path: target.clone(),
                    source,
                }
            })?;
            if !canonical.starts_with(&workspace) {
                return Err(ToolError::PathOutsideWorkspace {
                    tool: "write_file".to_string(),
                    path: requested.to_path_buf(),
                });
            }
            if !metadata.is_file() {
                return Err(ToolError::InvalidPath {
                    tool: "write_file".to_string(),
                    path: target,
                });
            }
        }
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(ToolError::InspectFile {
                path: target,
                source,
            });
        }
    }

    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        file_name.to_string_lossy(),
        Uuid::new_v4()
    ));
    Ok((target, temporary))
}

/// Resolve a write target's parent while creating missing directories.
///
/// Each existing component is canonicalized before the next component is
/// created. This keeps an existing symlink usable only when it resolves back
/// inside the workspace; `create_dir_all` is intentionally avoided because it
/// would follow an unvalidated symlink while creating descendants.
async fn ensure_workspace_write_parent(
    workspace: &Path,
    parent: &Path,
    requested: &Path,
) -> Result<PathBuf, ToolError> {
    let relative_parent =
        parent
            .strip_prefix(workspace)
            .map_err(|_| ToolError::PathOutsideWorkspace {
                tool: "write_file".to_string(),
                path: requested.to_path_buf(),
            })?;
    let mut current = workspace.to_path_buf();

    for component in relative_parent.components() {
        let name = match component {
            Component::Normal(name) => name,
            Component::CurDir => continue,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(ToolError::InvalidPath {
                    tool: "write_file".to_string(),
                    path: requested.to_path_buf(),
                });
            }
        };
        current.push(name);

        match tokio::fs::symlink_metadata(&current).await {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                validate_workspace_directory(workspace, &current, requested).await?;
            }
            Ok(metadata) if metadata.is_dir() => {
                validate_workspace_directory(workspace, &current, requested).await?;
            }
            Ok(_) => {
                return Err(ToolError::InvalidPath {
                    tool: "write_file".to_string(),
                    path: current,
                });
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                tokio::fs::create_dir(&current)
                    .await
                    .map_err(|source| ToolError::ResolveFile {
                        path: current.clone(),
                        source,
                    })?;
                validate_workspace_directory(workspace, &current, requested).await?;
            }
            Err(source) => {
                return Err(ToolError::ResolveFile {
                    path: current,
                    source,
                });
            }
        }
    }

    tokio::fs::canonicalize(parent)
        .await
        .map_err(|source| ToolError::ResolveFile {
            path: parent.to_path_buf(),
            source,
        })
}

async fn validate_workspace_directory(
    workspace: &Path,
    path: &Path,
    requested: &Path,
) -> Result<(), ToolError> {
    let canonical =
        tokio::fs::canonicalize(path)
            .await
            .map_err(|source| ToolError::ResolveFile {
                path: path.to_path_buf(),
                source,
            })?;
    if !canonical.starts_with(workspace) {
        return Err(ToolError::PathOutsideWorkspace {
            tool: "write_file".to_string(),
            path: requested.to_path_buf(),
        });
    }
    if !tokio::fs::metadata(path)
        .await
        .map_err(|source| ToolError::InspectFile {
            path: path.to_path_buf(),
            source,
        })?
        .is_dir()
    {
        return Err(ToolError::InvalidPath {
            tool: "write_file".to_string(),
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

async fn atomic_write_workspace_file(
    workspace: &Path,
    requested: &Path,
    content: &[u8],
) -> Result<(), ToolError> {
    let (target, temporary) = resolve_workspace_write_path(workspace, requested).await?;
    if let Err(source) = tokio::fs::write(&temporary, content).await {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(ToolError::WriteFile {
            path: temporary,
            source,
        });
    }

    if let Err(source) = tokio::fs::rename(&temporary, &target).await {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(ToolError::RenameFile {
            path: target,
            temporary,
            source,
        });
    }

    Ok(())
}

fn truncate_output(text: String, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text;
    }
    if max_bytes == 0 {
        return String::new();
    }

    const MARKER: &str = "…";
    let content_limit = max_bytes.saturating_sub(MARKER.len());
    let mut end = content_limit.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }

    format!("{}{}", &text[..end], MARKER)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_loop::CancellationToken;
    use crate::message::{FunctionCall, ToolCall};
    use crate::skills::{Skill, SkillCatalog};
    use std::{
        fs,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };
    use tempfile::tempdir;

    fn read_call(path: &str) -> ToolCall {
        ToolCall {
            id: "call-1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: serde_json::json!({ "path": path }).to_string(),
            },
        }
    }

    fn list_call(arguments: Value) -> ToolCall {
        ToolCall {
            id: "list-call-1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "list_files".to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn search_call(arguments: Value) -> ToolCall {
        ToolCall {
            id: "search-call-1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "search_files".to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn write_call(path: &str, content: &str) -> ToolCall {
        ToolCall {
            id: "write-call-1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "write_file".to_string(),
                arguments: serde_json::json!({
                    "path": path,
                    "content": content,
                })
                .to_string(),
            },
        }
    }

    fn edit_call(path: &str, old_text: &str, new_text: &str) -> ToolCall {
        ToolCall {
            id: "edit-call-1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "edit_file".to_string(),
                arguments: serde_json::json!({
                    "path": path,
                    "old_text": old_text,
                    "new_text": new_text,
                })
                .to_string(),
            },
        }
    }

    fn skill_call(name: &str) -> ToolCall {
        ToolCall {
            id: "skill-call-1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_skill".to_string(),
                arguments: serde_json::json!({ "name": name }).to_string(),
            },
        }
    }

    #[derive(Debug)]
    struct SchemaTool;

    impl Tool for SchemaTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "schema_tool",
                "A custom tool whose schema is enforced by the registry.",
                json!({
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "minLength": 1
                        }
                    },
                    "required": ["message"],
                    "additionalProperties": false
                }),
            )
        }

        fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
            Box::pin(async move {
                Ok(arguments["message"]
                    .as_str()
                    .expect("schema validation should ensure message is a string")
                    .to_string())
            })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    #[derive(Debug)]
    struct InvalidSchemaTool;

    impl Tool for InvalidSchemaTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "invalid_schema_tool",
                "A tool with an unsupported parameter schema.",
                json!({"type": "string"}),
            )
        }

        fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
            Box::pin(async { Ok("unused".to_string()) })
        }
    }

    #[derive(Debug)]
    struct InvalidMetadataTool;

    impl Tool for InvalidMetadataTool {
        fn definition(&self) -> ToolDefinition {
            let mut definition = ToolDefinition::function(
                "bad.name",
                "Invalid metadata fixture.",
                json!({"type": "object"}),
            );
            definition.definition_type = "custom".to_string();
            definition
        }

        fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
            Box::pin(async { Ok("unused".to_string()) })
        }
    }

    #[derive(Debug)]
    struct ChangingDefinitionTool {
        definition_calls: Arc<AtomicUsize>,
    }

    impl Tool for ChangingDefinitionTool {
        fn definition(&self) -> ToolDefinition {
            let call = self.definition_calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                ToolDefinition::function(
                    "changing_tool",
                    "A stable registration snapshot fixture.",
                    json!({
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                        "additionalProperties": false
                    }),
                )
            } else {
                ToolDefinition::function(
                    "changing_tool",
                    "A changed definition that must not escape the registry.",
                    json!({"type": "object", "properties": {}, "additionalProperties": false}),
                )
            }
        }

        fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
            Box::pin(async { Ok("changed".to_string()) })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    #[derive(Debug)]
    struct LargeResultTool;

    impl Tool for LargeResultTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "large_result",
                "Return an intentionally large structured result.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
            Box::pin(async { Ok::<String, ToolError>("unused".to_string()) })
        }

        fn execute_result<'a>(&'a self, _arguments: Value) -> ToolResultFuture<'a> {
            Box::pin(async {
                Ok(ToolResult {
                    content: "你好".repeat(10_000),
                    details: json!({"payload": "x".repeat(10_000)}),
                    is_error: true,
                    terminate: true,
                })
            })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    #[derive(Debug)]
    struct LargeUnknownResultTool;

    impl Tool for LargeUnknownResultTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "large_unknown_result",
                "Return an uncertain result with oversized details.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
            Box::pin(async { Ok::<String, ToolError>("unused".to_string()) })
        }

        fn execute_result<'a>(&'a self, _arguments: Value) -> ToolResultFuture<'a> {
            Box::pin(async {
                Ok(
                    ToolResult::error("external outcome is uncertain").with_details(json!({
                        "status": "unknown",
                        "payload": "x".repeat(10_000),
                    })),
                )
            })
        }

        fn risk_class(&self) -> RiskClass {
            RiskClass::ReadOnly
        }
    }

    struct RetainedContextTool {
        retained: Arc<Mutex<Option<ToolExecutionContext>>>,
    }

    impl Tool for RetainedContextTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "retained_context",
                "Retain a tool context to verify terminal update behavior.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
            Box::pin(async { Ok::<String, ToolError>("done".to_string()) })
        }

        fn execute_result_with_context<'a>(
            &'a self,
            _arguments: Value,
            context: ToolExecutionContext,
        ) -> ToolResultFuture<'a> {
            let retained = Arc::clone(&self.retained);
            Box::pin(async move {
                context.update(ToolUpdate::text("during"));
                retained
                    .lock()
                    .expect("retained context mutex should not be poisoned")
                    .replace(context.clone());
                Ok(ToolResult::text("done"))
            })
        }
    }

    #[test]
    fn execution_context_reports_cancellation_and_updates() {
        let cancellation = CancellationToken::new();
        let updates = Arc::new(Mutex::new(Vec::new()));
        let context = ToolExecutionContext::new(cancellation.clone()).with_update_sink({
            let updates = Arc::clone(&updates);
            move |update| {
                updates
                    .lock()
                    .expect("update mutex should not poison")
                    .push(update)
            }
        });

        context.update(ToolUpdate::text("started"));
        assert!(!context.is_cancelled());
        let child_token = context.cancellation_token();
        cancellation.cancel();
        assert!(context.is_cancelled());
        assert!(child_token.is_cancelled());
        assert_eq!(
            updates
                .lock()
                .expect("update mutex should not poison")
                .len(),
            1
        );
    }

    #[test]
    fn execution_context_bounds_progress_updates() {
        let updates = Arc::new(Mutex::new(Vec::new()));
        let context = ToolExecutionContext::new(CancellationToken::new())
            .with_output_limit(32)
            .with_update_sink({
                let updates = Arc::clone(&updates);
                move |update| {
                    updates
                        .lock()
                        .expect("update mutex should not poison")
                        .push(update)
                }
            });

        context.update(ToolUpdate::text("你好".repeat(32)).with_details(json!({
            "payload": "x".repeat(128)
        })));

        let updates = updates.lock().expect("update mutex should not poison");
        assert_eq!(updates.len(), 1);
        assert!(updates[0].content.len() <= 32);
        assert!(updates[0].details.is_null());
    }

    #[tokio::test]
    async fn registry_closes_progress_updates_after_tool_completion() {
        let retained = Arc::new(Mutex::new(None));
        let updates = Arc::new(Mutex::new(Vec::new()));
        let mut registry = ToolRegistry::new();
        registry
            .register(RetainedContextTool {
                retained: Arc::clone(&retained),
            })
            .expect("retained context tool should register");
        let call = ToolCall {
            id: "retained-call".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "retained_context".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let context = ToolExecutionContext::new(CancellationToken::new()).with_update_sink({
            let updates = Arc::clone(&updates);
            move |update| {
                updates
                    .lock()
                    .expect("updates mutex should not be poisoned")
                    .push(update.content);
            }
        });

        registry
            .execute_result_with_context(&call, context)
            .await
            .expect("tool should complete");
        retained
            .lock()
            .expect("retained context mutex should not be poisoned")
            .as_ref()
            .expect("tool should retain its context")
            .update(ToolUpdate::text("late"));

        assert_eq!(
            *updates
                .lock()
                .expect("updates mutex should not be poisoned"),
            vec!["during"]
        );
    }

    #[tokio::test]
    async fn registry_bounds_custom_tool_results() {
        let mut registry = ToolRegistry::new().with_output_limit(4_096);
        registry
            .register(LargeResultTool)
            .expect("large result tool should register");
        let call = ToolCall {
            id: "large-call".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "large_result".to_string(),
                arguments: "{}".to_string(),
            },
        };

        let result = registry
            .execute_result(&call)
            .await
            .expect("large result tool should execute");

        assert!(result.content.len() <= 4_096);
        assert_eq!(result.details["status"], "details_truncated");
        assert!(result.is_error);
        assert!(result.terminate);
    }

    #[tokio::test]
    async fn registry_preserves_uncertain_status_when_bounding_details() {
        let mut registry = ToolRegistry::new().with_output_limit(256);
        registry
            .register(LargeUnknownResultTool)
            .expect("uncertain result tool should register");
        let call = ToolCall {
            id: "unknown-call".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "large_unknown_result".to_string(),
                arguments: "{}".to_string(),
            },
        };

        let result = registry
            .execute_result(&call)
            .await
            .expect("uncertain result tool should execute");

        assert_eq!(result.details["status"], "unknown");
        assert_eq!(result.details["details_truncated"], true);
    }

    #[test]
    fn tool_result_bounding_preserves_reconciled_status() {
        let result = ToolResult::text("confirmed")
            .with_details(json!({
                "status": "reconciled",
                "evidence": "x".repeat(1_000),
            }))
            .bounded(64);

        assert_eq!(result.details["status"], "reconciled");
    }

    #[tokio::test]
    async fn custom_tool_schema_is_enforced_before_execution() {
        let mut registry = ToolRegistry::new();
        registry
            .register(SchemaTool)
            .expect("valid custom schema should register");

        let mut call = ToolCall {
            id: "schema-call".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "schema_tool".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let missing = registry
            .execute(&call)
            .await
            .expect_err("required field should be enforced");
        assert!(matches!(
            missing,
            ToolError::SchemaViolation { path, .. } if path == "arguments.message"
        ));

        call.function.arguments = json!({"message": 7}).to_string();
        let wrong_type = registry
            .execute(&call)
            .await
            .expect_err("property type should be enforced");
        assert!(matches!(
            wrong_type,
            ToolError::SchemaViolation { path, .. } if path == "arguments.message"
        ));

        call.function.arguments = json!({"message": "ok", "extra": true}).to_string();
        let unknown = registry
            .execute(&call)
            .await
            .expect_err("additional properties should be rejected");
        assert!(matches!(
            unknown,
            ToolError::SchemaViolation { path, .. } if path == "arguments.extra"
        ));

        call.function.arguments = json!({"message": "ok"}).to_string();
        assert_eq!(
            registry
                .execute(&call)
                .await
                .expect("valid schema arguments should execute"),
            "ok"
        );
    }

    #[tokio::test]
    async fn tool_arguments_have_size_and_depth_limits() {
        let mut registry = ToolRegistry::new();
        registry
            .register(SchemaTool)
            .expect("valid custom schema should register");
        let mut call = ToolCall {
            id: "bounded-call".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "schema_tool".to_string(),
                arguments: String::new(),
            },
        };

        call.function.arguments = json!({
            "message": "x".repeat(MAX_TOOL_ARGUMENT_BYTES)
        })
        .to_string();
        let oversized = registry
            .execute(&call)
            .await
            .expect_err("oversized arguments should be rejected");
        assert!(matches!(
            oversized,
            ToolError::ArgumentsTooLarge { limit, .. } if limit == MAX_TOOL_ARGUMENT_BYTES
        ));

        let mut nested = json!("leaf");
        for _ in 0..=MAX_TOOL_ARGUMENT_DEPTH {
            nested = json!([nested]);
        }
        call.function.arguments = json!({"message": nested}).to_string();
        let too_deep = registry
            .execute(&call)
            .await
            .expect_err("deep arguments should be rejected");
        assert!(matches!(
            too_deep,
            ToolError::ArgumentsTooDeep { limit, .. } if limit == MAX_TOOL_ARGUMENT_DEPTH
        ));
    }

    #[test]
    fn registry_rejects_an_unsupported_tool_schema() {
        let error = ToolRegistry::new()
            .register(InvalidSchemaTool)
            .expect_err("tool parameters must be an object schema");

        assert!(matches!(
            error,
            RegistryError::InvalidDefinition { name, reason }
                if name == "invalid_schema_tool" && reason.contains("type object")
        ));
    }

    #[test]
    fn registry_rejects_invalid_tool_definition_metadata() {
        let error = ToolRegistry::new()
            .register(InvalidMetadataTool)
            .expect_err("non-function tool definitions should be rejected");

        assert!(matches!(
            error,
            RegistryError::InvalidDefinition { name, reason }
                if name == "bad.name" && reason == "definition type must be function"
        ));
    }

    #[tokio::test]
    async fn registry_uses_the_definition_snapshot_for_advertising_and_execution() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry
            .register(ChangingDefinitionTool {
                definition_calls: Arc::clone(&calls),
            })
            .expect("initial definition should register");

        let definitions = registry.definitions();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            definitions[0].function.parameters["required"],
            json!(["message"])
        );

        let call = ToolCall {
            id: "changing-call".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "changing_tool".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let error = registry
            .execute(&call)
            .await
            .expect_err("the registration snapshot should still require message");
        assert!(matches!(
            error,
            ToolError::SchemaViolation { path, .. } if path == "arguments.message"
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn schema_subset_checks_unicode_lengths_numeric_bounds_and_array_items() {
        let definition = ToolDefinition::function(
            "bounded_schema",
            "A schema exercising the supported constraints.",
            json!({
                "type": "object",
                "properties": {
                    "label": {"type": "string", "minLength": 2},
                    "count": {"type": "integer", "minimum": 1, "maximum": 3},
                    "names": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["label", "count", "names"],
                "additionalProperties": false
            }),
        );
        validate_schema_definition(&definition).expect("supported schema should register");

        validate_schema_arguments(
            &definition,
            &json!({
                "label": "你好",
                "count": 2,
                "names": ["alpha", "beta"]
            }),
        )
        .expect("valid constraints should pass");

        let short_label =
            validate_schema_arguments(&definition, &json!({"label": "a", "count": 2, "names": []}))
                .expect_err("Unicode string length should be checked");
        assert!(matches!(
            short_label,
            ToolError::SchemaViolation { path, .. } if path == "arguments.label"
        ));

        let bad_item = validate_schema_arguments(
            &definition,
            &json!({"label": "ok", "count": 2, "names": [1]}),
        )
        .expect_err("array item types should be checked");
        assert!(matches!(
            bad_item,
            ToolError::SchemaViolation { path, .. } if path == "arguments.names[0]"
        ));

        let large_integer_definition = ToolDefinition::function(
            "large_integer_schema",
            "An integer precision regression fixture.",
            json!({
                "type": "object",
                "properties": {
                    "value": {"type": "integer", "maximum": 9007199254740992u64}
                },
                "required": ["value"],
                "additionalProperties": false
            }),
        );
        validate_schema_definition(&large_integer_definition)
            .expect("large integer schema should be valid");
        let too_large = validate_schema_arguments(
            &large_integer_definition,
            &json!({"value": 9007199254740993u64}),
        )
        .expect_err("integer bounds must not lose precision through f64");
        assert!(matches!(
            too_large,
            ToolError::SchemaViolation { path, .. } if path == "arguments.value"
        ));

        let enum_definition = ToolDefinition::function(
            "enum_schema",
            "A schema exercising enum and maximum string length.",
            json!({
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "safe"],
                        "maxLength": 4
                    },
                    "label": {
                        "type": "string",
                        "maxLength": 4
                    }
                },
                "required": ["mode", "label"],
                "additionalProperties": false
            }),
        );
        validate_schema_definition(&enum_definition)
            .expect("enum and maxLength should be supported");
        validate_schema_arguments(&enum_definition, &json!({"mode": "safe", "label": "ok"}))
            .expect("an allowed enum value within the length bound should pass");
        let invalid_enum =
            validate_schema_arguments(&enum_definition, &json!({"mode": "slow", "label": "ok"}))
                .expect_err("an unknown enum value should be rejected");
        assert!(matches!(
            invalid_enum,
            ToolError::SchemaViolation { path, .. } if path == "arguments.mode"
        ));
        let too_long =
            validate_schema_arguments(&enum_definition, &json!({"mode": "safe", "label": "hello"}))
                .expect_err("a string over maxLength should be rejected");
        assert!(matches!(
            too_long,
            ToolError::SchemaViolation { path, .. } if path == "arguments.label"
        ));

        let invalid_enum_definition = ToolDefinition::function(
            "invalid_enum_schema",
            "An enum with a value of the wrong type.",
            json!({
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": [1]}
                },
                "required": ["mode"],
                "additionalProperties": false
            }),
        );
        let invalid_enum_schema = validate_schema_definition(&invalid_enum_definition)
            .expect_err("enum members must match their declared type");
        assert!(invalid_enum_schema.contains("enum[0]"));

        let contradictory_schema = ToolDefinition::function(
            "contradictory_enum_schema",
            "An enum member outside the string length bounds.",
            json!({
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["long"], "maxLength": 3}
                },
                "required": ["mode"],
                "additionalProperties": false
            }),
        );
        let contradictory = validate_schema_definition(&contradictory_schema)
            .expect_err("an enum outside maxLength should be rejected at registration");
        assert!(contradictory.contains("enum[0]"));
    }

    #[test]
    fn bounded_details_falls_back_to_minimal_uncertain_status() {
        let result = ToolResult::error("uncertain")
            .with_details(json!({
                "status": "unknown",
                "payload": "x".repeat(128),
            }))
            .bounded(20);

        assert_eq!(result.details, json!({"status": "unknown"}));
    }

    #[tokio::test]
    async fn read_file_returns_workspace_content() {
        let directory = tempdir().expect("temporary directory should exist");
        fs::write(directory.path().join("hello.txt"), "hello from workspace")
            .expect("fixture should be written");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let registry = ToolRegistry::with_read_file(context).expect("tool should register");

        let result = registry
            .execute(&read_call("hello.txt"))
            .await
            .expect("read_file should succeed");

        assert_eq!(result, "hello from workspace");
    }

    #[tokio::test]
    async fn read_file_supports_one_indexed_offset_and_limit() {
        let directory = tempdir().expect("temporary directory should exist");
        fs::write(directory.path().join("lines.txt"), "one\ntwo\nthree\nfour")
            .expect("fixture should be written");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let registry = ToolRegistry::with_read_file(context).expect("tool should register");
        let mut call = read_call("lines.txt");
        call.function.arguments = serde_json::json!({
            "path": "lines.txt",
            "offset": 2,
            "limit": 2,
        })
        .to_string();

        let result = registry
            .execute(&call)
            .await
            .expect("ranged read should succeed");

        assert_eq!(result, "two\nthree");
    }

    #[tokio::test]
    async fn read_file_rejects_an_offset_outside_the_file() {
        let directory = tempdir().expect("temporary directory should exist");
        fs::write(directory.path().join("lines.txt"), "one\ntwo")
            .expect("fixture should be written");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let registry = ToolRegistry::with_read_file(context).expect("tool should register");
        let mut call = read_call("lines.txt");
        call.function.arguments = serde_json::json!({
            "path": "lines.txt",
            "offset": 3,
        })
        .to_string();

        let error = registry
            .execute(&call)
            .await
            .expect_err("out-of-range offset should fail");

        assert!(matches!(error, ToolError::InvalidRange { tool, .. } if tool == "read_file"));
    }

    #[tokio::test]
    async fn read_file_rejects_parent_directory() {
        let directory = tempdir().expect("temporary directory should exist");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let registry = ToolRegistry::with_read_file(context).expect("tool should register");

        let error = registry
            .execute(&read_call("../secret.txt"))
            .await
            .expect_err("parent path should be rejected");

        assert!(matches!(error, ToolError::InvalidPath { .. }));
    }

    #[tokio::test]
    async fn list_files_returns_sorted_entries_with_bounded_depth() {
        let directory = tempdir().expect("temporary directory should exist");
        fs::create_dir_all(directory.path().join("nested/deeper"))
            .expect("nested directories should be created");
        fs::write(directory.path().join("z.txt"), "z").expect("fixture should be written");
        fs::write(directory.path().join("a.txt"), "a").expect("fixture should be written");
        fs::write(directory.path().join("nested/child.txt"), "child")
            .expect("fixture should be written");
        fs::write(directory.path().join("nested/deeper/deep.txt"), "deep")
            .expect("fixture should be written");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register_list_files(context)
            .expect("list_files should register");

        let result = registry
            .execute(&list_call(json!({
                "max_depth": 1,
                "max_entries": 10
            })))
            .await
            .expect("list_files should succeed");

        assert_eq!(
            result.lines().collect::<Vec<_>>(),
            vec![
                "a.txt",
                "nested/",
                "z.txt",
                "nested/child.txt",
                "nested/deeper/"
            ]
        );
        assert!(!result.contains("deep.txt"));
    }

    #[tokio::test]
    async fn list_files_rejects_parent_directory_and_bounds_results() {
        let directory = tempdir().expect("temporary directory should exist");
        fs::write(directory.path().join("a.txt"), "a").expect("fixture should be written");
        fs::write(directory.path().join("b.txt"), "b").expect("fixture should be written");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register_list_files(context)
            .expect("list_files should register");

        let error = registry
            .execute(&list_call(json!({"path": "../"})))
            .await
            .expect_err("parent path should be rejected");
        assert!(matches!(error, ToolError::InvalidPath { tool, .. } if tool == "list_files"));

        let result = registry
            .execute(&list_call(json!({"max_entries": 1})))
            .await
            .expect("bounded listing should succeed");
        assert!(result.contains("[truncated after 1 entries]"));
    }

    #[tokio::test]
    async fn search_files_returns_literal_matches_with_line_numbers() {
        let directory = tempdir().expect("temporary directory should exist");
        fs::create_dir_all(directory.path().join("nested"))
            .expect("nested directory should be created");
        fs::create_dir_all(directory.path().join("target"))
            .expect("generated directory should be created");
        fs::write(
            directory.path().join("a.txt"),
            "needle at the beginning\nno match",
        )
        .expect("fixture should be written");
        fs::write(
            directory.path().join("nested/b.txt"),
            "first\nneedle in nested file",
        )
        .expect("fixture should be written");
        fs::write(
            directory.path().join("target/generated.txt"),
            "needle hidden",
        )
        .expect("generated fixture should be written");

        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register_search_files(context)
            .expect("search_files should register");

        let result = registry
            .execute(&search_call(json!({"query": "needle"})))
            .await
            .expect("search_files should succeed");

        assert!(result.contains("a.txt:1:needle at the beginning"));
        assert!(result.contains("nested/b.txt:2:needle in nested file"));
        assert!(!result.contains("generated.txt"));
    }

    #[tokio::test]
    async fn search_files_rejects_empty_query_and_bounds_matches() {
        let directory = tempdir().expect("temporary directory should exist");
        fs::write(directory.path().join("a.txt"), "needle one\nneedle two")
            .expect("fixture should be written");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register_search_files(context)
            .expect("search_files should register");

        let empty = registry
            .execute(&search_call(json!({"query": ""})))
            .await
            .expect_err("empty query should fail validation");
        assert!(matches!(
            empty,
            ToolError::InvalidArgumentValue { name, .. } if name == "search_files"
        ));

        let result = registry
            .execute(&search_call(json!({
                "query": "needle",
                "max_results": 1
            })))
            .await
            .expect("bounded search should succeed");
        assert!(result.contains("[search truncated after 1 matching lines]"));
    }

    #[tokio::test]
    async fn write_file_replaces_content_atomically_within_workspace() {
        let directory = tempdir().expect("temporary directory should exist");
        fs::write(directory.path().join("hello.txt"), "old").expect("fixture should be written");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let mut registry =
            ToolRegistry::with_read_file(context.clone()).expect("read tool should register");
        registry
            .register_write_file(context)
            .expect("write tool should register");

        let result = registry
            .execute(&write_call("hello.txt", "new contents"))
            .await
            .expect("write_file should succeed");

        assert!(result.contains("Wrote 12 bytes"));
        assert_eq!(
            fs::read_to_string(directory.path().join("hello.txt"))
                .expect("written file should be readable"),
            "new contents"
        );
        assert_eq!(
            registry.concurrency("write_file"),
            Some(ToolConcurrency::Exclusive)
        );
        assert_eq!(
            registry.risk_class("write_file"),
            Some(RiskClass::Reversible)
        );
    }

    #[tokio::test]
    async fn write_file_creates_missing_nested_parent_directories() {
        let directory = tempdir().expect("temporary directory should exist");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register_write_file(context)
            .expect("write tool should register");

        registry
            .execute(&write_call("nested/path/file.txt", "nested contents"))
            .await
            .expect("write_file should create nested parents");

        assert_eq!(
            fs::read_to_string(directory.path().join("nested/path/file.txt"))
                .expect("nested file should be readable"),
            "nested contents"
        );
    }

    #[tokio::test]
    async fn write_file_rejects_parent_directory() {
        let directory = tempdir().expect("temporary directory should exist");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register_write_file(context)
            .expect("write tool should register");

        let error = registry
            .execute(&write_call("../secret.txt", "blocked"))
            .await
            .expect_err("parent path should be rejected");

        assert!(matches!(error, ToolError::InvalidPath { tool, .. } if tool == "write_file"));
    }

    #[tokio::test]
    async fn write_file_enforces_the_file_size_limit_before_creating_a_temp_file() {
        let directory = tempdir().expect("temporary directory should exist");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 3,
            max_tool_output_bytes: 1024,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register_write_file(context)
            .expect("write tool should register");

        let error = registry
            .execute(&write_call("hello.txt", "too large"))
            .await
            .expect_err("oversized content should be rejected");

        assert!(matches!(error, ToolError::FileTooLarge { limit: 3, .. }));
        assert!(!directory.path().join("hello.txt").exists());
    }

    #[tokio::test]
    async fn edit_file_replaces_one_exact_match_atomically() {
        let directory = tempdir().expect("temporary directory should exist");
        fs::write(directory.path().join("hello.txt"), "hello world")
            .expect("fixture should be written");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register_edit_file(context)
            .expect("edit tool should register");

        registry
            .execute(&edit_call("hello.txt", "world", "rust"))
            .await
            .expect("edit_file should succeed");

        assert_eq!(
            fs::read_to_string(directory.path().join("hello.txt"))
                .expect("edited file should be readable"),
            "hello rust"
        );
    }

    #[tokio::test]
    async fn edit_file_requires_one_unique_match() {
        let directory = tempdir().expect("temporary directory should exist");
        fs::write(directory.path().join("hello.txt"), "same and same")
            .expect("fixture should be written");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register_edit_file(context)
            .expect("edit tool should register");

        let missing = registry
            .execute(&edit_call("hello.txt", "absent", "new"))
            .await
            .expect_err("missing target should fail");
        assert!(matches!(missing, ToolError::EditTargetNotFound { .. }));

        let duplicate = registry
            .execute(&edit_call("hello.txt", "same", "new"))
            .await
            .expect_err("duplicate target should fail");
        assert!(matches!(
            duplicate,
            ToolError::EditTargetNotUnique { matches: 2, .. }
        ));
    }

    #[tokio::test]
    async fn read_file_rejects_unknown_arguments() {
        let directory = tempdir().expect("temporary directory should exist");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let registry = ToolRegistry::with_read_file(context).expect("tool should register");
        let mut call = read_call("hello.txt");
        call.function.arguments = serde_json::json!({
            "path": "hello.txt",
            "unexpected": true
        })
        .to_string();

        let error = registry
            .execute(&call)
            .await
            .expect_err("unknown arguments should fail validation");

        assert!(matches!(error, ToolError::InvalidArguments { name, .. } if name == "read_file"));
    }

    #[test]
    fn validates_builtin_arguments_before_execution() {
        let directory = tempdir().expect("temporary directory should exist");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let registry = ToolRegistry::with_read_file(context).expect("tool should register");
        let mut call = read_call("hello.txt");
        call.function.arguments = serde_json::json!({
            "path": "hello.txt",
            "offset": 0
        })
        .to_string();
        let arguments: Value = serde_json::from_str(&call.function.arguments)
            .expect("test arguments should be valid JSON");

        let error = registry
            .validate_arguments(&call, &arguments)
            .expect_err("zero offset should fail preflight validation");

        assert!(matches!(
            error,
            ToolError::InvalidRange { tool, reason }
                if tool == "read_file" && reason == "offset must be greater than zero"
        ));
    }

    #[tokio::test]
    async fn rejects_invalid_tool_call_invariants_before_lookup() {
        let directory = tempdir().expect("temporary directory should exist");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let registry = ToolRegistry::with_read_file(context).expect("tool should register");
        let mut call = read_call("hello.txt");
        call.id.clear();

        let error = registry
            .execute(&call)
            .await
            .expect_err("empty call ids should fail validation");

        assert!(
            matches!(error, ToolError::InvalidCall { reason, .. } if reason == "call id must not be empty")
        );
    }

    #[test]
    fn built_in_read_tools_declare_parallel_execution() {
        let directory = tempdir().expect("temporary directory should exist");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let mut registry = ToolRegistry::with_read_file(context).expect("tool should register");
        assert_eq!(
            registry.concurrency("read_file"),
            Some(ToolConcurrency::Parallel)
        );
        assert_eq!(
            registry.replay_policy("read_file"),
            Some(ToolReplayPolicy::Safe)
        );
        assert_eq!(registry.risk_class("read_file"), Some(RiskClass::ReadOnly));
        assert_eq!(registry.concurrency("missing"), None);
        assert_eq!(registry.replay_policy("missing"), None);
        assert_eq!(registry.risk_class("missing"), None);

        registry
            .register_skill_reader(SkillCatalog::new(), 1024)
            .expect("read_skill should register");
        assert_eq!(
            registry.concurrency("read_skill"),
            Some(ToolConcurrency::Parallel)
        );
        assert_eq!(
            registry.replay_policy("read_skill"),
            Some(ToolReplayPolicy::Safe)
        );
        assert_eq!(registry.risk_class("read_skill"), Some(RiskClass::ReadOnly));
    }

    #[test]
    fn mutating_tools_default_to_never_replay() {
        let directory = tempdir().expect("temporary directory should exist");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let mut registry =
            ToolRegistry::with_read_file(context.clone()).expect("read tool should register");
        registry
            .register_write_file(context.clone())
            .expect("write tool should register");
        registry
            .register_edit_file(context)
            .expect("edit tool should register");

        assert_eq!(
            registry.replay_policy("write_file"),
            Some(ToolReplayPolicy::Never)
        );
        assert_eq!(
            registry.replay_policy("edit_file"),
            Some(ToolReplayPolicy::Never)
        );
    }

    #[derive(Debug)]
    struct HiddenTool;

    impl Tool for HiddenTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "hidden_tool",
                "A trusted caller-only tool.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn exposure(&self) -> ToolExposure {
            ToolExposure::Hidden
        }

        fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
            Box::pin(async { Ok("hidden".to_string()) })
        }
    }

    #[test]
    fn hidden_tools_remain_executable_but_are_not_model_visible() {
        let mut registry = ToolRegistry::new();
        registry
            .register(HiddenTool)
            .expect("hidden tool should register");

        assert_eq!(registry.exposure("hidden_tool"), Some(ToolExposure::Hidden));
        assert_eq!(registry.definitions().len(), 1);
        assert!(registry.model_definitions().is_empty());
    }

    #[tokio::test]
    async fn gateway_separates_model_discovery_from_trusted_execution() {
        let mut registry = ToolRegistry::new();
        registry
            .register(HiddenTool)
            .expect("hidden tool should register");
        let gateway = registry.gateway();
        let call = ToolCall {
            id: "hidden-call".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "hidden_tool".to_string(),
                arguments: "{}".to_string(),
            },
        };

        assert!(gateway.model_definitions().is_empty());
        assert_eq!(
            gateway.capabilities("hidden_tool"),
            Some(ToolCapabilities {
                concurrency: ToolConcurrency::Exclusive,
                replay_policy: ToolReplayPolicy::Never,
                exposure: ToolExposure::Hidden,
                risk_class: RiskClass::Unknown,
            })
        );
        assert!(matches!(
            gateway.prepare_model_call(&call, json!({})),
            Err(ToolError::ToolNotModelVisible { .. })
        ));

        let prepared = gateway
            .prepare_trusted_call(&call, json!({}))
            .expect("trusted callers may execute hidden tools");
        let result = gateway
            .execute_prepared(prepared)
            .await
            .expect("trusted execution should succeed");
        assert_eq!(result.content, "hidden");
    }

    #[test]
    fn registry_can_remove_a_child_capability() {
        let directory = tempdir().expect("temporary directory should exist");
        let context = ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        };
        let registry = ToolRegistry::with_read_file(context).expect("tool should register");
        let restricted = registry.without("read_file");

        assert!(registry.concurrency("read_file").is_some());
        assert!(restricted.concurrency("read_file").is_none());
    }

    #[tokio::test]
    async fn read_skill_returns_full_instructions_for_a_registered_name() {
        let mut catalog = SkillCatalog::new();
        catalog
            .insert(
                Skill::new(
                    "research",
                    "Research tasks",
                    "Use primary sources.",
                    "/skills/research/SKILL.md",
                )
                .expect("skill fixture should be valid")
                .with_model_invocation_disabled(true),
            )
            .expect("skill should be registered");
        let mut registry = ToolRegistry::new();
        registry
            .register_skill_reader(catalog, 1024)
            .expect("read_skill should register");

        let result = registry
            .execute(&skill_call("research"))
            .await
            .expect("registered skill should be readable");

        assert!(result.contains("Use primary sources."));
        assert!(result.contains("References are relative to /skills/research."));
    }

    #[tokio::test]
    async fn read_skill_rejects_unknown_arguments() {
        let mut catalog = SkillCatalog::new();
        catalog
            .insert(
                Skill::new(
                    "research",
                    "Research tasks",
                    "Use primary sources.",
                    "/skills/research/SKILL.md",
                )
                .expect("skill fixture should be valid"),
            )
            .expect("skill should be registered");
        let mut registry = ToolRegistry::new();
        registry
            .register_skill_reader(catalog, 1024)
            .expect("read_skill should register");
        let call = ToolCall {
            id: "skill-call-2".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_skill".to_string(),
                arguments: serde_json::json!({
                    "name": "research",
                    "path": "outside.txt"
                })
                .to_string(),
            },
        };

        let error = registry
            .execute(&call)
            .await
            .expect_err("unknown arguments should fail validation");

        assert!(matches!(error, ToolError::InvalidArguments { name, .. } if name == "read_skill"));
    }

    #[tokio::test]
    async fn read_skill_rejects_unknown_names() {
        let mut registry = ToolRegistry::new();
        registry
            .register_skill_reader(SkillCatalog::new(), 1024)
            .expect("read_skill should register");

        let error = registry
            .execute(&skill_call("missing"))
            .await
            .expect_err("unknown skill should fail");

        assert!(matches!(error, ToolError::UnknownSkill { name } if name == "missing"));
    }

    #[tokio::test]
    async fn read_skill_enforces_output_limit_without_truncating_instructions() {
        let mut catalog = SkillCatalog::new();
        catalog
            .insert(
                Skill::new(
                    "large",
                    "Large skill",
                    "This instruction must remain complete.",
                    "/skills/large/SKILL.md",
                )
                .expect("skill fixture should be valid"),
            )
            .expect("skill should be registered");
        let mut registry = ToolRegistry::new();
        registry
            .register_skill_reader(catalog, 8)
            .expect("read_skill should register");

        let error = registry
            .execute(&skill_call("large"))
            .await
            .expect_err("oversized skill should fail");

        assert!(matches!(error, ToolError::SkillTooLarge { name, .. } if name == "large"));
    }

    #[test]
    fn truncate_output_keeps_utf8_and_limit() {
        let result = truncate_output("你好，世界".to_string(), 7);

        assert!(result.is_char_boundary(result.len()));
        assert!(result.len() <= 7);
        assert!(result.ends_with('…'));
    }
}
