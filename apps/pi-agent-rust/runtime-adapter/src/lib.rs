//! Composition between the agent loop and the independent memory control plane.
//!
//! The core agent only knows about [`pi_agent_rust::MemoryHookRequest`] and a
//! versioned runtime proposal. This crate supplies the application-specific
//! glue that turns a request into a [`MemoryNeed`], runs
//! [`MemoryAgentCoordinator`], and returns the validated proposal to the core
//! loop. Provider implementations remain behind [`MemoryRegistry`].

pub mod command_tool;
pub mod preparation_scheduler;
pub mod prepared_memory_store;
pub mod prepared_tool;
pub mod speculative_memory;

pub use command_tool::{FixedCommandTool, FixedCommandToolError};
pub use preparation_scheduler::{PreparationScheduler, PreparationSchedulerError, PreparationTask};
pub use prepared_memory_store::{PreparedMemoryStore, PreparedMemoryStoreError};
pub use prepared_tool::{
    PreparedArtifactHandoffError, PreparedArtifactSource, PreparedArtifactSourceKind,
    PreparedArtifactTool, PreparedArtifactToolInput, PREPARED_ARTIFACT_TOOL_NAME,
    PREPARED_ARTIFACT_TOOL_SCHEMA_VERSION,
};
pub use speculative_memory::{
    SpeculativeMemoryError, SpeculativeMemoryHookAdapter, SpeculativeMemoryScheduler,
    SpeculativeMemoryTask, StructuralNeedPredictor,
};

use std::{collections::BTreeMap, sync::Arc};

use pi_agent_memory::{
    ArtifactRef, ContentRef, EpisodeRef, EvidenceLocator, MemoryBudget, MemoryItem, MemoryKind,
    MemoryManifest, MemoryProvenance, MemoryQuery, MemoryRecord, MemoryRegistry, MemoryScope,
    Modality, OperationContext, RecordIngestRequest, SubjectRef,
};
use pi_agent_memory_agent::{
    CoordinatorConfig, MemoryAgentCoordinator, MemoryEpoch, MemoryNeed, NeedHorizon, PolicyEpoch,
    TaskState,
};
use pi_agent_runtime::{PreparationPlan, PreparationStatus};
use pi_agent_rust::{
    AfterRunFuture, AfterRunRequest, AgentLoop, ChatModel, MemoryHookConfig, MemoryHookFuture,
    MemoryHookRequest, RunOutcome,
};
use serde_json::json;
use uuid::Uuid;

const DEFAULT_AFTER_RUN_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_MAX_TRANSCRIPT_BYTES: usize = 256 * 1024;

/// A planned memory operation for one model request.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    pub need: MemoryNeed,
    pub task: TaskState,
    pub operation: OperationContext,
}

/// Converts one agent request into a memory operation.
pub trait MemoryPlanner: Send + Sync {
    fn plan(&self, request: &MemoryHookRequest) -> Result<Option<MemoryPlan>, String>;
}

impl<F> MemoryPlanner for F
where
    F: Fn(&MemoryHookRequest) -> Result<Option<MemoryPlan>, String> + Send + Sync,
{
    fn plan(&self, request: &MemoryHookRequest) -> Result<Option<MemoryPlan>, String> {
        self(request)
    }
}

/// Installs a memory-agent hook without coupling the core agent to providers.
pub struct MemoryHookAdapter<P> {
    registry: Arc<MemoryRegistry>,
    planner: Arc<P>,
    coordinator_config: CoordinatorConfig,
    hook_config: MemoryHookConfig,
}

impl<P> MemoryHookAdapter<P>
where
    P: MemoryPlanner + 'static,
{
    pub fn new(
        registry: Arc<MemoryRegistry>,
        planner: P,
        coordinator_config: CoordinatorConfig,
        hook_config: MemoryHookConfig,
    ) -> Self {
        Self {
            registry,
            planner: Arc::new(planner),
            coordinator_config,
            hook_config,
        }
    }

    /// Adds the memory-agent hook to an existing agent.
    pub fn attach<M>(self, agent: AgentLoop<M>) -> AgentLoop<M>
    where
        M: ChatModel,
    {
        let registry = Arc::clone(&self.registry);
        let planner = Arc::clone(&self.planner);
        let coordinator_config = self.coordinator_config;

        agent.with_memory_hook(self.hook_config, move |request| {
            let registry = Arc::clone(&registry);
            let planner = Arc::clone(&planner);
            let plan = planner.plan(&request);
            Box::pin(async move {
                let Some(plan) = plan? else {
                    return Ok(None);
                };
                let mut coordinator = MemoryAgentCoordinator::new(&registry, coordinator_config)
                    .map_err(|error| error.to_string())?;
                let prepared = coordinator
                    .prepare(&plan.need, &plan.task, plan.operation, request.context)
                    .await
                    .map_err(|error| error.to_string())?;
                Ok(Some(prepared.proposal))
            }) as MemoryHookFuture
        })
    }
}

/// Determines whether a failed memory write changes the agent run outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfterRunIngestPolicy {
    /// Return the write error through [`pi_agent_rust::AgentError`].
    Strict,
    /// Keep the model outcome and report the write error to the optional sink.
    BestEffort,
}

/// Configuration for the provider-neutral run transcript writer.
#[derive(Clone)]
pub struct AfterRunMemoryConfig {
    pub method_id: String,
    pub namespace: String,
    pub producer: String,
    pub scope: MemoryScope,
    pub policy: AfterRunIngestPolicy,
    pub timeout_ms: Option<u64>,
    pub max_transcript_bytes: usize,
    pub warning_sink: Option<Arc<dyn Fn(String) + Send + Sync>>,
}

impl AfterRunMemoryConfig {
    pub fn new(method_id: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            method_id: method_id.into(),
            namespace: namespace.into(),
            producer: "pi-agent.runtime".to_owned(),
            scope: MemoryScope::Agent,
            policy: AfterRunIngestPolicy::Strict,
            timeout_ms: Some(DEFAULT_AFTER_RUN_TIMEOUT_MS),
            max_transcript_bytes: DEFAULT_MAX_TRANSCRIPT_BYTES,
            warning_sink: None,
        }
    }

    pub fn with_warning_sink<F>(mut self, sink: F) -> Self
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        self.warning_sink = Some(Arc::new(sink));
        self
    }
}

/// Configuration for converting consumed preparation outputs into memory
/// records without copying artifact payloads.
#[derive(Debug, Clone)]
pub struct PreparedArtifactIngestConfig {
    pub method_id: String,
    pub namespace: String,
    pub producer: String,
    pub scope: MemoryScope,
    pub kind: MemoryKind,
    pub modality: Modality,
    pub subject: Option<SubjectRef>,
    pub task: String,
    pub dataset_revision: String,
    pub source: String,
    pub episode_id: Option<String>,
    pub method_version: Option<String>,
    pub timeout_ms: Option<u64>,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl PreparedArtifactIngestConfig {
    pub fn new(method_id: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            method_id: method_id.into(),
            namespace: namespace.into(),
            producer: "pi-agent.runtime.preparation".to_owned(),
            scope: MemoryScope::Agent,
            kind: MemoryKind::Episodic,
            modality: Modality::Mixed,
            subject: None,
            task: "prepared artifact".to_owned(),
            dataset_revision: "runtime".to_owned(),
            source: "pi-agent.runtime".to_owned(),
            episode_id: None,
            method_version: None,
            timeout_ms: None,
            metadata: BTreeMap::new(),
        }
    }
}

/// Converts consumed preparation outputs into lossless memory records.
pub fn prepared_artifact_records(
    plan: &PreparationPlan,
    config: &PreparedArtifactIngestConfig,
    ingest_time_ms: u64,
) -> Result<Vec<MemoryRecord>, String> {
    if plan.status != PreparationStatus::Consumed {
        return Err(format!(
            "preparation {} must be consumed before memory ingest",
            plan.plan_id
        ));
    }
    if config.namespace.trim().is_empty() {
        return Err("prepared artifact memory namespace must be non-empty".to_owned());
    }
    if config.method_id.trim().is_empty() {
        return Err("prepared artifact memory method_id must be non-empty".to_owned());
    }
    if plan.artifacts.is_empty() {
        return Err(format!(
            "consumed preparation {} has no artifact outputs",
            plan.plan_id
        ));
    }
    let handoff = PreparedArtifactToolInput::from_consumed(plan)
        .map_err(|error| format!("invalid consumed preparation handoff: {error}"))?;

    let episode_id = config
        .episode_id
        .clone()
        .unwrap_or_else(|| format!("preparation:{}", plan.plan_id));
    plan.artifacts
        .iter()
        .map(|artifact| {
            prepared_artifact_record(
                plan,
                config,
                artifact,
                &handoff.evidence,
                &episode_id,
                ingest_time_ms,
            )
        })
        .collect()
}

fn prepared_artifact_record(
    plan: &PreparationPlan,
    config: &PreparedArtifactIngestConfig,
    artifact: &ArtifactRef,
    evidence: &[EvidenceLocator],
    episode_id: &str,
    ingest_time_ms: u64,
) -> Result<MemoryRecord, String> {
    if artifact.uri.trim().is_empty()
        || artifact.media_type.trim().is_empty()
        || artifact.sha256.trim().is_empty()
    {
        return Err(format!(
            "preparation artifact {} has incomplete metadata",
            artifact.artifact_id
        ));
    }

    let mut metadata = config.metadata.clone();
    metadata.insert(
        "preparation_plan_id".to_owned(),
        serde_json::Value::String(plan.plan_id.to_string()),
    );
    metadata.insert(
        "preparation_action_id".to_owned(),
        serde_json::Value::String(plan.action_id.as_uuid().to_string()),
    );
    let artifact_evidence = evidence
        .iter()
        .filter(|locator| locator.artifact.artifact_id == artifact.artifact_id)
        .cloned()
        .collect::<Vec<_>>();
    Ok(MemoryRecord {
        item: MemoryItem {
            id: artifact.artifact_id,
            version: 1,
            kind: config.kind,
            scope: config.scope.clone(),
            modality: config.modality,
            content: ContentRef::Blob {
                uri: artifact.uri.clone(),
                media_type: artifact.media_type.clone(),
                sha256: artifact.sha256.clone(),
                size_bytes: artifact.bytes,
            },
            namespace: config.namespace.clone(),
            episode_id: Some(episode_id.to_owned()),
            subject: config.subject.clone(),
            source: config.source.clone(),
            metadata,
            abstraction_level: 0,
            event_time: None,
            ingest_time_ms,
            schema_version: 1,
        },
        episode: Some(EpisodeRef {
            dataset_or_run_id: plan.run_id.as_uuid().to_string(),
            episode_id: episode_id.to_owned(),
            task: config.task.clone(),
            time_range: None,
            dataset_revision: config.dataset_revision.clone(),
            source: config.source.clone(),
        }),
        evidence: artifact_evidence,
        provenance: MemoryProvenance {
            producer: config.producer.clone(),
            source_id: plan.run_id.as_uuid().to_string(),
            source_event_id: Some(plan.action_id.as_uuid().to_string()),
            parent_record_ids: Vec::new(),
            method_version: config.method_version.clone(),
        },
    })
}

/// Adapter that ingests consumed preparation artifacts into a selected memory
/// method using the provider's lossless record contract.
pub struct PreparedArtifactMemoryAdapter {
    registry: Arc<MemoryRegistry>,
    config: PreparedArtifactIngestConfig,
}

impl PreparedArtifactMemoryAdapter {
    pub fn new(registry: Arc<MemoryRegistry>, config: PreparedArtifactIngestConfig) -> Self {
        Self { registry, config }
    }

    /// Ingests one consumed preparation. Retries for the same plan must reuse
    /// `ingest_time_ms`, because it is part of the persisted record identity.
    pub async fn ingest(
        &self,
        plan: &PreparationPlan,
        ingest_time_ms: u64,
    ) -> Result<pi_agent_memory::IngestReceipt, String> {
        let method = self
            .registry
            .get(&self.config.method_id)
            .ok_or_else(|| format!("memory method is not registered: {}", self.config.method_id))?;
        let records = prepared_artifact_records(plan, &self.config, ingest_time_ms)?;
        let idempotency_key = format!("preparation-artifacts:{}", plan.plan_id);
        let ingest = method.ingest_records(
            RecordIngestRequest {
                records,
                relations: Vec::new(),
                idempotency_key: idempotency_key.clone(),
            },
            OperationContext {
                namespace: self.config.namespace.clone(),
                run_id: plan.run_id.as_uuid(),
                session_id: None,
                episode_id: None,
                timeout_ms: self.config.timeout_ms,
                idempotency_key: Some(idempotency_key),
            },
        );
        match self.config.timeout_ms {
            Some(0) => Err("prepared artifact memory timeout must be greater than zero".to_owned()),
            Some(timeout_ms) => {
                tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), ingest)
                    .await
                    .map_err(|_| "prepared artifact memory ingest timed out".to_owned())?
                    .map_err(|error| error.to_string())
            }
            None => ingest.await.map_err(|error| error.to_string()),
        }
    }
}

/// Writes one completed or failed run to a selected memory method.
///
/// This adapter deliberately stores a JSON transcript. Providers that support
/// richer modalities can replace this adapter or derive additional records
/// from the same `AfterRunRequest` without changing the agent core.
pub struct AfterRunMemoryAdapter {
    registry: Arc<MemoryRegistry>,
    config: AfterRunMemoryConfig,
}

impl AfterRunMemoryAdapter {
    pub fn new(registry: Arc<MemoryRegistry>, config: AfterRunMemoryConfig) -> Self {
        Self { registry, config }
    }

    pub fn attach<M>(self, agent: AgentLoop<M>) -> AgentLoop<M>
    where
        M: ChatModel,
    {
        let registry = Arc::clone(&self.registry);
        let config = self.config;
        agent.with_after_run_hook(move |request| {
            let registry = Arc::clone(&registry);
            let config = config.clone();
            Box::pin(async move {
                let result = ingest_run(&registry, &config, request).await;
                match result {
                    Ok(()) => Ok(()),
                    Err(error) if config.policy == AfterRunIngestPolicy::BestEffort => {
                        if let Some(sink) = config.warning_sink {
                            sink(error);
                        }
                        Ok(())
                    }
                    Err(error) => Err(error),
                }
            }) as AfterRunFuture
        })
    }
}

async fn ingest_run(
    registry: &MemoryRegistry,
    config: &AfterRunMemoryConfig,
    request: AfterRunRequest,
) -> Result<(), String> {
    let method = registry
        .get(&config.method_id)
        .ok_or_else(|| format!("memory method is not registered: {}", config.method_id))?;
    let manifest = method.manifest();
    let status = match request.outcome {
        RunOutcome::Completed => "completed",
        RunOutcome::Failed => "failed",
        RunOutcome::Cancelled => "cancelled",
        RunOutcome::Unknown => "unknown",
    };
    let error = request.error;
    let steps = request
        .result
        .as_ref()
        .map_or(request.steps, |result| result.steps);
    let messages = request.transcript;
    let content = json!({
        "status": status,
        "error": error,
        "steps": steps,
        "model_retries": request.model_retries,
        "messages": messages,
    });
    let content_bytes = serde_json::to_vec(&content)
        .map_err(|error| format!("serialize run transcript: {error}"))?;
    if config.max_transcript_bytes == 0 {
        return Err("after-run transcript byte limit must be greater than zero".to_owned());
    }
    if content_bytes.len() > config.max_transcript_bytes {
        return Err(format!(
            "after-run transcript is {} bytes, limit is {}",
            content_bytes.len(),
            config.max_transcript_bytes
        ));
    }
    let run_id = request.context.run_id;
    let turn_id = request.context.turn_id;
    let (modality, content) = transcript_content(&manifest, content, content_bytes)?;
    let record = MemoryRecord {
        item: MemoryItem {
            // One runtime run may contain several user turns, especially when
            // recovery resumes an existing ledger. Each transcript is an
            // independently replayable episode, so its record identity and
            // idempotency key must use the turn identity.
            id: turn_id,
            version: 1,
            kind: MemoryKind::Episodic,
            scope: config.scope.clone(),
            modality,
            content,
            namespace: config.namespace.clone(),
            episode_id: Some(request.context.turn_id.to_string()),
            subject: None,
            source: config.producer.clone(),
            metadata: std::collections::BTreeMap::new(),
            abstraction_level: 0,
            event_time: None,
            ingest_time_ms: unix_time_ms(),
            schema_version: 1,
        },
        episode: Some(EpisodeRef {
            dataset_or_run_id: run_id.to_string(),
            episode_id: request.context.turn_id.to_string(),
            task: "agent run".to_owned(),
            time_range: None,
            dataset_revision: "runtime".to_owned(),
            source: config.producer.clone(),
        }),
        evidence: Vec::new(),
        provenance: MemoryProvenance {
            producer: config.producer.clone(),
            source_id: run_id.to_string(),
            source_event_id: Some(request.context.turn_id.to_string()),
            parent_record_ids: Vec::new(),
            method_version: Some(manifest.version),
        },
    };
    let ingest = method.ingest_records(
        RecordIngestRequest {
            records: vec![record],
            relations: Vec::new(),
            idempotency_key: format!("agent-turn:{turn_id}"),
        },
        OperationContext {
            namespace: config.namespace.clone(),
            run_id,
            session_id: request.context.session_id,
            episode_id: Some(request.context.turn_id),
            timeout_ms: config.timeout_ms,
            idempotency_key: Some(format!("agent-turn:{turn_id}")),
        },
    );
    let receipt = match config.timeout_ms {
        Some(0) => return Err("after-run memory timeout must be greater than zero".to_owned()),
        Some(timeout_ms) => {
            tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), ingest)
                .await
                .map_err(|_| "after-run memory ingest timed out".to_owned())?
        }
        None => ingest.await,
    };
    receipt.map_err(|error| error.to_string())?;
    Ok(())
}

fn transcript_content(
    manifest: &MemoryManifest,
    value: serde_json::Value,
    serialized: Vec<u8>,
) -> Result<(Modality, ContentRef), String> {
    if manifest.capabilities.modalities.contains(&Modality::Mixed) {
        return Ok((Modality::Mixed, ContentRef::Json { value }));
    }
    if manifest.capabilities.modalities.contains(&Modality::Text) {
        let text = String::from_utf8(serialized)
            .map_err(|error| format!("serialize text transcript: {error}"))?;
        return Ok((Modality::Text, ContentRef::Text { text }));
    }
    Err(format!(
        "memory method {} does not support Mixed or Text transcript storage",
        manifest.method_id
    ))
}

fn unix_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

/// Text planner for providers that implement semantic content search.
///
/// The user's text is kept in `MemoryQuery::input`, so a structural provider
/// that does not implement content search rejects the request explicitly
/// instead of silently returning unrelated recent records.
#[derive(Debug, Clone, Copy)]
pub struct LatestUserTextPlanner {
    pub max_query_chars: usize,
}

impl Default for LatestUserTextPlanner {
    fn default() -> Self {
        Self {
            max_query_chars: 4_096,
        }
    }
}

impl MemoryPlanner for LatestUserTextPlanner {
    fn plan(&self, request: &MemoryHookRequest) -> Result<Option<MemoryPlan>, String> {
        plan_latest_user_text(request, self.max_query_chars, true, Vec::new())
    }
}

/// Explicit structural baseline for providers that only support filters and
/// recency ordering. It intentionally leaves the query input empty; callers
/// should use this planner only when “latest records in this namespace” is the
/// desired experiment condition.
#[derive(Debug, Clone, Copy)]
pub struct StructuralLatestMemoryPlanner {
    pub max_query_chars: usize,
}

impl Default for StructuralLatestMemoryPlanner {
    fn default() -> Self {
        Self {
            max_query_chars: 4_096,
        }
    }
}

impl MemoryPlanner for StructuralLatestMemoryPlanner {
    fn plan(&self, request: &MemoryHookRequest) -> Result<Option<MemoryPlan>, String> {
        plan_latest_user_text(request, self.max_query_chars, false, Vec::new())
    }
}

/// Structural planner that limits retrieval to one memory modality.
///
/// This keeps modality evaluation explicit: a video benchmark does not
/// silently fall back to text or mixed records in the same namespace.
#[derive(Debug, Clone, Copy)]
pub struct ModalityMemoryPlanner {
    pub modality: Modality,
    pub max_query_chars: usize,
}

impl ModalityMemoryPlanner {
    pub const fn new(modality: Modality) -> Self {
        Self {
            modality,
            max_query_chars: 4_096,
        }
    }
}

impl MemoryPlanner for ModalityMemoryPlanner {
    fn plan(&self, request: &MemoryHookRequest) -> Result<Option<MemoryPlan>, String> {
        plan_latest_user_text(request, self.max_query_chars, false, vec![self.modality])
    }
}

/// Text-to-modality planner for providers that support cross-modal search.
///
/// Unlike [`ModalityMemoryPlanner`], this planner keeps the user's text as the
/// query input and restricts returned memories to the configured modalities.
/// The selected provider must advertise both the text and target modalities
/// together with `supports_cross_modal`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossModalMemoryPlanner {
    pub modalities: Vec<Modality>,
    pub max_query_chars: usize,
}

impl CrossModalMemoryPlanner {
    pub fn new(modalities: Vec<Modality>) -> Result<Self, String> {
        if modalities.is_empty() {
            return Err("cross-modal planner requires at least one modality".to_owned());
        }
        Ok(Self {
            modalities,
            max_query_chars: 4_096,
        })
    }
}

impl MemoryPlanner for CrossModalMemoryPlanner {
    fn plan(&self, request: &MemoryHookRequest) -> Result<Option<MemoryPlan>, String> {
        plan_latest_user_text(request, self.max_query_chars, true, self.modalities.clone())
    }
}

fn plan_latest_user_text(
    request: &MemoryHookRequest,
    max_query_chars: usize,
    include_query_input: bool,
    modalities: Vec<Modality>,
) -> Result<Option<MemoryPlan>, String> {
    let Some(text) = request
        .messages
        .iter()
        .rev()
        .filter(|message| message.role == "user")
        .filter_map(|message| message.content.as_deref())
        .map(str::trim)
        .find(|content| !content.is_empty())
    else {
        return Ok(None);
    };
    if max_query_chars == 0 {
        return Err("max_query_chars must be positive".to_owned());
    }

    let query_text = text.chars().take(max_query_chars).collect::<String>();
    let context = &request.context;
    let turn_id = context
        .turn_id
        .ok_or_else(|| "memory hook request requires a turn_id".to_owned())?;
    let run_id = context.run_id;
    let expiry = context.now_ms.saturating_add(2_000);
    let budget = MemoryBudget {
        max_hits: context.max_candidates as usize,
        max_context_tokens: context.max_tokens as usize,
        max_latency_ms: None,
        ..MemoryBudget::default()
    };
    let active_modalities = if modalities.is_empty() {
        vec![Modality::Text]
    } else {
        modalities.clone()
    };

    Ok(Some(MemoryPlan {
        need: MemoryNeed {
            need_id: Uuid::new_v4(),
            query: MemoryQuery {
                input: include_query_input.then(|| ContentRef::Text {
                    text: query_text.clone(),
                }),
                namespace: context.namespace.clone(),
                scope: None,
                subject: None,
                episode_id: None,
                kinds: Vec::new(),
                // The text input describes the request, but the memory may be
                // a mixed transcript, image, video, or embodied state. Leave
                // modality filtering to the selected provider/router.
                modalities,
                time_range: None,
                visibility_cutoff_ms: Some(context.now_ms),
                graph: None,
                budget,
            },
            provider_hints: Vec::new(),
            horizon: NeedHorizon::CurrentTurn,
            confidence: 1.0,
            estimated_benefit: 0.8,
            created_at_ms: context.now_ms,
            expires_at_ms: Some(expiry),
        },
        task: TaskState {
            run_id,
            turn_id,
            namespace: context.namespace.clone(),
            runtime_revision: context.current_revision,
            memory_epoch: MemoryEpoch::new(context.memory_revision.value()),
            policy_epoch: PolicyEpoch::new(context.policy_revision.value()),
            goal_summary: query_text.clone(),
            latest_query: Some(query_text),
            current_plan: None,
            recent_tools: Vec::new(),
            active_modalities,
            remaining_deadline_ms: None,
            context_token_budget: context.max_tokens,
        },
        operation: OperationContext {
            namespace: context.namespace.clone(),
            run_id: run_id.as_uuid(),
            session_id: None,
            episode_id: None,
            timeout_ms: None,
            idempotency_key: Some(format!("memory-hook:{}:{}", run_id.as_uuid(), turn_id)),
        },
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use super::*;
    use pi_agent_memory::{
        ArtifactRef, ContentRef, IngestReceipt, IngestRequest, JsonlProjectionStore,
        MemoryCapabilities, MemoryError, MemoryFuture, MemoryItem, MemoryKind, MemoryManifest,
        MemoryMethod, MemoryQuery, MemoryScope, Modality, OperationContext, ProjectionMemoryMethod,
        RetrieveResponse,
    };
    use pi_agent_runtime::{
        ActionId, Budget, PreparationEffectClass, PreparationPlan, PreparationStatus,
        ResourceScope, Revision, RunId,
    };
    use pi_agent_rust::{AfterRunRequest, ChatFuture, ChatOptions, ChatRequest, ToolRegistry};
    use pi_agent_rust::{ChatModel, Message, RunContext, RunOutcome};

    struct StubProvider;

    impl MemoryMethod for StubProvider {
        fn manifest(&self) -> MemoryManifest {
            MemoryManifest {
                method_id: "text.stub".to_owned(),
                version: "0.1.0".to_owned(),
                schema_version: 1,
                capabilities: MemoryCapabilities {
                    modalities: vec![Modality::Text],
                    memory_kinds: vec![MemoryKind::Semantic],
                    supports_content_search: true,
                    supports_time_filter: true,
                    supports_visibility_cutoff: true,
                    supports_graph: false,
                    supports_cross_modal: false,
                    supports_update: false,
                    supports_delete: false,
                    supports_batch: false,
                    supports_record_ingest: false,
                    supports_record_get: false,
                    deterministic: true,
                    persistent: false,
                },
            }
        }

        fn ingest<'a>(
            &'a self,
            _request: IngestRequest,
            _context: OperationContext,
        ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
            Box::pin(std::future::ready(Err(MemoryError::UnsupportedCapability(
                "test provider is read-only".to_owned(),
            ))))
        }

        fn retrieve<'a>(
            &'a self,
            _query: MemoryQuery,
            _context: OperationContext,
        ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
            Box::pin(std::future::ready(Ok(RetrieveResponse {
                query_id: Uuid::from_u128(10),
                hits: vec![pi_agent_memory::MemoryHit {
                    item_id: Uuid::from_u128(11),
                    source_record_ids: Vec::new(),
                    version: 1,
                    method_id: "text.stub".to_owned(),
                    rank: 1,
                    score: Some(0.9),
                    evidence: Vec::new(),
                    evidence_refs: Vec::new(),
                    projection: Some(ContentRef::Text {
                        text: "previous parser fix".to_owned(),
                    }),
                    provenance: "text.stub:record-11".to_owned(),
                    snapshot_id: Some("snapshot-1".to_owned()),
                    latency_ms: 1,
                }],
                snapshot_id: Some("snapshot-1".to_owned()),
                latency_ms: 1,
                truncated: false,
                telemetry: None,
            })))
        }

        fn get<'a>(
            &'a self,
            item_id: Uuid,
            _context: OperationContext,
        ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
            Box::pin(std::future::ready(Err(MemoryError::NotFound(item_id))))
        }
    }

    struct RecordingModel {
        requests: Arc<Mutex<Vec<Vec<Message>>>>,
    }

    fn consumed_artifact_plan() -> PreparationPlan {
        let run_id = RunId::new();
        let artifact = ArtifactRef {
            artifact_id: Uuid::from_u128(70),
            uri: "cas://prepared/sha256:video".to_owned(),
            media_type: "video/mp4".to_owned(),
            sha256: "sha256:video".to_owned(),
            bytes: 4_096,
            codec: Some("h264".to_owned()),
            schema: Some("episode-video-v1".to_owned()),
            stream_key: Some("camera/front".to_owned()),
            storage_revision: Some("fs-v1".to_owned()),
        };
        let mut plan = PreparationPlan::new(
            run_id,
            ActionId::new(),
            Revision::INITIAL,
            "memory.retrieve_video",
            "sha256:arguments",
            PreparationEffectClass::ReadOnly,
            ResourceScope::new(["memory:provider:video"]),
            Budget {
                wall_time_ms: Some(1_000),
                tokens: Some(256),
                tool_calls: Some(1),
            },
            0.9,
            1_000,
            5_000,
        )
        .expect("test preparation should be valid");
        plan.status = PreparationStatus::Consumed;
        plan.artifacts = vec![artifact];
        plan
    }

    #[test]
    fn consumed_preparation_becomes_a_lossless_artifact_record() {
        let mut plan = consumed_artifact_plan();
        plan.evidence = vec![EvidenceLocator {
            artifact: plan.artifacts[0].clone(),
            entity_path: Some("robot/gripper/object-3".to_owned()),
            time_range: None,
            frame_start: Some(36),
            frame_end: Some(42),
            step_start: Some(12),
            step_end: Some(14),
            byte_start: None,
            byte_end: None,
        }];
        let mut config = PreparedArtifactIngestConfig::new("jsonl-prepared", "bench/video");
        config.modality = Modality::Video;
        config.task = "retrieve front camera episode".to_owned();
        config.dataset_revision = "robotics-v2".to_owned();
        config.metadata.insert(
            "source_kind".to_owned(),
            serde_json::Value::String("prepared".to_owned()),
        );

        let records = prepared_artifact_records(&plan, &config, 2_000)
            .expect("consumed artifacts should convert to records");
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.item.id, Uuid::from_u128(70));
        assert_eq!(record.item.namespace, "bench/video");
        assert_eq!(record.item.modality, Modality::Video);
        assert_eq!(record.item.metadata["source_kind"], "prepared");
        assert_eq!(record.evidence[0].artifact, plan.artifacts[0]);
        assert_eq!(record.evidence[0].frame_start, Some(36));
        assert_eq!(record.evidence[0].frame_end, Some(42));
        assert_eq!(
            record.episode.as_ref().unwrap().dataset_revision,
            "robotics-v2"
        );
        assert_eq!(
            record.provenance.source_event_id.as_deref(),
            Some(plan.action_id.as_uuid().to_string().as_str())
        );
        assert!(matches!(
            record.item.content,
            ContentRef::Blob { ref uri, .. } if uri == "cas://prepared/sha256:video"
        ));
    }

    #[test]
    fn unconsumed_preparation_cannot_be_ingested() {
        let mut plan = consumed_artifact_plan();
        plan.status = PreparationStatus::Ready;
        let config = PreparedArtifactIngestConfig::new("jsonl-prepared", "bench/video");

        let error = prepared_artifact_records(&plan, &config, 2_000)
            .expect_err("only consumed preparation outputs may enter memory");
        assert!(error.contains("must be consumed"));
    }

    #[tokio::test]
    async fn prepared_artifact_adapter_ingests_records_into_projection_memory() {
        let directory = tempfile::tempdir().expect("temporary directory should exist");
        let store = JsonlProjectionStore::open(directory.path().join("memory.jsonl"))
            .expect("projection store should open");
        let provider = Arc::new(
            ProjectionMemoryMethod::new(Arc::new(store), "jsonl-prepared", "0.1.0")
                .expect("projection provider should build"),
        );
        let mut registry = MemoryRegistry::new();
        registry
            .register(provider.clone())
            .expect("prepared provider should register");
        let plan = consumed_artifact_plan();
        let adapter = PreparedArtifactMemoryAdapter::new(
            Arc::new(registry),
            PreparedArtifactIngestConfig::new("jsonl-prepared", "bench/video"),
        );

        let receipt = adapter
            .ingest(&plan, 2_000)
            .await
            .expect("prepared artifact should be ingested");
        assert_eq!(receipt.item_ids, vec![Uuid::from_u128(70)]);
        let retry = adapter
            .ingest(&plan, 2_000)
            .await
            .expect("the same preparation ingest should be idempotent");
        assert!(retry.deduplicated);
        let record = provider
            .get_record(
                Uuid::from_u128(70),
                OperationContext {
                    namespace: "bench/video".to_owned(),
                    run_id: plan.run_id.as_uuid(),
                    session_id: None,
                    episode_id: None,
                    timeout_ms: None,
                    idempotency_key: None,
                },
            )
            .await
            .expect("ingested artifact record should be readable");
        assert_eq!(record.evidence.len(), 1);
        assert_eq!(record.evidence[0].artifact, plan.artifacts[0]);
        assert_eq!(
            record.provenance.source_id,
            plan.run_id.as_uuid().to_string()
        );
    }

    impl ChatModel for RecordingModel {
        fn complete<'a>(&'a self, request: ChatRequest<'a>) -> ChatFuture<'a> {
            self.requests
                .lock()
                .expect("request mutex should not be poisoned")
                .push(request.messages.to_vec());
            Box::pin(async { Ok(Message::assistant(Some("done".to_owned()), None)) })
        }
    }

    #[test]
    fn latest_user_text_planner_builds_a_scoped_text_need() {
        let context = RunContext::new(None);
        let request = MemoryHookRequest {
            context: pi_agent_runtime::MemoryHookContext {
                run_id: RunId::from_uuid(context.run_id),
                turn_id: Some(context.turn_id),
                namespace: "bench/text".to_owned(),
                current_revision: pi_agent_runtime::Revision::INITIAL,
                memory_revision: pi_agent_runtime::MemoryRevision::INITIAL,
                policy_revision: pi_agent_runtime::PolicyRevision::INITIAL,
                now_ms: 1_000,
                max_tokens: 256,
                max_candidates: 4,
                max_projection_bytes: 4_096,
            },
            messages: vec![Message::user("fix the parser")],
        };

        let plan = LatestUserTextPlanner::default()
            .plan(&request)
            .expect("planner should succeed")
            .expect("a user message should produce a plan");
        assert_eq!(plan.need.query.namespace, "bench/text");
        assert!(plan.need.query.modalities.is_empty());
        assert_eq!(
            plan.need.query.input,
            Some(ContentRef::Text {
                text: "fix the parser".to_owned()
            })
        );
        assert_eq!(plan.task.latest_query.as_deref(), Some("fix the parser"));
        assert_eq!(plan.operation.run_id, context.run_id);
    }

    #[test]
    fn structural_planner_makes_recency_fallback_explicit() {
        let context = RunContext::new(None);
        let request = MemoryHookRequest {
            context: pi_agent_runtime::MemoryHookContext {
                run_id: RunId::from_uuid(context.run_id),
                turn_id: Some(context.turn_id),
                namespace: "bench/text".to_owned(),
                current_revision: pi_agent_runtime::Revision::INITIAL,
                memory_revision: pi_agent_runtime::MemoryRevision::INITIAL,
                policy_revision: pi_agent_runtime::PolicyRevision::INITIAL,
                now_ms: 1_000,
                max_tokens: 256,
                max_candidates: 4,
                max_projection_bytes: 4_096,
            },
            messages: vec![Message::user("fix the parser")],
        };

        let plan = StructuralLatestMemoryPlanner::default()
            .plan(&request)
            .expect("planner should succeed")
            .expect("a user message should produce a plan");
        assert!(plan.need.query.input.is_none());
    }

    #[test]
    fn modality_planner_limits_retrieval_to_video_records() {
        let context = RunContext::new(None);
        let request = MemoryHookRequest {
            context: pi_agent_runtime::MemoryHookContext {
                run_id: RunId::from_uuid(context.run_id),
                turn_id: Some(context.turn_id),
                namespace: "bench/video".to_owned(),
                current_revision: pi_agent_runtime::Revision::INITIAL,
                memory_revision: pi_agent_runtime::MemoryRevision::INITIAL,
                policy_revision: pi_agent_runtime::PolicyRevision::INITIAL,
                now_ms: 1_000,
                max_tokens: 256,
                max_candidates: 4,
                max_projection_bytes: 4_096,
            },
            messages: vec![Message::user("find the grasping frame")],
        };

        let plan = ModalityMemoryPlanner::new(Modality::Video)
            .plan(&request)
            .expect("planner should succeed")
            .expect("a user message should produce a plan");
        assert_eq!(plan.need.query.modalities, vec![Modality::Video]);
        assert!(plan.need.query.input.is_none());
        assert_eq!(plan.task.active_modalities, vec![Modality::Video]);
    }

    #[test]
    fn cross_modal_planner_keeps_text_query_for_video_memories() {
        let context = RunContext::new(None);
        let request = MemoryHookRequest {
            context: pi_agent_runtime::MemoryHookContext {
                run_id: RunId::from_uuid(context.run_id),
                turn_id: Some(context.turn_id),
                namespace: "bench/video".to_owned(),
                current_revision: pi_agent_runtime::Revision::INITIAL,
                memory_revision: pi_agent_runtime::MemoryRevision::INITIAL,
                policy_revision: pi_agent_runtime::PolicyRevision::INITIAL,
                now_ms: 1_000,
                max_tokens: 256,
                max_candidates: 4,
                max_projection_bytes: 4_096,
            },
            messages: vec![Message::user("find the grasping frame")],
        };

        let planner = CrossModalMemoryPlanner::new(vec![Modality::Video])
            .expect("cross-modal planner should require one target modality");
        let plan = planner
            .plan(&request)
            .expect("planner should succeed")
            .expect("a user message should produce a plan");
        assert_eq!(plan.need.query.modalities, vec![Modality::Video]);
        assert_eq!(
            plan.need.query.input,
            Some(ContentRef::Text {
                text: "find the grasping frame".to_owned()
            })
        );
        assert_eq!(plan.task.active_modalities, vec![Modality::Video]);
    }

    #[test]
    fn planner_skips_requests_without_user_text() {
        let context = RunContext::new(None);
        let request = MemoryHookRequest {
            context: pi_agent_runtime::MemoryHookContext {
                run_id: RunId::from_uuid(context.run_id),
                turn_id: Some(context.turn_id),
                namespace: "bench/text".to_owned(),
                current_revision: pi_agent_runtime::Revision::INITIAL,
                memory_revision: pi_agent_runtime::MemoryRevision::INITIAL,
                policy_revision: pi_agent_runtime::PolicyRevision::INITIAL,
                now_ms: 1_000,
                max_tokens: 256,
                max_candidates: 4,
                max_projection_bytes: 4_096,
            },
            messages: vec![Message::system("system")],
        };

        assert!(LatestUserTextPlanner::default()
            .plan(&request)
            .expect("planner should succeed")
            .is_none());
    }

    #[test]
    fn after_run_transcript_falls_back_to_text_for_text_only_provider() {
        let manifest = StubProvider.manifest();
        let value = json!({"status": "completed", "messages": []});
        let serialized = serde_json::to_vec(&value).expect("transcript should serialize");

        let (modality, content) = transcript_content(&manifest, value, serialized)
            .expect("text-only providers should accept a text transcript");

        assert_eq!(modality, Modality::Text);
        let ContentRef::Text { text } = content else {
            panic!("text-only provider should receive text content");
        };
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&text).unwrap()["status"],
            "completed"
        );
    }

    #[tokio::test]
    async fn adapter_retrieves_memory_before_the_model_request() {
        let mut registry = MemoryRegistry::new();
        registry
            .register(Arc::new(StubProvider))
            .expect("stub provider should register");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let agent = AgentLoop::new(
            RecordingModel {
                requests: Arc::clone(&requests),
            },
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("test agent should build");
        let agent = MemoryHookAdapter::new(
            Arc::new(registry),
            LatestUserTextPlanner::default(),
            CoordinatorConfig::default(),
            MemoryHookConfig {
                namespace: "bench/text".to_owned(),
                max_tokens: 64,
                ..MemoryHookConfig::default()
            },
        )
        .attach(agent);

        agent
            .run("fix the parser")
            .await
            .expect("memory-adapted agent should complete");

        let requests = requests
            .lock()
            .expect("request mutex should not be poisoned");
        assert_eq!(requests.len(), 1);
        assert!(requests[0].iter().any(|message| {
            message.role == "system"
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("previous parser fix"))
        }));
    }

    #[tokio::test]
    async fn adapter_reads_a_persisted_projection_store_before_the_model_request() {
        let directory = tempfile::tempdir().expect("temporary directory should exist");
        let store_path = directory.path().join("memory.jsonl");
        let store = JsonlProjectionStore::open(&store_path).expect("projection store should open");
        let provider = ProjectionMemoryMethod::new(Arc::new(store), "jsonl-projection", "0.1.0")
            .expect("projection provider should build");
        let seed_run_id = Uuid::new_v4();
        provider
            .ingest(
                IngestRequest {
                    items: vec![MemoryItem {
                        id: Uuid::from_u128(101),
                        version: 1,
                        kind: MemoryKind::Semantic,
                        scope: MemoryScope::Shared,
                        modality: Modality::Text,
                        content: ContentRef::Text {
                            text: "previous parser fix".to_owned(),
                        },
                        namespace: "bench/text".to_owned(),
                        episode_id: None,
                        subject: None,
                        source: "adapter-test".to_owned(),
                        metadata: BTreeMap::new(),
                        abstraction_level: 0,
                        event_time: None,
                        ingest_time_ms: 1_000,
                        schema_version: 1,
                    }],
                    relations: Vec::new(),
                    idempotency_key: "seed-1".to_owned(),
                },
                OperationContext {
                    namespace: "bench/text".to_owned(),
                    run_id: seed_run_id,
                    session_id: None,
                    episode_id: None,
                    timeout_ms: None,
                    idempotency_key: Some("seed-1".to_owned()),
                },
            )
            .await
            .expect("projection record should be persisted");

        let mut registry = MemoryRegistry::new();
        registry
            .register(Arc::new(provider))
            .expect("projection provider should register");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let agent = AgentLoop::new(
            RecordingModel {
                requests: Arc::clone(&requests),
            },
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("test agent should build");
        let agent = MemoryHookAdapter::new(
            Arc::new(registry),
            StructuralLatestMemoryPlanner::default(),
            CoordinatorConfig::default(),
            MemoryHookConfig {
                namespace: "bench/text".to_owned(),
                max_tokens: 64,
                ..MemoryHookConfig::default()
            },
        )
        .attach(agent);

        agent
            .run("fix the parser")
            .await
            .expect("persisted memory agent should complete");

        let requests = requests
            .lock()
            .expect("request mutex should not be poisoned");
        assert!(requests[0].iter().any(|message| {
            message.role == "system"
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("previous parser fix"))
        }));
    }

    #[tokio::test]
    async fn modality_planner_injects_video_artifact_metadata() {
        let directory = tempfile::tempdir().expect("temporary directory should exist");
        let store = JsonlProjectionStore::open(directory.path().join("memory.jsonl"))
            .expect("projection store should open");
        let provider = Arc::new(
            ProjectionMemoryMethod::new(Arc::new(store), "jsonl-video", "0.1.0")
                .expect("video projection provider should build"),
        );
        let mut registry = MemoryRegistry::new();
        registry
            .register(provider)
            .expect("video provider should register");
        let registry = Arc::new(registry);

        let mut plan = consumed_artifact_plan();
        plan.evidence = vec![EvidenceLocator {
            artifact: plan.artifacts[0].clone(),
            entity_path: Some("robot/gripper/object-3".to_owned()),
            time_range: None,
            frame_start: Some(36),
            frame_end: Some(42),
            step_start: Some(12),
            step_end: Some(14),
            byte_start: None,
            byte_end: None,
        }];
        let mut ingest_config = PreparedArtifactIngestConfig::new("jsonl-video", "bench/video");
        ingest_config.modality = Modality::Video;
        PreparedArtifactMemoryAdapter::new(Arc::clone(&registry), ingest_config)
            .ingest(&plan, 2_000)
            .await
            .expect("prepared video should be persisted");

        let requests = Arc::new(Mutex::new(Vec::new()));
        let agent = AgentLoop::new(
            RecordingModel {
                requests: Arc::clone(&requests),
            },
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("test agent should build");
        let agent = MemoryHookAdapter::new(
            registry,
            ModalityMemoryPlanner::new(Modality::Video),
            CoordinatorConfig::default(),
            MemoryHookConfig {
                namespace: "bench/video".to_owned(),
                max_tokens: 256,
                ..MemoryHookConfig::default()
            },
        )
        .attach(agent);

        agent
            .run("find the grasping frame")
            .await
            .expect("video memory agent should complete");

        let requests = requests
            .lock()
            .expect("request mutex should not be poisoned");
        let system_content = requests[0]
            .iter()
            .find(|message| message.role == "system")
            .and_then(|message| message.content.as_deref())
            .expect("memory injection should be in the system message");
        assert!(system_content.contains("cas://prepared/sha256:video"));
        assert!(system_content.contains("\"frame_start\":36"));
        assert!(system_content.contains("\"frame_end\":42"));
    }

    #[tokio::test]
    async fn after_run_adapter_persists_a_versioned_transcript_record() {
        let directory = tempfile::tempdir().expect("temporary directory should exist");
        let store = JsonlProjectionStore::open(directory.path().join("memory.jsonl"))
            .expect("projection store should open");
        let provider = Arc::new(
            ProjectionMemoryMethod::new(Arc::new(store), "jsonl-projection", "0.1.0")
                .expect("projection provider should build"),
        );
        let mut registry = MemoryRegistry::new();
        registry
            .register(provider.clone())
            .expect("projection provider should register");

        let run_context = Arc::new(Mutex::new(None));
        let observed_run_context = Arc::clone(&run_context);
        let agent = AgentLoop::new(
            RecordingModel {
                requests: Arc::new(Mutex::new(Vec::new())),
            },
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("test agent should build")
        .with_event_sink(move |event| {
            if let pi_agent_rust::AgentEvent::RunStarted { context } = event {
                *observed_run_context
                    .lock()
                    .expect("run context mutex should not be poisoned") =
                    Some((context.run_id, context.turn_id));
            }
        });
        let agent = AfterRunMemoryAdapter::new(
            Arc::new(registry),
            AfterRunMemoryConfig::new("jsonl-projection", "bench/runs"),
        )
        .attach(agent);

        let result = agent
            .run("remember this run")
            .await
            .expect("agent should complete");
        assert_eq!(result.steps, 1);
        let (run_id, turn_id) = run_context
            .lock()
            .expect("run context mutex should not be poisoned")
            .expect("run started event should be observed");
        let record = provider
            .get_record(
                turn_id,
                OperationContext {
                    namespace: "bench/runs".to_owned(),
                    run_id,
                    session_id: None,
                    episode_id: None,
                    timeout_ms: None,
                    idempotency_key: None,
                },
            )
            .await
            .expect("transcript record should be readable");
        assert_eq!(record.item.id, turn_id);
        assert_eq!(record.item.kind, MemoryKind::Episodic);
        assert_eq!(record.item.modality, Modality::Mixed);
        assert_eq!(record.item.namespace, "bench/runs");
        assert_eq!(
            record.episode.as_ref().unwrap().dataset_or_run_id,
            run_id.to_string()
        );
        let ContentRef::Json { value } = record.item.content else {
            panic!("transcript should be stored as JSON");
        };
        assert_eq!(value["status"], "completed");
        assert_eq!(value["messages"][0]["role"], "user");
    }

    #[tokio::test]
    async fn after_run_adapter_keeps_multiple_turns_in_one_runtime_run_distinct() {
        let directory = tempfile::tempdir().expect("temporary directory should exist");
        let store = JsonlProjectionStore::open(directory.path().join("memory.jsonl"))
            .expect("projection store should open");
        let provider = Arc::new(
            ProjectionMemoryMethod::new(Arc::new(store), "jsonl-projection", "0.1.0")
                .expect("projection provider should build"),
        );
        let mut registry = MemoryRegistry::new();
        registry
            .register(provider.clone())
            .expect("projection provider should register");
        let registry = Arc::new(registry);
        let config = AfterRunMemoryConfig::new("jsonl-projection", "bench/runs");
        let run_id = Uuid::from_u128(700);
        let first_turn = Uuid::from_u128(701);
        let second_turn = Uuid::from_u128(702);

        for (turn_id, prompt) in [(first_turn, "first turn"), (second_turn, "second turn")] {
            ingest_run(
                &registry,
                &config,
                AfterRunRequest {
                    context: RunContext {
                        session_id: None,
                        run_id,
                        turn_id,
                    },
                    transcript: vec![Message::user(prompt)],
                    steps: 1,
                    model_retries: 0,
                    outcome: RunOutcome::Completed,
                    result: None,
                    error: None,
                },
            )
            .await
            .expect("each turn should be independently idempotent");
        }

        for (turn_id, prompt) in [(first_turn, "first turn"), (second_turn, "second turn")] {
            let record = provider
                .get_record(
                    turn_id,
                    OperationContext {
                        namespace: "bench/runs".to_owned(),
                        run_id,
                        session_id: None,
                        episode_id: None,
                        timeout_ms: None,
                        idempotency_key: None,
                    },
                )
                .await
                .expect("turn transcript should be readable");
            assert_eq!(
                record.episode.as_ref().unwrap().dataset_or_run_id,
                run_id.to_string()
            );
            let ContentRef::Json { value } = record.item.content else {
                panic!("turn transcript should be stored as JSON");
            };
            assert_eq!(value["messages"][0]["content"], prompt);
        }
    }

    #[tokio::test]
    async fn after_run_ingest_policy_controls_provider_failures() {
        let mut registry = MemoryRegistry::new();
        registry
            .register(Arc::new(StubProvider))
            .expect("stub provider should register");
        let strict_agent = AgentLoop::new(
            RecordingModel {
                requests: Arc::new(Mutex::new(Vec::new())),
            },
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("test agent should build");
        let strict_agent = AfterRunMemoryAdapter::new(
            Arc::new(registry),
            AfterRunMemoryConfig::new("text.stub", "bench/runs"),
        )
        .attach(strict_agent);
        let error = strict_agent
            .run("strict write")
            .await
            .expect_err("strict policy should surface provider failure");
        assert!(error.to_string().contains("record_ingest"));

        let mut registry = MemoryRegistry::new();
        registry
            .register(Arc::new(StubProvider))
            .expect("stub provider should register");
        let warnings = Arc::new(Mutex::new(Vec::new()));
        let best_effort_agent = AgentLoop::new(
            RecordingModel {
                requests: Arc::new(Mutex::new(Vec::new())),
            },
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("test agent should build");
        let mut config = AfterRunMemoryConfig::new("text.stub", "bench/runs");
        config.policy = AfterRunIngestPolicy::BestEffort;
        let warning_store = Arc::clone(&warnings);
        config = config.with_warning_sink(move |error| {
            warning_store
                .lock()
                .expect("warning mutex should not be poisoned")
                .push(error);
        });
        let best_effort_agent =
            AfterRunMemoryAdapter::new(Arc::new(registry), config).attach(best_effort_agent);
        let result = best_effort_agent
            .run("best effort write")
            .await
            .expect("best effort should preserve the model result");
        assert_eq!(result.final_message.content.as_deref(), Some("done"));
        assert_eq!(
            warnings
                .lock()
                .expect("warning mutex should not be poisoned")
                .len(),
            1
        );
    }
}
