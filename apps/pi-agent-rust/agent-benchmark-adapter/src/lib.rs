//! Adapter that lets the existing Pi agent loop execute benchmark steps.
//!
//! The benchmark crates stay independent from the root binary. This crate is
//! the integration boundary: it turns a structured retrieval result into a
//! bounded prompt, runs `AgentLoop`, and maps only the final assistant text
//! back to the benchmark trace.

use std::sync::{Arc, Mutex};

use pi_agent_benchmark::{
    ActionCallTrace, ActionTrace, ObservationPart, RuntimeMemoryDecisionLink, RuntimeStepLink,
    TraceErrorEntry, TraceErrorStage,
};
use pi_agent_benchmark_runner::{
    AgentExecutionError, AgentFuture, AgentStepExecutor, AgentStepInput, AgentStepOutput,
    ArtifactValidationError, ArtifactValidationFuture, ArtifactValidationInput, ArtifactValidator,
};
use pi_agent_memory::{ContentRef, MemoryHit};
use pi_agent_memory_remote::{ArtifactValidationReference, RemoteMemoryMethod};
use pi_agent_runtime::{
    MemoryCandidate, MemoryHookContext, MemoryProjection, MemoryProposal, MemoryProviderFailure,
    MemoryRetrievalSummary,
};
use pi_agent_rust::{AgentError, AgentEvent, AgentLoop, ChatModel, MemoryHookConfig, RunContext};
use sha2::{Digest, Sha256};
use uuid::Uuid;

mod longmemeval;

pub use longmemeval::LightMemLongMemEvalMethod;

const MAX_PROMPT_BYTES: usize = 128 * 1024;

/// Validates remote artifact references at the benchmark admission boundary.
///
/// The runner decides when validation is required; this adapter only maps the
/// common memory payloads to the remote provider's artifact endpoint. An empty
/// reference set completes locally, so text-only runs do not make a needless
/// network request.
pub struct RemoteArtifactValidator {
    remote: RemoteMemoryMethod,
}

impl RemoteArtifactValidator {
    pub fn new(remote: RemoteMemoryMethod) -> Self {
        Self { remote }
    }

    fn references(input: ArtifactValidationInput<'_>) -> Vec<ArtifactValidationReference> {
        let mut references = Vec::new();
        if let Some(request) = input.ingest {
            for item in &request.items {
                push_content_reference(&mut references, &item.content);
            }
        }
        if let Some(request) = input.record_ingest {
            for record in &request.records {
                push_content_reference(&mut references, &record.item.content);
                for evidence in &record.evidence {
                    references.push(ArtifactValidationReference::artifact(
                        evidence.artifact.clone(),
                    ));
                }
            }
        }
        if let Some(input) = &input.query.input {
            push_content_reference(&mut references, input);
        }
        references
    }
}

impl ArtifactValidator for RemoteArtifactValidator {
    fn validate<'a>(
        &'a mut self,
        input: ArtifactValidationInput<'a>,
        context: pi_agent_memory::OperationContext,
    ) -> ArtifactValidationFuture<'a> {
        let references = Self::references(input);
        let remote = self.remote.clone();
        Box::pin(async move {
            if references.is_empty() {
                return Ok(());
            }
            remote
                .preflight_artifacts(references, context)
                .await
                .map(|_| ())
                .map_err(|error| ArtifactValidationError::new(error.to_string()))
        })
    }
}

fn push_content_reference(references: &mut Vec<ArtifactValidationReference>, content: &ContentRef) {
    if let ContentRef::Blob {
        uri,
        media_type,
        sha256,
        size_bytes,
    } = content
    {
        references.push(ArtifactValidationReference::blob(
            uri.clone(),
            media_type.clone(),
            sha256.clone(),
            *size_bytes,
        ));
    }
}

/// Executes one benchmark step with the existing Pi agent loop.
pub struct PiAgentExecutor<M> {
    agent: AgentLoop<M>,
    events: Arc<Mutex<Vec<AgentEvent>>>,
    runtime_run_id: Option<Uuid>,
    retrieval_memory: Option<Arc<Mutex<Option<pi_agent_benchmark_runner::RetrievalOutcome>>>>,
}

impl<M> PiAgentExecutor<M>
where
    M: ChatModel,
{
    pub fn new(agent: AgentLoop<M>) -> Self {
        Self::with_runtime(agent, None)
    }

    /// Attach the executor to a benchmark operation's runtime ledger.
    pub fn new_with_runtime(agent: AgentLoop<M>, runtime_run_id: Uuid) -> Self {
        Self::with_runtime(agent, Some(runtime_run_id))
    }

    /// Attach one benchmark retrieval result to the runtime memory hook.
    ///
    /// This is deliberately a separate constructor. Existing adapters keep
    /// rendering retrieval into their prompt, while this path lets
    /// `AgentLoop` evaluate the result as a runtime-owned proposal and record
    /// the resulting memory decision in the ledger.
    pub fn new_with_runtime_and_retrieval_memory(
        agent: AgentLoop<M>,
        runtime_run_id: Uuid,
        memory_config: MemoryHookConfig,
    ) -> Self {
        let retrieval_memory = Arc::new(Mutex::new(None));
        let hook_state = Arc::clone(&retrieval_memory);
        let agent =
            agent.with_memory_hook(memory_config, move |request| {
                let hook_state = Arc::clone(&hook_state);
                Box::pin(async move {
                    let retrieval = hook_state
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .clone();
                    Ok(retrieval
                        .map(|retrieval| proposal_from_retrieval(&request.context, &retrieval)))
                })
            });
        let mut executor = Self::with_runtime(agent, Some(runtime_run_id));
        executor.retrieval_memory = Some(retrieval_memory);
        executor
    }

    fn with_runtime(agent: AgentLoop<M>, runtime_run_id: Option<Uuid>) -> Self {
        let events = Arc::new(Mutex::new(Vec::new()));
        let event_buffer = Arc::clone(&events);
        let agent = agent.with_event_sink(move |event| {
            event_buffer
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(event);
        });
        Self {
            agent,
            events,
            runtime_run_id,
            retrieval_memory: None,
        }
    }

    pub fn into_inner(self) -> AgentLoop<M> {
        self.agent
    }
}

impl<M> AgentStepExecutor for PiAgentExecutor<M>
where
    M: ChatModel,
{
    fn execute<'a>(&'a mut self, input: AgentStepInput) -> AgentFuture<'a> {
        Box::pin(async move {
            self.events
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clear();
            if let Some(state) = &self.retrieval_memory {
                *state.lock().unwrap_or_else(|error| error.into_inner()) =
                    Some(input.retrieval.clone());
            }
            let prompt = if self.retrieval_memory.is_some() {
                render_prompt_without_retrieval(&input)?
            } else {
                render_prompt(&input)?
            };
            let context = self.runtime_run_id.map(|run_id| RunContext {
                session_id: input.session_id,
                run_id,
                turn_id: Uuid::new_v4(),
            });
            let result = match context {
                Some(context) => self.agent.run_with_context(prompt, context).await,
                None => self.agent.run(prompt).await,
            };
            let runtime_link = self.take_runtime_link();
            let result = match result {
                Ok(result) => result,
                Err(error) => {
                    let tool_errors = self.take_tool_errors();
                    let action = self.take_action_trace();
                    self.clear_retrieval_memory();
                    return Err(agent_error(error)
                        .with_action(action)
                        .with_trace_errors(tool_errors)
                        .with_runtime_link(runtime_link));
                }
            };
            let errors = self.take_tool_errors();
            self.clear_retrieval_memory();
            Ok(AgentStepOutput {
                answer: result.final_message.content,
                action: self.take_action_trace(),
                errors,
                runtime_link,
                ..AgentStepOutput::default()
            })
        })
    }
}

impl<M> PiAgentExecutor<M> {
    fn clear_retrieval_memory(&self) {
        if let Some(state) = &self.retrieval_memory {
            *state.lock().unwrap_or_else(|error| error.into_inner()) = None;
        }
    }

    fn take_runtime_link(&self) -> Option<RuntimeStepLink> {
        let configured_run_id = self.runtime_run_id?;
        let events = self
            .events
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let run_context = events.iter().find_map(|event| match event {
            AgentEvent::RunStarted { context } => Some(*context),
            _ => None,
        })?;
        if run_context.run_id != configured_run_id {
            return None;
        }
        let memory_decisions = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::MemoryDecision {
                    context,
                    proposal_id,
                    event_sequence,
                    decision_kind,
                    ..
                } if *context == run_context => Some(RuntimeMemoryDecisionLink {
                    proposal_id: *proposal_id,
                    event_sequence: *event_sequence,
                    decision_kind: decision_kind.clone(),
                }),
                _ => None,
            })
            .collect::<Vec<_>>();
        let memory_decision = memory_decisions.first().cloned();
        Some(RuntimeStepLink {
            runtime_run_id: run_context.run_id,
            turn_id: run_context.turn_id,
            memory_decision,
            memory_decisions,
        })
    }

    fn take_tool_errors(&self) -> Vec<TraceErrorEntry> {
        let events = self
            .events
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolFinished {
                    tool_name,
                    content,
                    is_error: true,
                    ..
                } => Some(TraceErrorEntry {
                    stage: TraceErrorStage::Tool,
                    code: format!("tool_{tool_name}"),
                    message: content.clone(),
                    retryable: false,
                }),
                _ => None,
            })
            .collect()
    }

    fn take_action_trace(&self) -> Option<ActionTrace> {
        let events = self
            .events
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut calls = Vec::new();
        for event in events.iter() {
            let AgentEvent::ToolStarted { step, call, .. } = event else {
                continue;
            };
            let finished = events.iter().find_map(|event| match event {
                AgentEvent::ToolFinished {
                    step: finished_step,
                    call_id,
                    content,
                    details,
                    is_error,
                    terminate,
                    ..
                } if finished_step == step && call_id == &call.id => Some((
                    digest_tool_result(content, details, *is_error, *terminate),
                    *is_error,
                    *terminate,
                )),
                _ => None,
            });
            calls.push(ActionCallTrace {
                step: *step,
                call_id: call.id.clone(),
                tool_name: call.function.name.clone(),
                arguments_digest: Some(digest_text(&call.function.arguments)),
                result_digest: finished.as_ref().map(|result| result.0.clone()),
                is_error: finished.as_ref().map(|result| result.1),
                terminate: finished.as_ref().map(|result| result.2),
            });
        }
        if calls.is_empty() {
            return None;
        }
        let action_bytes = serde_json::to_vec(&calls).expect("action trace calls are serializable");
        let action_digest = Some(digest_bytes(&action_bytes));
        Some(ActionTrace {
            action_type: "tool_calls".to_owned(),
            action_digest,
            calls,
        })
    }
}

fn digest_text(value: &str) -> String {
    digest_bytes(value.as_bytes())
}

fn digest_tool_result(
    content: &str,
    details: &serde_json::Value,
    is_error: bool,
    terminate: bool,
) -> String {
    let value = serde_json::json!({
        "content": content,
        "details": details,
        "is_error": is_error,
        "terminate": terminate,
    });
    let bytes = serde_json::to_vec(&value).expect("tool result digest payload is serializable");
    digest_bytes(&bytes)
}

fn digest_bytes(value: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(value))
}

#[allow(clippy::result_large_err)]
fn render_prompt(input: &AgentStepInput) -> Result<String, AgentExecutionError> {
    render_prompt_with_retrieval(input, true)
}

#[allow(clippy::result_large_err)]
fn render_prompt_without_retrieval(input: &AgentStepInput) -> Result<String, AgentExecutionError> {
    render_prompt_with_retrieval(input, false)
}

#[allow(clippy::result_large_err)]
fn render_prompt_with_retrieval(
    input: &AgentStepInput,
    include_retrieval: bool,
) -> Result<String, AgentExecutionError> {
    let mut prompt = String::new();
    append_line(
        &mut prompt,
        &format!("benchmark sample: {}", input.sample_id),
    )?;
    append_line(&mut prompt, "observation:")?;
    for part in &input.observation.parts {
        append_line(&mut prompt, &render_observation_part(part))?;
    }

    if include_retrieval {
        append_line(&mut prompt, "retrieved memory:")?;
        if let Some(response) = &input.retrieval.response {
            for (index, hit) in response
                .hits
                .iter()
                .take(input.retrieval.query.budget.max_hits)
                .enumerate()
            {
                append_line(&mut prompt, &render_hit(index + 1, hit))?;
            }
        } else {
            append_line(
                &mut prompt,
                "(memory retrieval failed; answer from the available observation)",
            )?;
        }
        for error in &input.retrieval.errors {
            append_line(
                &mut prompt,
                &format!("memory error [{}]: {}", error.code, error.message),
            )?;
        }
    }

    if let Some(query_input) = &input.retrieval.query.input {
        append_line(&mut prompt, "current input:")?;
        append_line(&mut prompt, &render_content(query_input))?;
    }
    append_line(
        &mut prompt,
        "Respond with the best final answer. Use tools when the agent policy allows them.",
    )?;
    Ok(prompt)
}

fn proposal_from_retrieval(
    context: &MemoryHookContext,
    retrieval: &pi_agent_benchmark_runner::RetrievalOutcome,
) -> MemoryProposal {
    let hits = retrieval
        .response
        .as_ref()
        .map_or(&[][..], |response| response.hits.as_slice());
    let candidates = hits
        .iter()
        .take(context.max_candidates as usize)
        .map(memory_candidate)
        .collect();
    let method_id = hits
        .first()
        .map(|hit| hit.method_id.clone())
        .unwrap_or_else(|| "benchmark.retrieval".to_owned());

    MemoryProposal {
        proposal_id: Uuid::new_v4(),
        run_id: context.run_id,
        namespace: context.namespace.clone(),
        base_revision: context.current_revision,
        memory_revision: context.memory_revision,
        policy_revision: context.policy_revision,
        candidates,
        created_at_ms: context.now_ms,
        expires_at_ms: None,
        retrieval_summary: Some(MemoryRetrievalSummary {
            hit_count: u32::try_from(hits.len()).unwrap_or(u32::MAX),
            truncated: retrieval
                .response
                .as_ref()
                .is_some_and(|response| response.truncated),
            provider_failures: retrieval
                .errors
                .iter()
                .map(|error| MemoryProviderFailure {
                    method_id: method_id.clone(),
                    code: error.code.clone(),
                    message: error.message.clone(),
                    retryable: error.retryable,
                })
                .collect(),
            ..MemoryRetrievalSummary::default()
        }),
    }
}

fn memory_candidate(hit: &MemoryHit) -> MemoryCandidate {
    let content = hit
        .projection
        .as_ref()
        .map(render_content)
        .unwrap_or_else(|| format!("memory item {}", hit.item_id));
    let evidence_refs = hit
        .evidence_refs
        .iter()
        .map(|reference| {
            serde_json::to_value(reference).expect("evidence references are serializable")
        })
        .collect::<Vec<_>>();
    let token_cost = projection_token_cost(&content, &evidence_refs);
    let provenance = if hit.provenance.trim().is_empty() {
        hit.method_id.clone()
    } else {
        format!("{}:{}", hit.method_id, hit.provenance)
    };
    let score = hit.score.unwrap_or(0.0).clamp(0.0, 1.0);

    MemoryCandidate {
        candidate_id: hit.item_id,
        projection: MemoryProjection {
            projection_id: hit.item_id,
            source_memory_id: hit.item_id,
            source_version: hit.version,
            content,
            evidence_refs,
            provenance,
            token_cost,
        },
        score,
        confidence: score,
        expires_at_ms: None,
    }
}

fn projection_token_cost(content: &str, evidence_refs: &[serde_json::Value]) -> u32 {
    let evidence_bytes = serde_json::to_vec(evidence_refs)
        .expect("evidence references are serializable")
        .len();
    let units = content.chars().count().saturating_add(evidence_bytes);
    units
        .saturating_add(3)
        .checked_div(4)
        .unwrap_or(usize::MAX)
        .max(1)
        .min(u32::MAX as usize) as u32
}

fn render_observation_part(part: &ObservationPart) -> String {
    match part {
        ObservationPart::Text(text) => format!("- {text}"),
        ObservationPart::Artifact {
            artifact_id,
            modality,
        } => format!("- artifact reference: id={artifact_id}, modality={modality}"),
    }
}

fn render_hit(index: usize, hit: &MemoryHit) -> String {
    let projection = hit
        .projection
        .as_ref()
        .map(render_content)
        .unwrap_or_else(|| "(no inline projection; use the reference only)".to_string());
    // 保留证据类型及其位置；JSON 会转义来源文本，空值保持为 null。
    // 这里只渲染 provider 已返回的引用，不读取媒体文件。
    let reference = serde_json::json!({
        "method_id": hit.method_id,
        "record_id": hit.item_id,
        "version": hit.version,
        "score": hit.score,
        "snapshot_id": hit.snapshot_id,
        "provenance": hit.provenance,
        "evidence": hit.evidence,
        "evidence_refs": hit.evidence_refs,
    });
    format!("{index}. reference={reference}\nprojection={projection}")
}

fn render_content(content: &ContentRef) -> String {
    match content {
        ContentRef::Text { text } => text.clone(),
        ContentRef::Json { value } => value.to_string(),
        ContentRef::Blob {
            uri,
            media_type,
            sha256,
            size_bytes,
        } => format!(
            "[artifact reference uri={uri} media_type={media_type} sha256={sha256} size_bytes={size_bytes}]"
        ),
    }
}

#[allow(clippy::result_large_err)]
fn append_line(prompt: &mut String, line: &str) -> Result<(), AgentExecutionError> {
    let additional = line.len().saturating_add(1);
    if prompt.len().saturating_add(additional) > MAX_PROMPT_BYTES {
        return Err(AgentExecutionError::new(
            "prompt_too_large",
            format!("benchmark prompt exceeds {MAX_PROMPT_BYTES} bytes"),
            false,
        ));
    }
    prompt.push_str(line);
    prompt.push('\n');
    Ok(())
}

fn agent_error(error: AgentError) -> AgentExecutionError {
    let retryable = match &error {
        AgentError::Model(model_error) => model_error.is_retryable(),
        AgentError::RunTimedOut { .. } => true,
        _ => false,
    };
    AgentExecutionError::new("agent", error.to_string(), retryable)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pi_agent_benchmark::{AgentObservation, ObservationPart};
    use pi_agent_memory::{
        ArtifactRef, EvidenceLocator, EvidenceSpan, MemoryBudget, MemoryHit, MemoryQuery,
        RetrieveResponse,
    };
    use pi_agent_runtime::{EventSequence, JsonlLedger};
    use pi_agent_rust::{FunctionCall, Message, ModelError, RunContext, ToolCall};

    use super::*;

    struct FakeModel;

    impl ChatModel for FakeModel {
        fn complete<'a>(
            &'a self,
            _request: pi_agent_rust::ChatRequest<'a>,
        ) -> pi_agent_rust::ChatFuture<'a> {
            Box::pin(async { Ok(Message::assistant(Some("fake final answer".into()), None)) })
        }
    }

    struct UnknownToolModel {
        calls: AtomicUsize,
    }

    impl ChatModel for UnknownToolModel {
        fn complete<'a>(
            &'a self,
            _request: pi_agent_rust::ChatRequest<'a>,
        ) -> pi_agent_rust::ChatFuture<'a> {
            let step = self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if step == 0 {
                    Ok(Message::assistant(
                        None,
                        Some(vec![ToolCall {
                            id: "call-1".into(),
                            call_type: "function".into(),
                            function: FunctionCall {
                                name: "missing_tool".into(),
                                arguments: "{}".into(),
                            },
                        }]),
                    ))
                } else {
                    Ok(Message::assistant(Some("recovered".into()), None))
                }
            })
        }
    }

    struct FailingAfterToolModel {
        calls: AtomicUsize,
    }

    impl ChatModel for FailingAfterToolModel {
        fn complete<'a>(
            &'a self,
            _request: pi_agent_rust::ChatRequest<'a>,
        ) -> pi_agent_rust::ChatFuture<'a> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if call == 0 {
                    Ok(Message::assistant(
                        None,
                        Some(vec![ToolCall {
                            id: "call-1".into(),
                            call_type: "function".into(),
                            function: FunctionCall {
                                name: "missing_tool".into(),
                                arguments: "{}".into(),
                            },
                        }]),
                    ))
                } else {
                    Err(ModelError::EmptyResponse)
                }
            })
        }
    }

    fn input() -> AgentStepInput {
        AgentStepInput {
            sample_id: "sample-1".into(),
            observation: AgentObservation {
                parts: vec![ObservationPart::Text("visible".into())],
            },
            session_id: None,
            episode_id: None,
            terminal: true,
            terminal_reason: None,
            retrieval: pi_agent_benchmark_runner::RetrievalOutcome {
                query: MemoryQuery {
                    input: Some(ContentRef::Text {
                        text: "question".into(),
                    }),
                    namespace: "bench/test".into(),
                    scope: None,
                    subject: None,
                    episode_id: None,
                    kinds: Vec::new(),
                    modalities: Vec::new(),
                    time_range: None,
                    visibility_cutoff_ms: None,
                    graph: None,
                    budget: MemoryBudget::default(),
                },
                response: None,
                hits: Vec::new(),
                latency_ms: 1,
                errors: Vec::new(),
            },
        }
    }

    fn executor_with_events(events: Vec<AgentEvent>) -> PiAgentExecutor<FakeModel> {
        let agent = AgentLoop::new(
            FakeModel,
            pi_agent_rust::ToolRegistry::new(),
            "fake-model",
            None,
            pi_agent_rust::ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .unwrap();
        let executor = PiAgentExecutor::new(agent);
        *executor.events.lock().unwrap() = events;
        executor
    }

    fn tool_started(step: usize, call_id: &str, tool_name: &str) -> AgentEvent {
        AgentEvent::ToolStarted {
            context: RunContext::new(None),
            step,
            call: ToolCall {
                id: call_id.to_owned(),
                call_type: "function".to_owned(),
                function: FunctionCall {
                    name: tool_name.to_owned(),
                    arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
                },
            },
        }
    }

    fn tool_finished(
        step: usize,
        call_id: &str,
        tool_name: &str,
        content: &str,
        details: serde_json::Value,
        is_error: bool,
        terminate: bool,
    ) -> AgentEvent {
        AgentEvent::ToolFinished {
            context: RunContext::new(None),
            step,
            call_id: call_id.to_owned(),
            tool_name: tool_name.to_owned(),
            content: content.to_owned(),
            is_error,
            details,
            terminate,
            duration_ms: 0,
            status: if is_error {
                "failed".to_owned()
            } else {
                "succeeded".to_owned()
            },
            action_id: None,
            grant_id: None,
        }
    }

    #[test]
    fn action_trace_pairs_reused_call_ids_by_step() {
        let executor = executor_with_events(vec![
            tool_started(1, "same-call", "read_file"),
            tool_started(2, "same-call", "read_file"),
            tool_finished(
                1,
                "same-call",
                "read_file",
                "first",
                serde_json::json!({"path": "first"}),
                false,
                false,
            ),
            tool_finished(
                2,
                "same-call",
                "read_file",
                "second",
                serde_json::json!({"path": "second"}),
                false,
                true,
            ),
        ]);

        let action = executor
            .take_action_trace()
            .expect("tool calls should be captured");
        assert_eq!(action.calls.len(), 2);
        assert_eq!(action.calls[0].step, 1);
        assert_eq!(action.calls[1].step, 2);
        assert_eq!(action.calls[0].is_error, Some(false));
        assert_eq!(action.calls[1].terminate, Some(true));
        assert_ne!(
            action.calls[0].result_digest, action.calls[1].result_digest,
            "step-matched results should remain distinct"
        );
    }

    #[test]
    fn action_trace_keeps_tool_calls_without_a_finished_event() {
        let executor = executor_with_events(vec![tool_started(3, "unfinished", "read_file")]);

        let action = executor
            .take_action_trace()
            .expect("unfinished tool calls remain auditable");
        let call = &action.calls[0];
        assert_eq!(call.call_id, "unfinished");
        assert_eq!(call.result_digest, None);
        assert_eq!(call.is_error, None);
        assert_eq!(call.terminate, None);
    }

    #[test]
    fn action_digest_is_stable_and_result_digest_covers_details() {
        fn trace(details: serde_json::Value) -> ActionTrace {
            executor_with_events(vec![
                tool_started(1, "call-1", "read_file"),
                tool_finished(1, "call-1", "read_file", "same", details, false, false),
            ])
            .take_action_trace()
            .expect("tool action should be captured")
        }

        let first = trace(serde_json::json!({"bytes": 1}));
        let same = trace(serde_json::json!({"bytes": 1}));
        let changed = trace(serde_json::json!({"bytes": 2}));

        assert_eq!(first.action_digest, same.action_digest);
        assert_ne!(first.action_digest, changed.action_digest);
        assert_ne!(first.calls[0].result_digest, changed.calls[0].result_digest);
        assert_eq!(first.calls[0].is_error, Some(false));
        assert_eq!(first.calls[0].terminate, Some(false));
    }

    #[test]
    fn prompt_includes_observation_and_current_query() {
        let input = input();
        let prompt = render_prompt(&input).unwrap();
        assert!(prompt.contains("visible"));
        assert!(prompt.contains("question"));
        assert!(!prompt.contains("reference answer"));
    }

    fn memory_hit() -> MemoryHit {
        MemoryHit {
            item_id: "00000000-0000-0000-0000-000000000007".parse().unwrap(),
            source_record_ids: Vec::new(),
            version: 3,
            method_id: "video-memory".into(),
            rank: 1,
            score: Some(0.8),
            evidence: vec![EvidenceSpan::Video {
                start_ms: 1_000,
                end_ms: Some(2_000),
                frame_start: Some(30),
                frame_end: Some(60),
            }],
            evidence_refs: vec![EvidenceLocator {
                artifact: ArtifactRef {
                    artifact_id: "00000000-0000-0000-0000-000000000070".parse().unwrap(),
                    uri: "cas://episode-7.mp4".into(),
                    media_type: "video/mp4".into(),
                    sha256: "sha256:episode-7".into(),
                    bytes: 4_096,
                    codec: Some("h264".into()),
                    schema: Some("camera-recording-v1".into()),
                    stream_key: Some("camera/front".into()),
                    storage_revision: Some("store-r1".into()),
                },
                entity_path: Some("robot/gripper/object-3".into()),
                time_range: None,
                frame_start: Some(30),
                frame_end: Some(60),
                step_start: None,
                step_end: None,
                byte_start: None,
                byte_end: None,
            }],
            projection: Some(ContentRef::Blob {
                uri: "cas://episode-7.mp4".into(),
                media_type: "video/mp4".into(),
                sha256: "sha256:episode-7".into(),
                size_bytes: 4_096,
            }),
            provenance: "episode-summarizer".into(),
            snapshot_id: Some("snapshot-4".into()),
            latency_ms: 2,
        }
    }

    fn input_with_hit(hit: MemoryHit) -> AgentStepInput {
        let mut input = input();
        input.retrieval.response = Some(RetrieveResponse {
            query_id: "00000000-0000-0000-0000-000000000008".parse().unwrap(),
            snapshot_id: hit.snapshot_id.clone(),
            hits: vec![hit],
            latency_ms: 2,
            truncated: false,
            telemetry: None,
        });
        input
    }

    fn reference_in_prompt(prompt: &str) -> serde_json::Value {
        let reference = prompt
            .lines()
            .find_map(|line| line.strip_prefix("1. reference="))
            .expect("retrieved memory reference should be in the final prompt");
        serde_json::from_str(reference).unwrap()
    }

    #[test]
    fn prompt_preserves_multimodal_evidence_and_source_identity() {
        let mut hit = memory_hit();
        hit.provenance = "producer=\"episode-summarizer\"\nsource=episode-7".into();
        hit.evidence.extend([
            EvidenceSpan::Text {
                start_char: 2,
                end_char: 8,
            },
            EvidenceSpan::Image {
                x: 10,
                y: 20,
                width: 30,
                height: 40,
            },
            EvidenceSpan::Audio {
                start_ms: 500,
                end_ms: None,
            },
            EvidenceSpan::Trajectory {
                start_step: 12,
                end_step: Some(18),
            },
        ]);
        let input = input_with_hit(hit.clone());
        let prompt = render_prompt(&input).unwrap();
        let reference = reference_in_prompt(&prompt);

        assert_eq!(reference["method_id"], "video-memory");
        assert_eq!(reference["record_id"], hit.item_id.to_string());
        assert_eq!(reference["version"], 3);
        assert_eq!(reference["snapshot_id"], "snapshot-4");
        assert_eq!(reference["provenance"], hit.provenance);
        assert_eq!(
            reference["evidence"],
            serde_json::to_value(&hit.evidence).unwrap()
        );
        assert_eq!(
            reference["evidence_refs"],
            serde_json::to_value(&hit.evidence_refs).unwrap()
        );
        assert!(prompt.contains("[artifact reference uri=cas://episode-7.mp4 media_type=video/mp4 sha256=sha256:episode-7 size_bytes=4096]"));
    }

    #[test]
    fn prompt_preserves_absent_evidence_snapshot_and_projection() {
        let mut hit = memory_hit();
        hit.evidence.clear();
        hit.snapshot_id = None;
        hit.projection = None;
        let prompt = render_prompt(&input_with_hit(hit)).unwrap();
        let reference = reference_in_prompt(&prompt);

        assert_eq!(reference["evidence"], serde_json::json!([]));
        assert_eq!(reference["snapshot_id"], serde_json::Value::Null);
        assert!(prompt.contains("projection=(no inline projection; use the reference only)"));
    }

    #[test]
    fn prompt_limit_includes_memory_reference_metadata() {
        let mut hit = memory_hit();
        hit.provenance = "x".repeat(MAX_PROMPT_BYTES);
        let error = render_prompt(&input_with_hit(hit)).unwrap_err();

        assert_eq!(error.code, "prompt_too_large");
        assert!(!error.retryable);
    }

    #[test]
    fn remote_artifact_validator_collects_blob_query_inputs() {
        let mut value = input();
        value.retrieval.query.input = Some(ContentRef::Blob {
            uri: "cas://video/episode-1.mp4".to_owned(),
            media_type: "video/mp4".to_owned(),
            sha256: "sha256:episode-1".to_owned(),
            size_bytes: 42,
        });

        let references = RemoteArtifactValidator::references(ArtifactValidationInput {
            ingest: None,
            record_ingest: None,
            query: &value.retrieval.query,
        });

        assert!(matches!(
            references.as_slice(),
            [ArtifactValidationReference::Blob {
                uri,
                media_type,
                sha256,
                size_bytes,
            }] if uri == "cas://video/episode-1.mp4"
                && media_type == "video/mp4"
                && sha256 == "sha256:episode-1"
                && *size_bytes == 42
        ));
    }

    #[tokio::test]
    async fn adapter_returns_the_agent_final_message() {
        let agent = AgentLoop::new(
            FakeModel,
            pi_agent_rust::ToolRegistry::new(),
            "fake-model",
            None,
            pi_agent_rust::ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .unwrap();
        let mut executor = PiAgentExecutor::new(agent);

        let output = executor.execute(input()).await.unwrap();

        assert_eq!(output.answer.as_deref(), Some("fake final answer"));
        assert!(output.action.is_none());
        assert!(output.runtime_link.is_none());
    }

    #[tokio::test]
    async fn linked_executor_records_runtime_run_and_turn_identity() {
        let runtime_run_id = Uuid::from_u128(0xabc);
        let agent = AgentLoop::new(
            FakeModel,
            pi_agent_rust::ToolRegistry::new(),
            "fake-model",
            None,
            pi_agent_rust::ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .unwrap();
        let mut executor = PiAgentExecutor::new_with_runtime(agent, runtime_run_id);

        let output = executor.execute(input()).await.unwrap();
        let link = output
            .runtime_link
            .expect("linked execution should be traceable");
        assert_eq!(link.runtime_run_id, runtime_run_id);
        assert_ne!(link.turn_id, Uuid::nil());
        assert!(link.memory_decision.is_none());
    }

    #[tokio::test]
    async fn retrieval_memory_hook_injects_and_persists_decision() {
        let directory = tempfile::tempdir().unwrap();
        let ledger_path = directory.path().join("runtime.jsonl");
        let runtime_run_id = Uuid::from_u128(0xabc);
        let agent = AgentLoop::new(
            FakeModel,
            pi_agent_rust::ToolRegistry::new(),
            "fake-model",
            None,
            pi_agent_rust::ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .unwrap()
        .with_runtime_ledger_file(&ledger_path);
        let mut executor = PiAgentExecutor::new_with_runtime_and_retrieval_memory(
            agent,
            runtime_run_id,
            MemoryHookConfig {
                namespace: "bench/test".into(),
                ..MemoryHookConfig::default()
            },
        );

        let output = executor
            .execute(input_with_hit(memory_hit()))
            .await
            .unwrap();
        let link = output
            .runtime_link
            .expect("explicit retrieval adapter should be linked");
        let decision = link
            .memory_decision
            .expect("retrieval hit should produce a memory decision");
        assert_eq!(decision.decision_kind, "injected");

        let ledger = JsonlLedger::open(&ledger_path).unwrap();
        let events = ledger.scan_after(EventSequence::INITIAL).unwrap();
        assert!(events
            .iter()
            .any(|event| event.draft.kind.as_str() == "memory_injected"));
    }

    #[test]
    fn runtime_link_extracts_memory_decision_from_events() {
        let context = RunContext {
            session_id: None,
            run_id: Uuid::from_u128(0x111),
            turn_id: Uuid::from_u128(0x222),
        };
        let mut executor = executor_with_events(vec![
            AgentEvent::RunStarted { context },
            AgentEvent::MemoryDecision {
                context,
                step: 0,
                proposal_id: Uuid::from_u128(0x333),
                event_sequence: 9,
                decision_kind: "injected".to_owned(),
                injection_count: 1,
                total_token_cost: 4,
                skip_reason: None,
            },
            AgentEvent::MemoryDecision {
                context,
                step: 1,
                proposal_id: Uuid::from_u128(0x334),
                event_sequence: 10,
                decision_kind: "skipped".to_owned(),
                injection_count: 0,
                total_token_cost: 0,
                skip_reason: Some("no_eligible_candidate".to_owned()),
            },
        ]);
        executor.runtime_run_id = Some(context.run_id);

        let link = executor
            .take_runtime_link()
            .expect("runtime events should produce a link");
        assert_eq!(link.runtime_run_id, context.run_id);
        assert_eq!(link.turn_id, context.turn_id);
        assert_eq!(
            link.memory_decision,
            Some(RuntimeMemoryDecisionLink {
                proposal_id: Uuid::from_u128(0x333),
                event_sequence: 9,
                decision_kind: "injected".to_owned(),
            })
        );
        assert_eq!(link.memory_decisions.len(), 2);
        assert_eq!(link.memory_decisions[1].proposal_id, Uuid::from_u128(0x334));
        assert_eq!(link.memory_decisions[1].event_sequence, 10);
        assert_eq!(link.memory_decisions[1].decision_kind, "skipped");
    }

    #[tokio::test]
    async fn adapter_records_tool_failures_as_tool_stage_errors() {
        let agent = AgentLoop::new(
            UnknownToolModel {
                calls: AtomicUsize::new(0),
            },
            pi_agent_rust::ToolRegistry::new(),
            "fake-model",
            None,
            pi_agent_rust::ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .unwrap();
        let mut executor = PiAgentExecutor::new(agent);

        let output = executor.execute(input()).await.unwrap();

        assert_eq!(output.answer.as_deref(), Some("recovered"));
        assert_eq!(output.errors.len(), 1);
        assert_eq!(output.errors[0].stage, TraceErrorStage::Tool);
        assert_eq!(output.errors[0].code, "tool_missing_tool");
        let action = output.action.expect("tool action should be recorded");
        assert_eq!(action.action_type, "tool_calls");
        assert_eq!(action.calls.len(), 1);
        assert_eq!(action.calls[0].call_id, "call-1");
        assert_eq!(action.calls[0].tool_name, "missing_tool");
        assert_eq!(action.calls[0].is_error, Some(true));
        assert!(action.calls[0]
            .arguments_digest
            .as_deref()
            .is_some_and(|digest| digest.starts_with("sha256:")));
        assert!(action
            .action_digest
            .as_deref()
            .is_some_and(|digest| digest.starts_with("sha256:")));
    }

    #[tokio::test]
    async fn adapter_preserves_tool_errors_when_the_agent_fails_afterward() {
        let agent = AgentLoop::new(
            FailingAfterToolModel {
                calls: AtomicUsize::new(0),
            },
            pi_agent_rust::ToolRegistry::new(),
            "fake-model",
            None,
            pi_agent_rust::ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .unwrap();
        let mut executor = PiAgentExecutor::new(agent);

        let error = executor.execute(input()).await.unwrap_err();

        assert_eq!(error.trace_errors.len(), 1);
        assert_eq!(error.trace_errors[0].stage, TraceErrorStage::Tool);
        assert_eq!(error.trace_errors[0].code, "tool_missing_tool");
        let action = error.action.expect("partial tool action should be kept");
        assert_eq!(action.calls.len(), 1);
        assert_eq!(action.calls[0].is_error, Some(true));
    }
}
