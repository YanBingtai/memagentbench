use super::*;
use crate::{
    message::{FunctionCall, ToolCall, ToolDefinition},
    model::{ChatFuture, ChatRequest, DeltaCallback, ModelError, ReasoningDeltaCallback},
    session::SessionStore,
    skills::Skill,
    tools::{
        Tool, ToolConcurrency, ToolContext, ToolError, ToolExecutionContext, ToolExposure,
        ToolFuture, ToolReplayPolicy, ToolResult, ToolResultFuture, ToolUpdate,
    },
};
use pi_agent_runtime::{EventSequence, JsonlLedger, MemoryCandidate, MemoryProjection};
use serde_json::json;
use std::{
    collections::VecDeque,
    fs,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tempfile::tempdir;
use tokio::sync::Notify;

#[test]
fn memory_context_keeps_source_identity_and_evidence_references() {
    let context = format_memory_context(&[MemoryInjection {
        candidate_id: Uuid::from_u128(1),
        projection: MemoryProjection {
            projection_id: Uuid::from_u128(2),
            source_memory_id: Uuid::from_u128(3),
            source_version: 4,
            content: "视频中的红色方块".to_owned(),
            evidence_refs: vec![json!({
                "artifact_id": "artifact-7",
                "uri": "cas://episode-7.mp4",
                "frame_start": 360,
                "step_start": 30
            })],
            provenance: "video-memory:episode-7".to_owned(),
            token_cost: 12,
        },
    }]);

    assert!(context.contains("source_memory_id=00000000-0000-0000-0000-000000000003"));
    assert!(context.contains("version=4"));
    assert!(context.contains("artifact-7"));
    assert!(context.contains("frame_start"));
    assert!(context.contains("step_start"));
}

#[test]
fn unknown_runtime_action_has_a_distinct_run_outcome() {
    let error = AgentError::RuntimeActionUnknown {
        action_id: Uuid::new_v4(),
        tool_name: "write_file".to_string(),
        reason: "executor disconnected".to_string(),
    };

    assert_eq!(run_outcome_for_error(&error), RunOutcome::Unknown);
    assert_eq!(
        run_outcome_for_error(&AgentError::RunCancelled),
        RunOutcome::Cancelled
    );
    assert_eq!(
        run_outcome_for_error(&AgentError::EmptyPrompt),
        RunOutcome::Failed
    );
}

#[test]
fn reconciled_tool_results_keep_their_terminal_status() {
    let result = ToolResult::text("confirmed").with_details(json!({
        "status": "reconciled",
        "outcome": "succeeded",
    }));

    assert_eq!(tool_result_status(&result), "reconciled");
}

use uuid::Uuid;
use wiremock::{
    matchers::{body_string_contains, method},
    Mock, MockServer, ResponseTemplate,
};

fn test_config(base_url: String) -> Config {
    Config {
        base_url,
        model: "test-model".to_string(),
        max_steps: 2,
        request_timeout_secs: 5,
        ..Config::default()
    }
}

fn streaming_response(message: Message) -> ResponseTemplate {
    streaming_response_with_optional_usage(message, None)
}

fn streaming_response_with_usage(message: Message, usage: Value) -> ResponseTemplate {
    streaming_response_with_optional_usage(message, Some(usage))
}

fn streaming_response_with_optional_usage(
    message: Message,
    usage: Option<Value>,
) -> ResponseTemplate {
    let mut delta = serde_json::to_value(message.clone()).expect("message should serialize");
    if let Some(tool_calls) = delta
        .get_mut("tool_calls")
        .and_then(|value| value.as_array_mut())
    {
        for (index, tool_call) in tool_calls.iter_mut().enumerate() {
            tool_call["index"] = json!(index);
        }
    }
    let finish_reason = if message
        .tool_calls
        .as_ref()
        .is_some_and(|calls| !calls.is_empty())
    {
        "tool_calls"
    } else {
        "stop"
    };
    let mut event = json!({
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    });
    if let Some(usage) = usage {
        event["usage"] = usage;
    }
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(format!(
            "data: {}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&event).expect("stream event should serialize")
        ))
}

#[test]
fn truncates_old_turns_without_splitting_tool_results() {
    let messages = vec![
        Message::system("system rule"),
        Message::user("old question"),
        Message::assistant(Some("old answer".to_string()), None),
        Message::user("current question"),
        Message::assistant(
            None,
            Some(vec![ToolCall {
                id: "call-1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: r#"{"path":"Cargo.toml"}"#.to_string(),
                },
            }]),
        ),
        Message::tool("call-1", "file content"),
        Message::assistant(Some("current answer".to_string()), None),
    ];

    let truncated =
        truncate_context(&messages, 2, usize::MAX).expect("latest complete turn should fit");

    assert_eq!(truncated.len(), 5);
    assert_eq!(truncated[0].role, "system");
    assert_eq!(truncated[1].content.as_deref(), Some("current question"));
    assert!(truncated[2].tool_calls.is_some());
    assert_eq!(truncated[3].role, "tool");
    assert_eq!(truncated[4].content.as_deref(), Some("current answer"));
    assert!(!truncated
        .iter()
        .any(|message| message.content.as_deref() == Some("old question")));
}

#[test]
fn context_byte_budget_rejects_an_oversized_latest_tool_turn() {
    let messages = vec![
        Message::system("system rule"),
        Message::user("old question"),
        Message::assistant(Some("old answer".to_string()), None),
        Message::user("current question"),
        Message::assistant(None, Some(vec![tool_call("read_file", "call-1")])),
        Message::tool("call-1", "current tool result"),
        Message::assistant(Some("current answer".to_string()), None),
    ];
    let latest_turn_bytes = messages[3..].iter().map(message_size).sum::<usize>();

    let error = truncate_context(&messages, 128, latest_turn_bytes - 1)
        .expect_err("an oversized latest turn must not exceed the byte budget");

    assert!(matches!(
        error,
        AgentError::ContextTooLarge {
            limit_bytes,
            required_bytes
        } if limit_bytes == latest_turn_bytes - 1 && required_bytes > limit_bytes
    ));
}

#[test]
fn context_byte_budget_rejects_oversized_system_messages() {
    let messages = vec![Message::system("system instruction that is too large")];

    let error = truncate_context(&messages, 128, message_size(&messages[0]) - 1)
        .expect_err("system context must respect the byte budget");

    assert!(matches!(error, AgentError::ContextTooLarge { .. }));
}

#[tokio::test]
async fn context_compactor_recovers_an_oversized_request() {
    let invocations = Arc::new(AtomicUsize::new(0));
    let observed_invocations = Arc::clone(&invocations);
    let agent = AgentLoop::new_with_policy(
        FakeChatModel {
            response: Message::assistant(Some("compacted answer".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            max_steps: 1,
            max_context_bytes: 128,
            ..RunPolicy::default()
        },
    )
    .expect("compaction agent should be valid")
    .with_context_compactor(move |request| {
        observed_invocations.fetch_add(1, Ordering::SeqCst);
        assert_eq!(request.reason, ContextCompactionReason::ByteBudget);
        assert_eq!(request.step, 1);
        Box::pin(async { Ok(vec![Message::user("compacted context")]) })
    });

    let result = agent
        .run("x".repeat(1_024))
        .await
        .expect("the compactor should produce a request within budget");

    assert_eq!(
        result.final_message.content.as_deref(),
        Some("compacted answer")
    );
    assert_eq!(invocations.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn context_compactor_rejects_unpaired_tool_calls() {
    let agent = AgentLoop::new_with_policy(
        FakeChatModel {
            response: Message::assistant(Some("must not run".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            max_steps: 1,
            max_context_bytes: 512,
            ..RunPolicy::default()
        },
    )
    .expect("compaction agent should be valid")
    .with_context_compactor(|_request| {
        Box::pin(async {
            Ok(vec![
                Message::user("compacted"),
                Message::assistant(None, Some(vec![tool_call("read_file", "call-1")])),
            ])
        })
    });

    let error = agent
        .run("x".repeat(1_024))
        .await
        .expect_err("compaction must preserve tool-call/result pairing");

    assert!(matches!(
        error,
        AgentError::InvalidContext { reason }
            if reason.contains("missing results")
    ));
}

#[test]
fn rejects_duplicate_tool_call_ids_before_runtime_admission() {
    let calls = vec![
        tool_call("read_file", "call-1"),
        tool_call("read_file", "call-1"),
    ];

    let error =
        validate_tool_call_batch(2, &calls).expect_err("duplicate call IDs should be rejected");

    assert!(matches!(
        error,
        AgentError::InvalidToolCall {
            step: 2,
            index: 1,
            reason,
        } if reason.contains("duplicate call id")
    ));
}

#[tokio::test]
async fn returns_non_object_tool_arguments_as_a_tool_result() {
    let mut call = tool_call("echo_tool", "call-1");
    call.function.arguments = "[]".to_string();

    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![call])),
        Message::assistant(Some("recovered".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("echo tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("invalid argument agent should be valid");

    let result = agent
        .run("recover from invalid arguments")
        .await
        .expect("invalid arguments should be reported to the model");

    assert_eq!(result.final_message.content.as_deref(), Some("recovered"));
    let tool_message = result
        .messages
        .iter()
        .find(|message| message.role == "tool")
        .expect("invalid arguments should produce a tool message");
    assert_eq!(tool_message.tool_call_id.as_deref(), Some("call-1"));
    assert_eq!(
        tool_message.content.as_deref(),
        Some("tool arguments must be a JSON object")
    );
}

#[tokio::test]
async fn dispatches_parallel_read_tools_concurrently_and_preserves_order() {
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(
            None,
            Some(vec![
                tool_call("parallel_sleep", "call-1"),
                tool_call("parallel_sleep", "call-2"),
            ]),
        ),
        Message::assistant(Some("parallel complete".to_string()), None),
    ])));
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools
        .register(ParallelSleepTool {
            active: active.clone(),
            max_active: max_active.clone(),
        })
        .expect("parallel tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("parallel agent should be valid");

    let result = agent
        .run("run both read tools")
        .await
        .expect("parallel tools should complete");

    assert_eq!(max_active.load(Ordering::SeqCst), 2);
    let tool_messages = result
        .messages
        .iter()
        .filter(|message| message.role == "tool")
        .collect::<Vec<_>>();
    assert_eq!(tool_messages.len(), 2);
    assert_eq!(tool_messages[0].tool_call_id.as_deref(), Some("call-1"));
    assert_eq!(tool_messages[1].tool_call_id.as_deref(), Some("call-2"));
}

#[tokio::test]
async fn parallel_worker_panic_closes_admitted_runtime_actions() {
    let directory = tempfile::tempdir().expect("runtime ledger directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
        None,
        Some(vec![
            tool_call("panic_parallel", "call-1"),
            tool_call("panic_parallel", "call-2"),
        ]),
    )])));
    let mut tools = ToolRegistry::new();
    tools
        .register(PanicParallelTool)
        .expect("panic tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("panic agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone());

    let error = agent
        .run("exercise the parallel panic path")
        .await
        .expect_err("a panicking worker should fail the run");
    assert!(error.to_string().contains("parallel tool worker failed"));

    let ledger = std::fs::read_to_string(ledger_path).expect("runtime ledger should exist");
    assert_eq!(ledger.matches("\"kind\":\"action_admitted\"").count(), 2);
    assert_eq!(ledger.matches("\"kind\":\"action_unknown\"").count(), 2);
}

#[derive(Debug)]
struct FakeChatModel {
    response: Message,
}

impl ChatModel for FakeChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        let response = self.response.clone();
        Box::pin(async move { Ok(response) })
    }
}

#[derive(Debug)]
struct FailingChatModel;

impl ChatModel for FailingChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        Box::pin(async { Err(ModelError::EmptyResponse) })
    }
}

#[derive(Debug)]
struct FailingAfterToolChatModel {
    calls: Arc<AtomicUsize>,
}

impl ChatModel for FailingAfterToolChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Box::pin(async {
                Ok(Message::assistant(
                    None,
                    Some(vec![tool_call("echo_tool", "call-1")]),
                ))
            })
        } else {
            Box::pin(async { Err(ModelError::EmptyResponse) })
        }
    }
}

#[derive(Debug)]
struct ReasoningChatModel;

impl ChatModel for ReasoningChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        Box::pin(async { Ok(Message::assistant(Some("answer".to_string()), None)) })
    }

    fn stream_with_reasoning<'a>(
        &'a self,
        _request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
        on_reasoning: ReasoningDeltaCallback<'a>,
    ) -> ChatFuture<'a> {
        on_reasoning("thinking");
        on_delta("answer");
        Box::pin(async {
            let mut message = Message::assistant(Some("answer".to_string()), None);
            message.reasoning_content = Some("thinking".to_string());
            Ok(message)
        })
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct RetryChatModel {
    calls: Arc<AtomicUsize>,
    response: Message,
}

impl ChatModel for RetryChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
        let response = self.response.clone();
        Box::pin(async move {
            if attempt == 0 {
                Err(ModelError::Api {
                    status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
                    message: "temporary provider outage".to_string(),
                })
            } else {
                Ok(response)
            }
        })
    }
}

#[derive(Debug)]
struct PartialRetryChatModel {
    calls: Arc<AtomicUsize>,
}

impl ChatModel for PartialRetryChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        Box::pin(async { Err(ModelError::StreamIncomplete) })
    }

    fn stream_with_reasoning<'a>(
        &'a self,
        _request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
        on_reasoning: ReasoningDeltaCallback<'a>,
    ) -> ChatFuture<'a> {
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
        if attempt == 0 {
            on_reasoning("draft");
            on_delta("Hel");
            Box::pin(async { Err(ModelError::StreamIncomplete) })
        } else {
            on_delta("Hello");
            Box::pin(async { Ok(Message::assistant(Some("Hello".to_string()), None)) })
        }
    }
}

#[derive(Debug)]
struct GatedStreamingChatModel {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

impl ChatModel for GatedStreamingChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        Box::pin(async { Ok(Message::assistant(Some("partial".to_string()), None)) })
    }

    fn stream_with_reasoning<'a>(
        &'a self,
        _request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
        _on_reasoning: ReasoningDeltaCallback<'a>,
    ) -> ChatFuture<'a> {
        on_delta("partial");
        self.started.notify_one();
        let release = Arc::clone(&self.release);
        Box::pin(async move {
            release.notified().await;
            Ok(Message::assistant(Some("partial".to_string()), None))
        })
    }
}

#[derive(Debug)]
struct SequenceChatModel {
    responses: Arc<Mutex<VecDeque<Message>>>,
}

impl ChatModel for SequenceChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        let response = self
            .responses
            .lock()
            .expect("response queue should not be poisoned")
            .pop_front()
            .expect("test response queue should contain a response");
        Box::pin(async move { Ok(response) })
    }
}

#[derive(Debug)]
struct RecordingChatModel {
    response: Message,
    requests: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl ChatModel for RecordingChatModel {
    fn complete<'a>(&'a self, request: ChatRequest<'a>) -> ChatFuture<'a> {
        self.requests
            .lock()
            .expect("request mutex should not be poisoned")
            .push(request.messages.to_vec());
        let response = self.response.clone();
        Box::pin(async move { Ok(response) })
    }
}

#[derive(Debug)]
struct SlowChatModel {
    delay: Duration,
    response: Message,
}

#[derive(Debug)]
struct BlockingChatModel {
    started: Arc<Notify>,
    release: Arc<Notify>,
    response: Message,
}

impl ChatModel for BlockingChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        let started = Arc::clone(&self.started);
        let release = Arc::clone(&self.release);
        let response = self.response.clone();
        Box::pin(async move {
            started.notify_one();
            release.notified().await;
            Ok(response)
        })
    }
}

impl ChatModel for SlowChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        let delay = self.delay;
        let response = self.response.clone();
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            Ok(response)
        })
    }
}

#[derive(Debug)]
struct SlowTool;

impl Tool for SlowTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "slow_tool",
            "A test tool that completes after a delay.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok::<String, ToolError>("done".to_string())
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
struct ParallelSleepTool {
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct PanicParallelTool;

impl Tool for PanicParallelTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "panic_parallel",
            "A test tool that panics inside a parallel worker.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("unreachable".to_string()) })
    }

    fn execute_result_with_context<'a>(
        &'a self,
        _arguments: serde_json::Value,
        _context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(async { panic!("intentional parallel worker panic") })
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
}

impl Tool for ParallelSleepTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "parallel_sleep",
            "A read-only tool used to verify parallel dispatch.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("done".to_string()) })
    }

    fn execute_result_with_context<'a>(
        &'a self,
        _arguments: serde_json::Value,
        _context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        let active = Arc::clone(&self.active);
        let max_active = Arc::clone(&self.max_active);
        Box::pin(async move {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            max_active.fetch_max(current, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(100)).await;
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(ToolResult::text("done"))
        })
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
}

#[derive(Debug)]
struct CancelInsideTool;

impl Tool for CancelInsideTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "cancel_inside_tool",
            "Cancel the caller while the tool is running.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("unused".to_string()) })
    }

    fn execute_result_with_context<'a>(
        &'a self,
        _arguments: serde_json::Value,
        context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(async move {
            context.cancellation_token().cancel();
            std::future::pending::<Result<ToolResult, ToolError>>().await
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
struct ToolTimeoutTool;

impl Tool for ToolTimeoutTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "tool_timeout_tool",
            "A test tool that exceeds the per-tool timeout.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async {
            tokio::time::sleep(Duration::from_millis(1_100)).await;
            Ok::<String, ToolError>("too late".to_string())
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
struct ImmediateTool;

impl Tool for ImmediateTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "echo_tool",
            "Return a stable test result.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("runtime-ok".to_string()) })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
struct UnknownRiskTool;

impl Tool for UnknownRiskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "unknown_risk_tool",
            "A tool whose risk class is intentionally unspecified.",
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("must not execute".to_string()) })
    }
}

#[derive(Debug)]
struct ErrorResultTool;

impl Tool for ErrorResultTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "error_tool",
            "Return a structured tool error.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("legacy error".to_string()) })
    }

    fn execute_result<'a>(&'a self, _arguments: serde_json::Value) -> ToolResultFuture<'a> {
        Box::pin(async {
            Ok(ToolResult::error("known failure").with_details(json!({"kind": "test"})))
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
struct UncertainResultTool;

impl Tool for UncertainResultTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "uncertain_tool",
            "Return a result whose external outcome is uncertain.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("uncertain".to_string()) })
    }

    fn execute_result<'a>(&'a self, _arguments: serde_json::Value) -> ToolResultFuture<'a> {
        Box::pin(async {
            Ok(ToolResult::error("command timed out").with_details(json!({"status": "timed_out"})))
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::Reversible
    }
}

#[derive(Debug)]
struct CountingValidationTool {
    validations: Arc<AtomicUsize>,
}

impl Tool for CountingValidationTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "counting_validation_tool",
            "A test tool that records argument validation calls.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn validate_arguments(&self, _arguments: &serde_json::Value) -> Result<(), ToolError> {
        self.validations.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("validated".to_string()) })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
struct HiddenModelTool {
    executions: Arc<AtomicUsize>,
}

impl Tool for HiddenModelTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "hidden_model_tool",
            "A trusted caller-only test tool.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Hidden
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        self.executions.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok::<String, ToolError>("should not execute".to_string()) })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
struct ProgressTool;

impl Tool for ProgressTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "progress_tool",
            "Emit one progress update before completing.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("legacy".to_string()) })
    }

    fn execute_result_with_context<'a>(
        &'a self,
        _arguments: serde_json::Value,
        context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(async move {
            context.update(ToolUpdate::text("halfway").with_details(json!({"progress": 0.5})));
            Ok(ToolResult::text("done"))
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
struct StructuredTool;

impl Tool for StructuredTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "structured_tool",
            "Return structured output and terminate the run.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("legacy".to_string()) })
    }

    fn execute_result<'a>(&'a self, _arguments: serde_json::Value) -> ToolResultFuture<'a> {
        Box::pin(async {
            Ok(ToolResult::text("structured")
                .with_details(json!({"source": "test"}))
                .terminating())
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

fn tool_call(name: &str, id: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: "{}".to_string(),
        },
    }
}

fn recovery_memory_hook(request: MemoryHookRequest) -> MemoryHookFuture {
    Box::pin(async move {
        Ok(Some(MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: request.context.run_id,
            namespace: request.context.namespace.clone(),
            base_revision: request.context.current_revision,
            memory_revision: request.context.memory_revision,
            policy_revision: request.context.policy_revision,
            candidates: vec![MemoryCandidate {
                candidate_id: Uuid::new_v4(),
                projection: MemoryProjection {
                    projection_id: Uuid::new_v4(),
                    source_memory_id: Uuid::new_v4(),
                    source_version: 1,
                    content: "recovered memory".to_owned(),
                    evidence_refs: Vec::new(),
                    provenance: "recovery-test".to_owned(),
                    token_cost: 4,
                },
                score: 1.0,
                confidence: 1.0,
                expires_at_ms: Some(request.context.now_ms.saturating_add(1_000)),
            }],
            created_at_ms: request.context.now_ms,
            expires_at_ms: Some(request.context.now_ms.saturating_add(1_000)),
            retrieval_summary: None,
        }))
    })
}

#[tokio::test]
async fn before_tool_hook_can_block_a_call_without_runtime_admission() {
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
        Message::assistant(Some("blocked result handled".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("echo tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("hook agent should be valid")
    .with_before_tool_hook(|context| {
        Box::pin(async move {
            assert_eq!(context.call.function.name, "echo_tool");
            Ok(BeforeToolDecision::Block {
                reason: "approval required".to_string(),
            })
        })
    });

    let result = agent.run("call echo").await.expect("run should succeed");

    assert_eq!(result.messages.len(), 4);
    assert_eq!(result.messages[2].role, "tool");
    assert!(result.messages[2]
        .content
        .as_deref()
        .is_some_and(|content| content.contains("approval required")));
    let metadata = result.messages[2]
        .tool_result
        .as_ref()
        .expect("blocked tool should retain structured metadata");
    assert_eq!(metadata.details["status"], "rejected");
    assert!(!metadata.terminate);
}

#[tokio::test]
async fn before_tool_hook_can_terminate_a_tool_batch_before_admission() {
    let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
        None,
        Some(vec![
            tool_call("echo_tool", "call-1"),
            tool_call("echo_tool", "call-2"),
        ]),
    )])));
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("echo tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("hook agent should be valid")
    .with_event_sink({
        let events = events.clone();
        move |event| {
            events
                .lock()
                .expect("event mutex should not poison")
                .push(event)
        }
    })
    .with_before_tool_hook(|_context| {
        Box::pin(async {
            Ok(BeforeToolDecision::Terminate {
                reason: "approval was denied".to_string(),
            })
        })
    });

    let error = agent
        .run("call echo")
        .await
        .expect_err("termination should stop the run");
    assert!(matches!(
        error,
        AgentError::ToolTerminated { reason } if reason.contains("approval was denied")
    ));

    let events = events.lock().expect("event mutex should not poison");
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolFinished {
            call_id,
            status,
            terminate: true,
            action_id: None,
            grant_id: None,
            ..
        } if call_id == "call-1" && status == "rejected"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolFinished {
            call_id,
            status,
            action_id: None,
            grant_id: None,
            ..
        } if call_id == "call-2" && status == "cancelled"
    )));
}

#[tokio::test]
async fn before_tool_hook_obeys_the_run_deadline() {
    let directory = tempdir().expect("temporary directory should exist");
    let session = SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
        .await
        .expect("session should open");
    let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
        None,
        Some(vec![tool_call("echo_tool", "call-1")]),
    )])));
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("echo tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("hook agent should be valid")
    .with_session(session.clone())
    .with_before_tool_hook(|_context| {
        Box::pin(async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(BeforeToolDecision::Allow)
        })
    });

    let error = agent
        .run_with_timeout("call echo", Duration::from_millis(10))
        .await
        .expect_err("a hanging before hook should exceed the run deadline");

    assert!(matches!(error, AgentError::RunTimedOut { .. }));
    let records = session.load().await.expect("session should load");
    assert_eq!(records.len(), 3);
    assert!(records[2]
        .message
        .content
        .as_deref()
        .is_some_and(|content| content.contains("before_tool_call hook timed out")));
}

#[tokio::test]
async fn invalid_tool_arguments_are_rejected_before_hooks_and_admission() {
    let directory = tempdir().expect("temporary directory should exist");
    let mut invalid_call = tool_call("write_file", "call-1");
    invalid_call.function.arguments = json!({
        "path": "preflight.txt",
        "content": 42
    })
    .to_string();
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![invalid_call])),
        Message::assistant(Some("handled".to_string()), None),
    ])));
    let before_calls = Arc::new(AtomicUsize::new(0));
    let config = Config {
        workspace: directory.path().to_path_buf(),
        ..Config::default()
    };
    let ledger_path = directory.path().join("runtime.jsonl");
    let mut tools = ToolRegistry::new();
    tools
        .register_write_file(ToolContext::from_config(&config))
        .expect("write tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("preflight agent should be valid")
    .with_before_tool_hook({
        let before_calls = before_calls.clone();
        move |_context| {
            before_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(BeforeToolDecision::Allow) })
        }
    })
    .with_runtime_ledger_file(ledger_path.clone());

    let result = agent
        .run("write with invalid arguments")
        .await
        .expect("invalid tool arguments should become a tool error");

    assert_eq!(before_calls.load(Ordering::SeqCst), 0);
    assert!(result.messages[2]
        .content
        .as_deref()
        .is_some_and(|content| content.contains("invalid tool arguments")));
    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable");
    assert!(!events
        .iter()
        .any(|event| event.draft.kind.as_str() == "action_admitted"));
}

#[tokio::test]
async fn runtime_admission_rejections_are_structured_before_model_retry() {
    let directory = tempdir().expect("temporary directory should exist");
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("unknown_risk_tool", "call-1")])),
        Message::assistant(Some("handled".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(UnknownRiskTool)
        .expect("unknown-risk tool should register");
    let ledger_path = directory.path().join("runtime.jsonl");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("admission rejection agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone());

    let result = agent
        .run("call a tool without a risk declaration")
        .await
        .expect("admission rejection should be returned to the model");

    let metadata = result.messages[2]
        .tool_result
        .as_ref()
        .expect("rejected call should have structured metadata");
    assert_eq!(metadata.details["status"], "rejected");
    assert_eq!(metadata.details["executed"], false);
    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    assert!(ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable")
        .is_empty());
}

#[tokio::test]
async fn prepared_tool_call_reuses_validation_for_execution() {
    let validations = Arc::new(AtomicUsize::new(0));
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(
            None,
            Some(vec![tool_call("counting_validation_tool", "call-1")]),
        ),
        Message::assistant(Some("finished".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(CountingValidationTool {
            validations: validations.clone(),
        })
        .expect("counting tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("prepared-call agent should be valid");

    let result = agent
        .run("use the counting tool")
        .await
        .expect("prepared tool call should complete");

    assert_eq!(result.final_message.content.as_deref(), Some("finished"));
    assert_eq!(validations.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn model_cannot_invoke_a_hidden_tool_by_name() {
    let executions = Arc::new(AtomicUsize::new(0));
    let hidden_call = tool_call("hidden_model_tool", "call-hidden");
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![hidden_call])),
        Message::assistant(Some("recovered".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(HiddenModelTool {
            executions: executions.clone(),
        })
        .expect("hidden tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("hidden-tool agent should be valid");

    let result = agent
        .run("try the hidden tool")
        .await
        .expect("hidden call should become a recoverable tool error");

    assert_eq!(executions.load(Ordering::SeqCst), 0);
    assert!(result.messages.iter().any(|message| {
        message.role == "tool"
            && message
                .content
                .as_deref()
                .is_some_and(|content| content.contains("not available to the model"))
    }));
}

#[tokio::test]
async fn truncated_tool_calls_are_persisted_as_unexecuted_errors() {
    let mut truncated =
        Message::assistant(None, Some(vec![tool_call("read_file", "call-truncated")]));
    truncated.stop_reason = Some("length".to_string());
    let responses = Arc::new(Mutex::new(VecDeque::from([
        truncated,
        Message::assistant(Some("recovered".to_string()), None),
    ])));
    let directory = tempdir().expect("temporary directory should exist");
    let config = Config {
        workspace: directory.path().to_path_buf(),
        ..Config::default()
    };
    let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
        .expect("read tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("truncation agent should be valid");

    let result = agent
        .run("recover a truncated tool call")
        .await
        .expect("model should recover after synthetic tool error");

    assert_eq!(result.final_message.content.as_deref(), Some("recovered"));
    assert_eq!(result.messages[1].stop_reason.as_deref(), Some("length"));
    let tool = &result.messages[2];
    assert!(tool
        .content
        .as_deref()
        .is_some_and(|content| content.contains("truncated")));
    assert_eq!(
        tool.tool_result.as_ref().map(|result| &result.details),
        Some(&json!({"status": "truncated", "executed": false}))
    );
}

#[tokio::test]
async fn content_filtered_responses_stop_the_run_explicitly() {
    let mut filtered = Message::assistant(Some("blocked".to_string()), None);
    filtered.stop_reason = Some("content_filter".to_string());
    let agent = AgentLoop::new(
        FakeChatModel { response: filtered },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("content-filter agent should be valid");

    let error = agent
        .run("filtered")
        .await
        .expect_err("content-filtered response should terminate the run");

    assert!(matches!(
        error,
        AgentError::ModelTerminated { reason } if reason == "content_filter"
    ));
}

#[tokio::test]
async fn model_error_finish_reason_is_terminal_failure() {
    let mut failed = Message::assistant(Some("provider failed".to_string()), None);
    failed.stop_reason = Some("error".to_string());
    let agent = AgentLoop::new(
        FakeChatModel { response: failed },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("model-error agent should be valid");

    let error = agent
        .run("handle provider error")
        .await
        .expect_err("provider error finish reason must fail the run");

    assert!(matches!(
        error,
        AgentError::ModelTerminated { reason } if reason == "error"
    ));
}

#[tokio::test]
async fn model_aborted_tool_calls_are_closed_without_execution() {
    let directory = tempdir().expect("temporary directory should exist");
    let session = SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
        .await
        .expect("session should open");
    let mut aborted = Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")]));
    aborted.stop_reason = Some("aborted".to_string());
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("echo tool should register");
    let agent = AgentLoop::new(
        FakeChatModel { response: aborted },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("aborted tool agent should be valid")
    .with_session(session.clone());

    let error = agent
        .run("handle aborted tool call")
        .await
        .expect_err("aborted model output must fail the run");
    assert!(matches!(
        error,
        AgentError::ModelTerminated { reason } if reason == "aborted"
    ));

    let records = session.load().await.expect("session should load");
    assert_eq!(records.len(), 3);
    let tool_result = records[2]
        .message
        .tool_result
        .as_ref()
        .expect("aborted tool call should have a synthetic result");
    assert_eq!(tool_result.details["status"], "aborted");
    assert_eq!(tool_result.details["executed"], false);
    assert!(tool_result.is_error);
}

#[tokio::test]
async fn content_filter_closes_tool_calls_before_stopping() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let session = SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
        .await
        .expect("session should open");
    let mut filtered = Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")]));
    filtered.stop_reason = Some("content_filter".to_string());
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("echo tool should register");
    let agent = AgentLoop::new(
        FakeChatModel { response: filtered },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("content-filter tool agent should be valid")
    .with_session(session.clone())
    .with_runtime_ledger_file(ledger_path.clone());

    let error = agent
        .run("filtered tool call")
        .await
        .expect_err("content-filtered tool calls should terminate the run");
    assert!(matches!(
        error,
        AgentError::ModelTerminated { reason } if reason == "content_filter"
    ));

    let records = session.load().await.expect("session should load");
    assert_eq!(records.len(), 3);
    let tool_result = records[2]
        .message
        .tool_result
        .as_ref()
        .expect("content-filtered call should have a result");
    assert_eq!(tool_result.details["status"], "rejected");
    assert_eq!(tool_result.details["executed"], false);

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    assert!(ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable")
        .is_empty());
}

#[tokio::test]
async fn after_tool_hook_can_replace_a_tool_result() {
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
        Message::assistant(Some("redacted result handled".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("echo tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("hook agent should be valid")
    .with_after_tool_hook(|context| {
        Box::pin(async move {
            assert_eq!(context.content, "runtime-ok");
            Ok(ToolResultOverride {
                content: Some("redacted".to_string()),
                is_error: Some(false),
                details: None,
                terminate: None,
            })
        })
    });

    let result = agent.run("call echo").await.expect("run should succeed");

    assert_eq!(result.messages[2].content.as_deref(), Some("redacted"));
}

#[tokio::test]
async fn after_tool_hook_can_terminate_a_tool_result() {
    let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
        None,
        Some(vec![tool_call("echo_tool", "call-1")]),
    )])));
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("immediate tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("hook agent should be valid")
    .with_after_tool_hook(|_context| {
        Box::pin(async {
            Ok(ToolResultOverride {
                content: Some("approved".to_string()),
                is_error: Some(false),
                details: Some(json!({"source": "after-hook"})),
                terminate: Some(true),
            })
        })
    });

    let result = agent
        .run("call echo")
        .await
        .expect("terminating hook result should complete the run");

    assert_eq!(result.final_message.role, "tool");
    assert_eq!(result.final_message.content.as_deref(), Some("approved"));
    let metadata = result
        .final_message
        .tool_result
        .as_ref()
        .expect("hook metadata should be persisted");
    assert_eq!(metadata.details, json!({"source": "after-hook"}));
    assert!(metadata.terminate);
    assert!(!metadata.is_error);
}

#[tokio::test]
async fn after_tool_hook_errors_become_structured_tool_errors() {
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
        Message::assistant(Some("recovered".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("immediate tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("hook agent should be valid")
    .with_after_tool_hook(|_context| {
        Box::pin(async { Err::<ToolResultOverride, _>("hook failed".to_string()) })
    });

    let result = agent
        .run("call echo")
        .await
        .expect("hook errors should be returned as tool results");
    assert!(result.messages[2]
        .content
        .as_deref()
        .is_some_and(|content| content.contains("after_tool_call hook failed: hook failed")));
    assert_eq!(
        result.messages[2]
            .tool_result
            .as_ref()
            .map(|value| value.is_error),
        Some(true)
    );
    assert_eq!(result.final_message.content.as_deref(), Some("recovered"));
}

#[tokio::test]
async fn after_tool_hook_is_skipped_for_unknown_tool_outcomes() {
    let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
        None,
        Some(vec![tool_call("uncertain_tool", "call-1")]),
    )])));
    let called = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools
        .register(UncertainResultTool)
        .expect("uncertain tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("uncertain agent should be valid")
    .with_after_tool_hook({
        let called = Arc::clone(&called);
        move |_context| {
            called.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(ToolResultOverride::default()) })
        }
    });

    let error = agent
        .run("call uncertain tool")
        .await
        .expect_err("unknown tool outcomes should stop the run");
    assert!(matches!(error, AgentError::RuntimeActionUnknown { .. }));
    assert_eq!(called.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn after_tool_hook_results_are_bounded_before_persistence() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
        Message::assistant(Some("handled".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new().with_output_limit(64);
    tools
        .register(ImmediateTool)
        .expect("echo tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("hook agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone())
    .with_after_tool_hook(|_context| {
        Box::pin(async {
            Ok(ToolResultOverride {
                content: Some("你好".repeat(100)),
                is_error: Some(true),
                details: Some(json!({"payload": "x".repeat(1000)})),
                terminate: Some(false),
            })
        })
    });

    let result = agent
        .run("call echo")
        .await
        .expect("bounded hook result should complete");

    let tool = &result.messages[2];
    assert!(tool
        .content
        .as_deref()
        .is_some_and(|content| content.len() <= 64));
    assert_eq!(
        tool.tool_result.as_ref().map(|value| value.is_error),
        Some(true)
    );
    assert_eq!(
        tool.tool_result.as_ref().map(|value| &value.details),
        Some(&Value::Null)
    );

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable");
    assert_eq!(events.len(), 2);
    assert_eq!(events[1].draft.kind.as_str(), "action_failed");
}

#[tokio::test]
async fn after_tool_hook_obeys_the_run_deadline() {
    let directory = tempdir().expect("temporary directory should exist");
    let session = SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
        .await
        .expect("session should open");
    let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
        None,
        Some(vec![tool_call("echo_tool", "call-1")]),
    )])));
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("echo tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("hook agent should be valid")
    .with_session(session.clone())
    .with_after_tool_hook(|_context| {
        Box::pin(async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(ToolResultOverride::default())
        })
    });

    let error = agent
        .run_with_timeout("call echo", Duration::from_millis(10))
        .await
        .expect_err("a hanging after hook should exceed the run deadline");

    assert!(matches!(error, AgentError::RunTimedOut { .. }));
    let records = session.load().await.expect("session should load");
    assert_eq!(records.len(), 3);
    assert!(records[2]
        .message
        .content
        .as_deref()
        .is_some_and(|content| content.contains("after_tool_call hook timed out")));
}

#[tokio::test]
async fn structured_tool_result_preserves_details_and_can_terminate() {
    let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
        None,
        Some(vec![tool_call("structured_tool", "call-1")]),
    )])));
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut tools = ToolRegistry::new();
    tools
        .register(StructuredTool)
        .expect("structured tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("structured result agent should be valid")
    .with_event_sink({
        let events = events.clone();
        move |event| {
            events
                .lock()
                .expect("event mutex should not poison")
                .push(event)
        }
    });

    let result = agent
        .run("run structured tool")
        .await
        .expect("terminating tool should complete the run");

    assert_eq!(result.final_message.role, "tool");
    assert_eq!(result.final_message.content.as_deref(), Some("structured"));
    let metadata = result
        .final_message
        .tool_result
        .as_ref()
        .expect("structured metadata should be persisted");
    assert_eq!(metadata.details, json!({"source": "test"}));
    assert!(!metadata.is_error);
    assert!(metadata.terminate);
    let events = events.lock().expect("event mutex should not poison");
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolFinished {
            details,
            terminate: true,
            is_error: false,
            ..
        } if details == &json!({"source": "test"})
    )));
    assert!(!events
        .iter()
        .any(|event| matches!(event, AgentEvent::ModelStarted { step: 2, .. })));
}

#[tokio::test]
async fn forwards_tool_progress_updates_to_the_event_sink() {
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("progress_tool", "call-1")])),
        Message::assistant(Some("finished".to_string()), None),
    ])));
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut tools = ToolRegistry::new();
    tools
        .register(ProgressTool)
        .expect("progress tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("progress agent should be valid")
    .with_event_sink({
        let events = Arc::clone(&events);
        move |event| {
            events
                .lock()
                .expect("event mutex should not poison")
                .push(event)
        }
    });

    agent
        .run("run progress tool")
        .await
        .expect("progress tool should complete");

    let events = events.lock().expect("event mutex should not poison");
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolUpdate {
            call_id,
            content,
            details,
            ..
        } if call_id == "call-1"
            && content == "halfway"
            && details == &json!({"progress": 0.5})
    )));
}

#[tokio::test]
async fn terminate_only_ends_a_tool_batch_when_every_call_terminates() {
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(
            None,
            Some(vec![
                tool_call("structured_tool", "call-terminate"),
                tool_call("echo_tool", "call-continue"),
            ]),
        ),
        Message::assistant(Some("batch completed".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(StructuredTool)
        .expect("structured tool should register");
    tools
        .register(ImmediateTool)
        .expect("echo tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("batch termination agent should be valid");

    let result = agent
        .run("run a mixed tool batch")
        .await
        .expect("a non-terminating call should allow the next model turn");

    assert_eq!(
        result.final_message.content.as_deref(),
        Some("batch completed")
    );
    let tool_messages: Vec<_> = result
        .messages
        .iter()
        .filter(|message| message.role == "tool")
        .collect();
    assert_eq!(tool_messages.len(), 2);
    assert!(tool_messages
        .iter()
        .any(|message| message.tool_call_id.as_deref() == Some("call-terminate")));
    assert!(tool_messages
        .iter()
        .any(|message| message.tool_call_id.as_deref() == Some("call-continue")));
}

#[tokio::test]
async fn accepts_an_injected_chat_model_without_http() {
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("fake response".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("fake model agent should be valid");

    let result = agent
        .run("hello")
        .await
        .expect("fake model run should succeed");

    assert_eq!(result.steps, 1);
    assert_eq!(
        result.final_message.content.as_deref(),
        Some("fake response")
    );
}

#[tokio::test]
async fn after_run_hook_receives_a_successful_transcript_snapshot() {
    let observed = Arc::new(Mutex::new(None));
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("fake response".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("fake model agent should be valid")
    .with_after_run_hook({
        let observed = Arc::clone(&observed);
        move |request| {
            let observed = Arc::clone(&observed);
            Box::pin(async move {
                *observed
                    .lock()
                    .expect("observation mutex should not poison") = Some(request);
                Ok(())
            })
        }
    });

    agent
        .run("hello")
        .await
        .expect("fake model run should complete");

    let request = observed
        .lock()
        .expect("observation mutex should not poison")
        .take()
        .expect("after-run hook should be called");
    assert!(request.error.is_none());
    let result = request.result.expect("successful run should have a result");
    assert_eq!(
        result.final_message.content.as_deref(),
        Some("fake response")
    );
    assert!(result
        .messages
        .iter()
        .any(|message| message.role == "user" && message.content.as_deref() == Some("hello")));
}

#[tokio::test]
async fn after_run_hook_runs_after_the_model_deadline() {
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("done".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("fake model agent should be valid")
    .with_after_run_hook(|_request| {
        Box::pin(async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(())
        })
    });

    let result = agent
        .run_with_timeout("hello", Duration::from_millis(10))
        .await
        .expect("after-run persistence should have its own lifetime");

    assert_eq!(result.final_message.content.as_deref(), Some("done"));
}

#[tokio::test]
async fn after_run_hook_runs_after_caller_cancellation() {
    let observed = Arc::new(Mutex::new(None));
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("unused".to_owned()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("cancelled agent should be valid")
    .with_after_run_hook({
        let observed = Arc::clone(&observed);
        move |request| {
            let observed = Arc::clone(&observed);
            Box::pin(async move {
                *observed
                    .lock()
                    .expect("observation mutex should not poison") = Some(request);
                Ok(())
            })
        }
    });
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let error = agent
        .run_with_cancellation("cancel before model", cancellation)
        .await
        .expect_err("cancelled run should fail");
    assert!(matches!(error, AgentError::RunCancelled));
    let request = observed
        .lock()
        .expect("observation mutex should not poison")
        .take()
        .expect("after-run hook should observe cancellation");
    assert_eq!(request.outcome, RunOutcome::Cancelled);
    assert!(request.transcript.is_empty());
}

#[tokio::test]
async fn after_run_hook_receives_failure_and_propagates_hook_errors() {
    let observed = Arc::new(Mutex::new(None));
    let agent = AgentLoop::new(
        FailingChatModel,
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("failing model agent should be valid")
    .with_after_run_hook({
        let observed = Arc::clone(&observed);
        move |request| {
            let observed = Arc::clone(&observed);
            Box::pin(async move {
                *observed
                    .lock()
                    .expect("observation mutex should not poison") = Some(request);
                Ok(())
            })
        }
    });

    let error = agent
        .run("hello")
        .await
        .expect_err("model failure should be returned");
    assert!(matches!(
        error,
        AgentError::Model(ModelError::EmptyResponse)
    ));
    let request = observed
        .lock()
        .expect("observation mutex should not poison")
        .take()
        .expect("after-run hook should observe model failure");
    assert!(request.result.is_none());
    assert!(request
        .error
        .as_deref()
        .is_some_and(|error| error.contains("model returned no choices")));

    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("fake response".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("fake model agent should be valid")
    .with_after_run_hook(|_request| Box::pin(async { Err("persist failed".to_owned()) }));
    assert!(matches!(
        agent.run("hello").await,
        Err(AgentError::AfterRunHook(error)) if error == "persist failed"
    ));

    let agent = AgentLoop::new(
        FailingChatModel,
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("failing model agent should be valid")
    .with_after_run_hook(|_request| Box::pin(async { Err("persist failed".to_owned()) }));
    assert!(matches!(
        agent.run("hello").await,
        Err(AgentError::Model(ModelError::EmptyResponse))
    ));
}

#[tokio::test]
async fn after_run_hook_keeps_partial_transcript_after_tool_failure() {
    let observed = Arc::new(Mutex::new(None));
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("echo tool should register");
    let agent = AgentLoop::new(
        FailingAfterToolChatModel {
            calls: Arc::new(AtomicUsize::new(0)),
        },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("partial failure agent should be valid")
    .with_after_run_hook({
        let observed = Arc::clone(&observed);
        move |request| {
            let observed = Arc::clone(&observed);
            Box::pin(async move {
                *observed
                    .lock()
                    .expect("observation mutex should not poison") = Some(request);
                Ok(())
            })
        }
    });

    let error = agent
        .run("call echo then fail")
        .await
        .expect_err("second model request should fail");
    assert!(matches!(
        error,
        AgentError::Model(ModelError::EmptyResponse)
    ));
    let request = observed
        .lock()
        .expect("observation mutex should not poison")
        .take()
        .expect("after-run hook should observe the failure");
    assert!(request.result.is_none());
    assert_eq!(
        request
            .transcript
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>(),
        vec!["user", "assistant", "tool"]
    );
}

#[tokio::test]
async fn evaluates_and_injects_memory_before_the_model_request() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let emitted_events = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("answer with memory".to_owned()), None),
            requests: requests.clone(),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("memory hook agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone())
    .with_event_sink({
        let emitted_events = Arc::clone(&emitted_events);
        move |event| {
            emitted_events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event);
        }
    })
    .with_memory_hook(
        MemoryHookConfig {
            namespace: "bench/text".to_owned(),
            max_tokens: 32,
            ..MemoryHookConfig::default()
        },
        |request| {
            Box::pin(async move {
                Ok(Some(MemoryProposal {
                    proposal_id: Uuid::from_u128(7),
                    run_id: request.context.run_id,
                    namespace: request.context.namespace.clone(),
                    base_revision: request.context.current_revision,
                    memory_revision: request.context.memory_revision,
                    policy_revision: request.context.policy_revision,
                    candidates: vec![MemoryCandidate {
                        candidate_id: Uuid::from_u128(8),
                        projection: MemoryProjection {
                            projection_id: Uuid::from_u128(9),
                            source_memory_id: Uuid::from_u128(10),
                            source_version: 1,
                            content: "previous parser fix".to_owned(),
                            evidence_refs: Vec::new(),
                            provenance: "text-memory:record-1".to_owned(),
                            token_cost: 5,
                        },
                        score: 0.9,
                        confidence: 0.8,
                        expires_at_ms: None,
                    }],
                    created_at_ms: request.context.now_ms,
                    expires_at_ms: Some(request.context.now_ms + 1_000),
                    retrieval_summary: None,
                }))
            })
        },
    );

    agent
        .run("fix the parser")
        .await
        .expect("run should succeed");

    let requests = requests
        .lock()
        .expect("request mutex should not be poisoned");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].len(), 2);
    assert_eq!(requests[0][0].role, "system");
    assert!(requests[0][0]
        .content
        .as_deref()
        .is_some_and(|content| content.contains("previous parser fix")));
    assert_eq!(requests[0][1].content.as_deref(), Some("fix the parser"));

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("memory decision should be readable");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].draft.kind.as_str(), "memory_injected");

    let emitted_events = emitted_events
        .lock()
        .expect("event mutex should not be poisoned");
    assert!(emitted_events.iter().any(|event| matches!(
        event,
        AgentEvent::MemoryDecision {
            step: 1,
            proposal_id,
            event_sequence: 1,
            decision_kind,
            injection_count: 1,
            total_token_cost: 5,
            skip_reason: None,
            ..
        } if *proposal_id == Uuid::from_u128(7) && decision_kind == "injected"
    )));
}

#[tokio::test]
async fn rejects_memory_that_would_exceed_context_before_runtime_audit() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let agent = AgentLoop::new_with_policy(
        RecordingChatModel {
            response: Message::assistant(Some("unused".to_owned()), None),
            requests: Arc::new(Mutex::new(Vec::new())),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            max_context_bytes: 128,
            ..RunPolicy::default()
        },
    )
    .expect("memory budget agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone())
    .with_memory_hook(MemoryHookConfig::default(), |request| {
        Box::pin(async move {
            Ok(Some(MemoryProposal {
                proposal_id: Uuid::from_u128(17),
                run_id: request.context.run_id,
                namespace: request.context.namespace.clone(),
                base_revision: request.context.current_revision,
                memory_revision: request.context.memory_revision,
                policy_revision: request.context.policy_revision,
                candidates: vec![MemoryCandidate {
                    candidate_id: Uuid::from_u128(18),
                    projection: MemoryProjection {
                        projection_id: Uuid::from_u128(19),
                        source_memory_id: Uuid::from_u128(20),
                        source_version: 1,
                        content: "x".repeat(512),
                        evidence_refs: Vec::new(),
                        provenance: "oversized-memory".to_owned(),
                        token_cost: 128,
                    },
                    score: 1.0,
                    confidence: 1.0,
                    expires_at_ms: None,
                }],
                created_at_ms: request.context.now_ms,
                expires_at_ms: None,
                retrieval_summary: None,
            }))
        })
    });

    let error = agent
        .run("prompt")
        .await
        .expect_err("oversized memory should fail before the model request");
    assert!(
        matches!(error, AgentError::ContextTooLarge { .. }),
        "{error:?}"
    );

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable");
    assert!(events.is_empty());
}

#[tokio::test]
async fn system_prompt_is_ephemeral_and_precedes_session_messages() {
    let directory = tempdir().expect("temporary directory should exist");
    let session = SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
        .await
        .expect("session should open");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("done".to_owned()), None),
            requests: requests.clone(),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("system prompt agent should be valid")
    .with_system_prompt("You are a careful coding agent.")
    .with_session(session.clone());

    agent
        .run("inspect the project")
        .await
        .expect("run should succeed");

    {
        let requests = requests
            .lock()
            .expect("request mutex should not be poisoned");
        assert_eq!(requests[0][0].role, "system");
        assert_eq!(
            requests[0][0].content.as_deref(),
            Some("You are a careful coding agent.")
        );
        assert_eq!(
            requests[0][1].content.as_deref(),
            Some("inspect the project")
        );
    }

    let records = session.load().await.expect("session should load");
    assert!(records.iter().all(|record| {
        record.message.content.as_deref() != Some("You are a careful coding agent.")
    }));
}

#[tokio::test]
async fn rejects_a_second_run_while_the_agent_is_active() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let agent = Arc::new(
        AgentLoop::new(
            BlockingChatModel {
                started: started.clone(),
                release: release.clone(),
                response: Message::assistant(Some("done".to_string()), None),
            },
            ToolRegistry::new(),
            "blocking-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("blocking agent should be valid"),
    );
    let first_agent = Arc::clone(&agent);
    let first = tokio::spawn(async move { first_agent.run("first").await });
    started.notified().await;

    let error = agent
        .run("second")
        .await
        .expect_err("a concurrent run should be rejected");
    assert!(matches!(error, AgentError::RunBusy));

    release.notify_one();
    first
        .await
        .expect("first run task should join")
        .expect("first run should complete");
}

#[tokio::test]
async fn steering_is_injected_before_the_next_model_request() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("answer".to_string()), None),
            requests: requests.clone(),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("steering agent should be valid");
    agent
        .steer("additional constraint")
        .expect("steering should queue");

    agent.run("initial task").await.expect("run should succeed");

    let requests = requests.lock().expect("request mutex should not poison");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0][0].content.as_deref(), Some("initial task"));
    assert_eq!(
        requests[0][1].content.as_deref(),
        Some("additional constraint")
    );
}

#[tokio::test]
async fn follow_up_starts_after_a_natural_assistant_stop() {
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(Some("first answer".to_string()), None),
        Message::assistant(Some("follow-up answer".to_string()), None),
    ])));
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        3,
    )
    .expect("follow-up agent should be valid");
    agent
        .follow_up("continue with a second task")
        .expect("follow-up should queue");

    let result = agent.run("first task").await.expect("run should succeed");

    assert_eq!(result.steps, 2);
    assert_eq!(
        result.final_message.content.as_deref(),
        Some("follow-up answer")
    );
    assert_eq!(result.messages.len(), 4);
    assert_eq!(
        result.messages[2].content.as_deref(),
        Some("continue with a second task")
    );
}

#[tokio::test]
async fn rejects_idle_control_messages_after_the_first_run() {
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("done".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("agent should be valid");

    agent
        .run("first task")
        .await
        .expect("first run should succeed");

    assert!(matches!(
        agent.steer("stale steering"),
        Err(AgentError::ControlQueueRequiresActiveRun { kind: "steering" })
    ));
    assert!(matches!(
        agent.follow_up("stale follow-up"),
        Err(AgentError::ControlQueueRequiresActiveRun { kind: "follow-up" })
    ));
}

#[tokio::test]
async fn clears_unconsumed_control_messages_when_a_run_finishes() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("done".to_string()), None),
            requests: Arc::clone(&requests),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("agent should be valid");
    agent
        .follow_up("stale follow-up")
        .expect("initial follow-up should be queueable");

    agent
        .run("first task")
        .await
        .expect("first run should succeed");
    agent
        .run("second task")
        .await
        .expect("second run should succeed");

    let requests = requests.lock().expect("request mutex should not poison");
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].len(), 1);
    assert_eq!(requests[1].len(), 1);
    assert_eq!(requests[1][0].content.as_deref(), Some("second task"));
}

#[tokio::test]
async fn retries_transient_model_failures_without_duplicate_deltas() {
    let calls = Arc::new(AtomicUsize::new(0));
    let events = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new_with_policy(
        RetryChatModel {
            calls: calls.clone(),
            response: Message::assistant(Some("recovered".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            max_model_retries: 1,
            model_retry_backoff_ms: 0,
            ..RunPolicy::default()
        },
    )
    .expect("retry policy should be valid")
    .with_event_sink({
        let events = events.clone();
        move |event| {
            events
                .lock()
                .expect("event mutex should not poison")
                .push(event)
        }
    });

    let result = agent.run("hello").await.expect("retry should recover");

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(result.final_message.content.as_deref(), Some("recovered"));
    let events = events.lock().expect("event mutex should not poison");
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::AssistantDelta { .. }))
            .count(),
        1
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ModelRetry {
            attempt: 1,
            delay_ms: 0,
            ..
        }
    )));
}

#[tokio::test]
async fn resets_partial_stream_output_before_retrying() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = AgentLoop::new_with_policy(
        PartialRetryChatModel {
            calls: calls.clone(),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(true),
        },
        RunPolicy {
            max_model_retries: 1,
            model_retry_backoff_ms: 0,
            ..RunPolicy::default()
        },
    )
    .expect("partial retry agent should be valid")
    .with_event_sink({
        let events = events.clone();
        move |event| {
            events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event)
        }
    });

    let result = agent
        .run("stream a response")
        .await
        .expect("retry should recover after resetting partial output");

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(result.final_message.content.as_deref(), Some("Hello"));
    let events = events.lock().expect("event mutex should not be poisoned");
    let stream_events = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::AssistantDelta { .. }
                    | AgentEvent::ReasoningDelta { .. }
                    | AgentEvent::AssistantStreamReset { .. }
                    | AgentEvent::ModelRetry { .. }
            )
        })
        .collect::<Vec<_>>();
    assert!(matches!(
        stream_events.as_slice(),
        [
            AgentEvent::ReasoningDelta { content, .. },
            AgentEvent::AssistantDelta { content: first, .. },
            AgentEvent::AssistantStreamReset { attempt: 1, .. },
            AgentEvent::ModelRetry { attempt: 1, .. },
            AgentEvent::AssistantDelta { content: second, .. },
        ] if content == "draft" && first == "Hel" && second == "Hello"
    ));
    assert!(events
        .iter()
        .any(|event| matches!(event, AgentEvent::AssistantMessage { .. })));
}

#[tokio::test]
async fn emits_stream_delta_before_model_completion() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let agent = AgentLoop::new(
        GatedStreamingChatModel {
            started: started.clone(),
            release: release.clone(),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("streaming agent should be valid")
    .with_event_sink({
        let events = events.clone();
        move |event| {
            events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event)
        }
    });

    let task = tokio::spawn(async move { agent.run("stream now").await });
    started.notified().await;
    assert!(events.lock().expect("event mutex should not be poisoned").iter().any(
            |event| matches!(event, AgentEvent::AssistantDelta { content, .. } if content == "partial")
        ));

    release.notify_one();
    task.await
        .expect("streaming task should join")
        .expect("streaming run should succeed");
}

#[tokio::test]
async fn tool_execution_passes_through_runtime_admission() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
        Message::assistant(Some("tool completed".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("immediate tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone());

    let result = agent
        .run("use the echo tool")
        .await
        .expect("runtime-admitted tool run should succeed");

    assert_eq!(result.messages[2].content.as_deref(), Some("runtime-ok"));
    let metadata = result.messages[2]
        .tool_result
        .as_ref()
        .expect("runtime metadata should be persisted");
    assert!(metadata.action_id.is_some());
    assert!(metadata.grant_id.is_some());
    assert_eq!(
        result.final_message.content.as_deref(),
        Some("tool completed")
    );

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].draft.kind.as_str(), "action_admitted");
    assert_eq!(events[1].draft.kind.as_str(), "action_succeeded");
}

#[tokio::test]
async fn sequential_tool_batch_preserves_event_ledger_and_session_order() {
    let directory = tempdir().expect("temporary directory should exist");
    let session_path = directory.path().join("conversation.jsonl");
    let ledger_path = directory.path().join("runtime.jsonl");
    let session_id = Uuid::new_v4();
    let session = SessionStore::open(&session_path, session_id)
        .await
        .expect("session should open");
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
        Message::assistant(Some("done".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("immediate tool should register");
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured_events = Arc::clone(&events);
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("agent should be valid")
    .with_session(session)
    .with_runtime_ledger_file(ledger_path.clone())
    .with_event_sink(move |event| {
        captured_events
            .lock()
            .expect("event mutex should not be poisoned")
            .push(event);
    });

    let result = agent
        .run("use the echo tool")
        .await
        .expect("sequential tool run should succeed");
    assert_eq!(result.final_message.content.as_deref(), Some("done"));
    drop(agent);

    {
        let events = events.lock().expect("event mutex should not be poisoned");
        let tool_started = events
            .iter()
            .position(|event| {
                matches!(event, AgentEvent::ToolStarted { call, .. } if call.id == "call-1")
            })
            .expect("tool start should be emitted");
        let tool_finished = events
            .iter()
            .position(|event| {
                matches!(event, AgentEvent::ToolFinished { call_id, .. } if call_id == "call-1")
            })
            .expect("tool finish should be emitted");
        let turn_finished = events
            .iter()
            .position(|event| matches!(event, AgentEvent::TurnFinished { step: 1, .. }))
            .expect("turn finish should be emitted");
        assert!(tool_started < tool_finished);
        assert!(tool_finished < turn_finished);
    }

    let reopened = SessionStore::open(&session_path, session_id)
        .await
        .expect("session should reopen");
    let records = reopened.load_repaired().await.expect("session should load");
    let roles = records
        .iter()
        .map(|record| record.message.role.as_str())
        .collect::<Vec<_>>();
    assert_eq!(roles, vec!["user", "assistant", "tool", "assistant"]);
    assert_eq!(records[2].message.content.as_deref(), Some("runtime-ok"));

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let ledger_events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable");
    assert_eq!(ledger_events.len(), 2);
    assert_eq!(ledger_events[0].draft.kind.as_str(), "action_admitted");
    assert_eq!(ledger_events[1].draft.kind.as_str(), "action_succeeded");
}

#[tokio::test]
async fn structured_tool_errors_are_recorded_as_failed_runtime_actions() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("error_tool", "call-1")])),
        Message::assistant(Some("recovered".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(ErrorResultTool)
        .expect("error tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone());

    let result = agent
        .run("use the error tool")
        .await
        .expect("structured tool errors should return to the model");
    assert_eq!(result.final_message.content.as_deref(), Some("recovered"));
    assert!(result.messages[2]
        .tool_result
        .as_ref()
        .is_some_and(|metadata| metadata.is_error));

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].draft.kind.as_str(), "action_admitted");
    assert_eq!(events[1].draft.kind.as_str(), "action_failed");
    assert_eq!(events[1].draft.payload["error"], "known failure");
}

#[tokio::test]
async fn uncertain_structured_tool_results_remain_pending_for_reconciliation() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("uncertain_tool", "call-1")])),
        Message::assistant(Some("must not run".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(UncertainResultTool)
        .expect("uncertain tool should register");
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone());

    let error = agent
        .run("use the uncertain tool")
        .await
        .expect_err("uncertain outcome must stop the run");
    assert!(matches!(error, AgentError::RuntimeActionUnknown { .. }));

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable");
    assert_eq!(events.len(), 2);
    assert_eq!(events[1].draft.kind.as_str(), "action_unknown");
}

#[tokio::test]
async fn reconciled_runtime_results_are_projected_into_the_next_model_context() {
    let directory = tempdir().expect("temporary directory should exist");
    let session_path = directory.path().join("conversation.jsonl");
    let ledger_path = directory.path().join("runtime.jsonl");
    let session_id = Uuid::new_v4();
    let session = SessionStore::open(&session_path, session_id)
        .await
        .expect("session should open");
    let mut tools = ToolRegistry::new();
    tools
        .register(UncertainResultTool)
        .expect("uncertain tool should register");
    let first = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(None, Some(vec![tool_call("uncertain_tool", "call-1")])),
        },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("initial agent should be valid")
    .with_session(session)
    .with_runtime_ledger_file(ledger_path.clone());

    first
        .run("run an uncertain tool")
        .await
        .expect_err("uncertain execution should remain pending");
    let action_id = first
        .pending_runtime_actions()
        .expect("pending action should be visible")
        .first()
        .expect("one action should be pending")
        .action_id;
    first
        .reconcile_runtime_action(
            action_id,
            RuntimeReconcileOutcome::Succeeded {
                output: json!("confirmed output"),
                evidence: "provider-check-2".to_string(),
            },
        )
        .expect("reconciliation should be persisted");

    let requests = Arc::new(Mutex::new(Vec::new()));
    let reopened = SessionStore::open(&session_path, session_id)
        .await
        .expect("session should reopen");
    let second = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("continued".to_string()), None),
            requests: requests.clone(),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("recovery agent should be valid")
    .with_session(reopened)
    .with_runtime_ledger_file(ledger_path)
    .with_runtime_recovery();
    second
        .run("continue after reconciliation")
        .await
        .expect("reconciled context should resume");

    let requests = requests
        .lock()
        .expect("request mutex should not be poisoned");
    let tool_message = requests[0]
        .iter()
        .find(|message| message.role == "tool")
        .expect("the model request should include the reconciled tool result");
    assert_eq!(tool_message.content.as_deref(), Some("confirmed output"));
    assert_eq!(
        tool_message
            .tool_result
            .as_ref()
            .map(|metadata| metadata.details["status"].as_str()),
        Some(Some("reconciled"))
    );
}

#[tokio::test]
async fn fresh_runtime_run_accepts_a_completed_ledger_without_recovery() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
        Message::assistant(Some("first run".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(ImmediateTool)
        .expect("immediate tool should register");
    AgentLoop::new(
        SequenceChatModel { responses },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("first agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone())
    .run("first turn")
    .await
    .expect("first run should succeed");

    let second = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("second run".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("second agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone());

    let result = second
        .run("second turn")
        .await
        .expect("completed old ledger must not be treated as a fresh-run mismatch");

    assert_eq!(result.final_message.content.as_deref(), Some("second run"));
}

#[tokio::test]
async fn runtime_recovery_resumes_the_latest_run_revision() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");

    let first_responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("echo_tool", "call-1")])),
        Message::assistant(Some("first run".to_string()), None),
    ])));
    let mut first_tools = ToolRegistry::new();
    first_tools
        .register(ImmediateTool)
        .expect("immediate tool should register");
    let first = AgentLoop::new(
        SequenceChatModel {
            responses: first_responses,
        },
        first_tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("first agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone());
    first
        .run("first turn")
        .await
        .expect("first run should succeed");

    let second_responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(None, Some(vec![tool_call("echo_tool", "call-2")])),
        Message::assistant(Some("resumed run".to_string()), None),
    ])));
    let mut second_tools = ToolRegistry::new();
    second_tools
        .register(ImmediateTool)
        .expect("immediate tool should register");
    let started_run_id = Arc::new(Mutex::new(None));
    let started_run_id_sink = Arc::clone(&started_run_id);
    let after_run_id = Arc::new(Mutex::new(None));
    let after_run_id_hook = Arc::clone(&after_run_id);
    let second = AgentLoop::new(
        SequenceChatModel {
            responses: second_responses,
        },
        second_tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("second agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone())
    .with_runtime_recovery()
    .with_event_sink(move |event| {
        if let AgentEvent::RunStarted { context } = event {
            *started_run_id_sink
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(context.run_id);
        }
    })
    .with_memory_hook(
        MemoryHookConfig {
            namespace: "recovery-test".to_owned(),
            max_tokens: 32,
            ..MemoryHookConfig::default()
        },
        recovery_memory_hook,
    )
    .with_after_run_hook(move |request| {
        *after_run_id_hook
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(request.context.run_id);
        Box::pin(async { Ok(()) })
    });
    second
        .run("resumed turn")
        .await
        .expect("resumed run should succeed");

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable");
    assert_eq!(events.len(), 6);
    assert_eq!(events[0].draft.kind.as_str(), "action_admitted");
    assert_eq!(events[1].draft.kind.as_str(), "action_succeeded");
    assert_eq!(events[2].draft.kind.as_str(), "memory_injected");
    assert_eq!(events[3].draft.kind.as_str(), "action_admitted");
    assert_eq!(events[4].draft.kind.as_str(), "action_succeeded");
    assert_eq!(events[5].draft.kind.as_str(), "memory_injected");
    assert_eq!(events[0].draft.run_id, events[3].draft.run_id);
    assert_eq!(events[3].draft.revision, pi_agent_runtime::Revision::new(1));
    assert_eq!(events[4].draft.revision, pi_agent_runtime::Revision::new(2));
    let recovered_run_id = events[3].draft.run_id.as_uuid();
    assert_eq!(
        *started_run_id
            .lock()
            .unwrap_or_else(|error| error.into_inner()),
        Some(recovered_run_id)
    );
    assert_eq!(
        *after_run_id
            .lock()
            .unwrap_or_else(|error| error.into_inner()),
        Some(recovered_run_id)
    );
}

#[tokio::test]
async fn injects_skill_index_into_model_context() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut catalog = SkillCatalog::new();
    catalog
        .insert(
            Skill::new(
                "research",
                "Research tasks",
                "Use primary sources.",
                "/skills/research/SKILL.md",
            )
            .expect("test skill should be valid"),
        )
        .expect("test catalog should accept the skill");

    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("done".to_string()), None),
            requests: Arc::clone(&requests),
        },
        ToolRegistry::new(),
        "recording-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("recording model agent should be valid")
    .with_skills(catalog);

    assert!(agent
        .tools
        .model_definitions()
        .iter()
        .any(|definition| definition.function.name == "read_skill"));

    agent
        .run("find evidence")
        .await
        .expect("run should succeed");

    let requests = requests
        .lock()
        .expect("request mutex should not be poisoned");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0][0].role, "system");
    assert!(requests[0][0]
        .content
        .as_deref()
        .is_some_and(|content| content.contains("<name>research</name>")));
    assert_eq!(
        requests[0].last().map(|message| message.role.as_str()),
        Some("user")
    );
}

#[tokio::test]
async fn steering_prompt_is_injected_before_the_next_model_request() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("done".to_owned()), None),
            requests: Arc::clone(&requests),
        },
        ToolRegistry::new(),
        "recording-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("recording model agent should be valid");
    agent
        .steer("stop and inspect the result")
        .expect("steering prompt should be queued");

    agent
        .run("original request")
        .await
        .expect("run should succeed");

    let requests = requests
        .lock()
        .expect("request mutex should not be poisoned");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0][0].content.as_deref(), Some("original request"));
    assert_eq!(
        requests[0][1].content.as_deref(),
        Some("stop and inspect the result")
    );
}

#[tokio::test]
async fn follow_up_prompt_runs_after_the_current_assistant_turn() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("done".to_owned()), None),
            requests: Arc::clone(&requests),
        },
        ToolRegistry::new(),
        "recording-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("recording model agent should be valid");
    agent
        .follow_up("now summarize it")
        .expect("follow-up prompt should be queued");

    let result = agent
        .run("original request")
        .await
        .expect("run should succeed");

    let requests = requests
        .lock()
        .expect("request mutex should not be poisoned");
    assert_eq!(result.steps, 2);
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].len(), 1);
    assert_eq!(requests[0][0].content.as_deref(), Some("original request"));
    assert_eq!(requests[1].len(), 3);
    assert_eq!(requests[1][2].content.as_deref(), Some("now summarize it"));
}

#[tokio::test]
async fn timeout_cancels_model_and_emits_one_cancelled_terminal_event() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured_events = Arc::clone(&events);
    let agent = AgentLoop::new(
        SlowChatModel {
            delay: Duration::from_millis(100),
            response: Message::assistant(Some("too late".to_string()), None),
        },
        ToolRegistry::new(),
        "slow-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("slow model agent should be valid")
    .with_event_sink(move |event| {
        captured_events
            .lock()
            .expect("event mutex should not be poisoned")
            .push(event);
    });

    let error = agent
        .run_with_timeout("hello", Duration::from_millis(10))
        .await
        .expect_err("slow model should exceed the run deadline");

    assert!(matches!(error, AgentError::RunTimedOut { timeout_ms: 10 }));
    let events = events.lock().expect("event mutex should not be poisoned");
    assert_eq!(events.len(), 4);
    assert!(matches!(events[0], AgentEvent::RunStarted { .. }));
    assert!(matches!(events[1], AgentEvent::TurnStarted { step: 1, .. }));
    assert!(matches!(
        events[2],
        AgentEvent::ModelStarted { step: 1, .. }
    ));
    assert!(matches!(
        events[3],
        AgentEvent::RunFinished {
            outcome: RunOutcome::Cancelled,
            steps: 1,
            ..
        }
    ));
}

#[tokio::test]
async fn caller_cancellation_stops_model_and_emits_cancelled_terminal_event() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured_events = Arc::clone(&events);
    let agent = AgentLoop::new(
        SlowChatModel {
            delay: Duration::from_millis(100),
            response: Message::assistant(Some("too late".to_string()), None),
        },
        ToolRegistry::new(),
        "slow-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("slow model agent should be valid")
    .with_event_sink(move |event| {
        captured_events
            .lock()
            .expect("event mutex should not be poisoned")
            .push(event);
    });
    let cancellation = CancellationToken::new();
    let run_cancellation = cancellation.clone();
    let task =
        tokio::spawn(async move { agent.run_with_cancellation("hello", run_cancellation).await });

    tokio::time::sleep(Duration::from_millis(10)).await;
    cancellation.cancel();
    let error = task
        .await
        .expect("agent task should join")
        .expect_err("caller cancellation should stop the run");

    assert!(matches!(error, AgentError::RunCancelled));
    let events = events.lock().expect("event mutex should not be poisoned");
    assert!(matches!(
        events.last(),
        Some(AgentEvent::RunFinished {
            outcome: RunOutcome::Cancelled,
            steps: 1,
            error: Some(error),
            ..
        }) if error.contains("cancelled by the caller")
    ));
}

#[tokio::test]
async fn empty_prompt_is_rejected_before_an_immediate_deadline() {
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("unused".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("fake model agent should be valid");

    let error = agent
        .run_with_timeout("  ", Duration::ZERO)
        .await
        .expect_err("empty prompt should be rejected");

    assert!(matches!(error, AgentError::EmptyPrompt));
}

#[tokio::test]
async fn timeout_persists_cancelled_results_for_pending_tools() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let session = SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
        .await
        .expect("session should open");
    let events = Arc::new(Mutex::new(Vec::new()));
    let after_run_outcome = Arc::new(Mutex::new(None));
    let mut tools = ToolRegistry::new();
    tools.register(SlowTool).expect("slow tool should register");
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(None, Some(vec![tool_call("slow_tool", "call-1")])),
        },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("tool timeout agent should be valid")
    .with_session(session.clone())
    .with_runtime_ledger_file(ledger_path.clone())
    .with_event_sink({
        let events = Arc::clone(&events);
        move |event| {
            events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event);
        }
    })
    .with_after_run_hook({
        let after_run_outcome = Arc::clone(&after_run_outcome);
        move |request| {
            let after_run_outcome = Arc::clone(&after_run_outcome);
            Box::pin(async move {
                *after_run_outcome
                    .lock()
                    .expect("outcome mutex should not be poisoned") = Some(request.outcome);
                Ok(())
            })
        }
    });

    let error = agent
        .run_with_timeout("use the slow tool", Duration::from_millis(10))
        .await
        .expect_err("slow tool should exceed the run deadline");

    assert!(matches!(error, AgentError::RunTimedOut { .. }));
    assert_eq!(
        *after_run_outcome
            .lock()
            .expect("outcome mutex should not be poisoned"),
        Some(RunOutcome::Unknown)
    );
    assert!(matches!(
        events
            .lock()
            .expect("event mutex should not be poisoned")
            .last(),
        Some(AgentEvent::RunFinished {
            outcome: RunOutcome::Unknown,
            ..
        })
    ));
    let records = session.load().await.expect("session should load");
    assert_eq!(records.len(), 3);
    assert_eq!(records[1].message.role, "assistant");
    assert_eq!(records[2].message.role, "tool");
    assert!(records[2]
        .message
        .content
        .as_deref()
        .is_some_and(|content| content.contains("cancelled")));
    assert!(records[2]
        .message
        .tool_result
        .as_ref()
        .is_some_and(|metadata| metadata.is_error && !metadata.terminate));
    assert!(records[2]
        .message
        .tool_result
        .as_ref()
        .is_some_and(|metadata| metadata.action_id.is_some() && metadata.grant_id.is_some()));
    assert_eq!(
        records[2]
            .message
            .tool_result
            .as_ref()
            .and_then(|metadata| metadata.details.get("status"))
            .and_then(Value::as_str),
        Some("unknown")
    );
    assert_eq!(
        records[2]
            .message
            .tool_result
            .as_ref()
            .and_then(|metadata| metadata.details.get("executed"))
            .and_then(Value::as_bool),
        Some(true)
    );

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].draft.kind.as_str(), "action_admitted");
    assert_eq!(events[1].draft.kind.as_str(), "action_unknown");
}

#[tokio::test]
async fn admitted_caller_cancellation_reports_unknown_run_outcome() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let session = SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
        .await
        .expect("session should open");
    let events = Arc::new(Mutex::new(Vec::new()));
    let after_run_outcome = Arc::new(Mutex::new(None));
    let mut tools = ToolRegistry::new();
    tools
        .register(CancelInsideTool)
        .expect("cancellation tool should register");
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(
                None,
                Some(vec![tool_call("cancel_inside_tool", "call-1")]),
            ),
        },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("cancellation agent should be valid")
    .with_session(session.clone())
    .with_runtime_ledger_file(ledger_path.clone())
    .with_event_sink({
        let events = Arc::clone(&events);
        move |event| {
            events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event);
        }
    })
    .with_after_run_hook({
        let after_run_outcome = Arc::clone(&after_run_outcome);
        move |request| {
            let after_run_outcome = Arc::clone(&after_run_outcome);
            Box::pin(async move {
                *after_run_outcome
                    .lock()
                    .expect("outcome mutex should not be poisoned") = Some(request.outcome);
                Ok(())
            })
        }
    });

    let cancellation = CancellationToken::new();
    let error = agent
        .run_with_cancellation("run the cancellation tool", cancellation)
        .await
        .expect_err("admitted cancellation should stop the run");

    assert!(matches!(error, AgentError::RunCancelled));
    assert_eq!(
        *after_run_outcome
            .lock()
            .expect("outcome mutex should not be poisoned"),
        Some(RunOutcome::Unknown)
    );
    assert!(matches!(
        events
            .lock()
            .expect("event mutex should not be poisoned")
            .last(),
        Some(AgentEvent::RunFinished {
            outcome: RunOutcome::Unknown,
            ..
        })
    ));

    let records = session.load().await.expect("session should load");
    let metadata = records[2]
        .message
        .tool_result
        .as_ref()
        .expect("cancelled tool should retain metadata");
    assert_eq!(metadata.details["status"], "unknown");
    assert_eq!(metadata.details["executed"], true);
    assert!(metadata.action_id.is_some() && metadata.grant_id.is_some());

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable");
    assert_eq!(events.len(), 2);
    assert_eq!(events[1].draft.kind.as_str(), "action_unknown");
}

#[tokio::test]
async fn tool_timeout_stops_before_requesting_another_model_turn() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let session = SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
        .await
        .expect("session should open");
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(
            None,
            Some(vec![tool_call("tool_timeout_tool", "call-timeout")]),
        ),
        Message::assistant(Some("this turn must not run".to_string()), None),
    ])));
    let mut tools = ToolRegistry::new();
    tools
        .register(ToolTimeoutTool)
        .expect("timeout tool should register");
    let agent = AgentLoop::new_with_policy(
        SequenceChatModel {
            responses: responses.clone(),
        },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            tool_timeout_secs: 1,
            ..RunPolicy::default()
        },
    )
    .expect("timeout agent should be valid")
    .with_session(session.clone())
    .with_runtime_ledger_file(ledger_path.clone());

    let error = agent
        .run("use the timeout tool")
        .await
        .expect_err("unknown tool outcome must stop the run");

    assert!(matches!(
        error,
        AgentError::RuntimeActionUnknown {
            tool_name,
            reason,
            ..
        } if tool_name == "tool_timeout_tool" && reason.contains("outcome is unknown")
    ));
    assert_eq!(
        responses
            .lock()
            .expect("response mutex should not poison")
            .len(),
        1,
        "the next model response must remain unused"
    );

    let records = session.load().await.expect("session should load");
    assert_eq!(records.len(), 3);
    assert_eq!(records[2].message.role, "tool");
    assert_eq!(
        records[2]
            .message
            .tool_result
            .as_ref()
            .map(|result| &result.details),
        Some(&serde_json::json!({
            "status": "unknown",
            "executed": true,
        }))
    );

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].draft.kind.as_str(), "action_admitted");
    assert_eq!(events[1].draft.kind.as_str(), "action_unknown");
    drop(ledger);
    assert_eq!(
        agent
            .pending_runtime_actions()
            .expect("pending action should be inspectable")
            .len(),
        1
    );
}

#[tokio::test]
async fn exposes_and_reconciles_an_unknown_runtime_action() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let mut tools = ToolRegistry::new();
    tools.register(SlowTool).expect("slow tool should register");
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(None, Some(vec![tool_call("slow_tool", "call-1")])),
        },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("agent should be valid")
    .with_runtime_ledger_file(ledger_path);

    let error = agent
        .run_with_timeout("use the slow tool", Duration::from_millis(10))
        .await
        .expect_err("tool should exceed the run deadline");
    assert!(matches!(error, AgentError::RunTimedOut { .. }));

    let pending = agent
        .pending_runtime_actions()
        .expect("pending action should be inspectable");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].tool_name, "slow_tool");
    assert_eq!(pending[0].replay_policy, ToolReplayPolicy::Never);
    let action_id = pending[0].action_id;

    agent
        .reconcile_runtime_action(
            action_id,
            RuntimeReconcileOutcome::Failed {
                error: "provider confirmed no execution".to_string(),
                evidence: "provider-check-1".to_string(),
            },
        )
        .expect("external evidence should reconcile the action");
    assert!(agent
        .pending_runtime_actions()
        .expect("pending actions should be inspectable after reconciliation")
        .is_empty());
}

#[tokio::test]
async fn runtime_recovery_refuses_unresolved_actions() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let mut tools = ToolRegistry::new();
    tools.register(SlowTool).expect("slow tool should register");
    AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(None, Some(vec![tool_call("slow_tool", "call-1")])),
        },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("initial agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone())
    .run_with_timeout("create an unresolved action", Duration::from_millis(10))
    .await
    .expect_err("the initial action should become unresolved");

    let mut recovery_tools = ToolRegistry::new();
    recovery_tools
        .register(SlowTool)
        .expect("recovery tool should register");
    let recovery = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("must not run".to_string()), None),
        },
        recovery_tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("recovery agent should be valid")
    .with_runtime_ledger_file(ledger_path)
    .with_runtime_recovery();

    let error = recovery
        .run("resume after the interrupted action")
        .await
        .expect_err("recovery must require reconciliation first");
    assert!(matches!(
        error,
        AgentError::Runtime(message) if message.contains("unresolved actions")
    ));
}

#[tokio::test]
async fn rejects_session_runtime_identity_without_ledger_evidence() {
    let directory = tempdir().expect("temporary directory should exist");
    let session_id = Uuid::new_v4();
    let session = SessionStore::open(directory.path().join("session.jsonl"), session_id)
        .await
        .expect("session should open");
    session
        .append(&Message::assistant(
            None,
            Some(vec![tool_call("echo_tool", "call-1")]),
        ))
        .await
        .expect("assistant tool call should append");
    session
        .append(&Message::tool_with_result(
            "call-1",
            "stale result",
            ToolResultMetadata::new(Value::Null, false, false)
                .with_runtime_identity(Uuid::new_v4(), Uuid::new_v4()),
        ))
        .await
        .expect("stale tool result should append");

    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("must not run".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("agent should be valid")
    .with_session(session)
    .with_runtime_ledger_file(directory.path().join("runtime.jsonl"));

    let error = agent
        .run("resume stale transcript")
        .await
        .expect_err("stale runtime identity must fail closed");
    assert!(matches!(
        error,
        AgentError::RuntimeSessionMismatch { reason, .. }
            if reason == "durable admission event is missing"
    ));
}

#[tokio::test]
async fn rejects_anonymous_uncertain_tool_results_with_runtime_ledger() {
    let directory = tempdir().expect("temporary directory should exist");
    let session_id = Uuid::new_v4();
    let session = SessionStore::open(directory.path().join("session.jsonl"), session_id)
        .await
        .expect("session should open");
    session
        .append(&Message::assistant(
            None,
            Some(vec![tool_call("echo_tool", "call-1")]),
        ))
        .await
        .expect("assistant tool call should append");
    session
        .append(&Message::tool_with_result(
            "call-1",
            "recovered unknown result",
            ToolResultMetadata::new(json!({"status": "unknown"}), true, false),
        ))
        .await
        .expect("anonymous uncertain result should append");

    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("must not run".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("agent should be valid")
    .with_session(session)
    .with_runtime_ledger_file(directory.path().join("runtime.jsonl"));

    let error = agent
        .run("resume anonymous unknown result")
        .await
        .expect_err("anonymous uncertain result must fail closed");
    assert!(matches!(
        error,
        AgentError::UnidentifiedToolOutcome { call_id, status }
            if call_id == "call-1" && status == "unknown"
    ));
}

#[tokio::test]
async fn caller_cancellation_persists_cancelled_results_for_pending_tools() {
    let directory = tempdir().expect("temporary directory should exist");
    let session_path = directory.path().join("conversation.jsonl");
    let session_id = Uuid::new_v4();
    let session = SessionStore::open(&session_path, session_id)
        .await
        .expect("session should open");
    let ledger_path = directory.path().join("runtime.jsonl");
    let mut tools = ToolRegistry::new();
    tools.register(SlowTool).expect("slow tool should register");
    let cancellation = CancellationToken::new();
    let cancel_on_tool_start = cancellation.clone();
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(None, Some(vec![tool_call("slow_tool", "call-1")])),
        },
        tools,
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("tool cancellation agent should be valid")
    .with_session(session.clone())
    .with_runtime_ledger_file(ledger_path.clone())
    .with_event_sink(move |event| {
        if matches!(event, AgentEvent::ToolStarted { .. }) {
            cancel_on_tool_start.cancel();
        }
    });
    let run_cancellation = cancellation.clone();
    let task = tokio::spawn(async move {
        agent
            .run_with_cancellation("use the slow tool", run_cancellation)
            .await
    });

    let error = task
        .await
        .expect("agent task should join")
        .expect_err("caller cancellation should stop the tool");

    assert!(matches!(error, AgentError::RunCancelled));
    let records = session.load().await.expect("session should load");
    assert_eq!(records.len(), 3);
    assert_eq!(records[1].message.role, "assistant");
    assert_eq!(records[2].message.role, "tool");
    assert_eq!(
        records[2].message.content.as_deref(),
        Some("Tool execution cancelled by the caller")
    );

    let reopened_session = SessionStore::open(&session_path, session_id)
        .await
        .expect("the cancelled session should reopen");
    let resumed = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("resumed".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("resume agent should be valid")
    .with_session(reopened_session)
    .with_runtime_ledger_file(ledger_path);

    let result = resumed
        .run("resume after cancellation")
        .await
        .expect("an unexecuted cancellation should not block recovery");
    assert_eq!(result.final_message.content.as_deref(), Some("resumed"));
}

#[tokio::test]
async fn agent_loads_a_repaired_tail_before_requesting_the_next_turn() {
    let directory = tempdir().expect("temporary directory should exist");
    let session = SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
        .await
        .expect("session should open");
    session
        .append(&Message::user("previous question"))
        .await
        .expect("previous user message should append");
    session
        .append(&Message::assistant(
            None,
            Some(vec![tool_call("read_file", "orphan-call")]),
        ))
        .await
        .expect("orphan assistant call should append");

    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("recovered context".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("agent should be valid")
    .with_session(session.clone());

    let result = agent
        .run("new question")
        .await
        .expect("repaired transcript should be accepted");

    assert_eq!(result.messages[2].role, "tool");
    assert_eq!(
        result.messages[2].tool_call_id.as_deref(),
        Some("orphan-call")
    );
    assert_eq!(
        result.final_message.content.as_deref(),
        Some("recovered context")
    );
    assert_eq!(
        session.load().await.expect("session should reload").len(),
        5
    );
}

#[tokio::test]
async fn emits_reasoning_delta_events_from_thinking_models() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        ReasoningChatModel,
        ToolRegistry::new(),
        "thinking-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(true),
        },
        1,
    )
    .expect("reasoning agent should be valid")
    .with_event_sink({
        let events = Arc::clone(&events);
        move |event| {
            events
                .lock()
                .expect("event mutex should not poison")
                .push(event)
        }
    });

    let result = agent.run("think").await.expect("run should succeed");

    assert_eq!(
        result.final_message.reasoning_content.as_deref(),
        Some("thinking")
    );
    let events = events.lock().expect("event mutex should not poison");
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ReasoningDelta { content, step: 1, .. } if content == "thinking"
    )));
}

#[tokio::test]
async fn emits_lifecycle_events_for_a_simple_run() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured_events = Arc::clone(&events);
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("event response".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("fake model agent should be valid")
    .with_event_sink(move |event| {
        captured_events
            .lock()
            .expect("event mutex should not be poisoned")
            .push(event);
    });

    agent.run("hello").await.expect("run should succeed");

    let events = events.lock().expect("event mutex should not be poisoned");
    assert_eq!(events.len(), 7);
    assert!(matches!(events[0], AgentEvent::RunStarted { .. }));
    assert!(matches!(events[1], AgentEvent::TurnStarted { step: 1, .. }));
    assert!(matches!(
        events[2],
        AgentEvent::ModelStarted { step: 1, .. }
    ));
    assert!(matches!(
        events[3],
        AgentEvent::AssistantDelta {
            step: 1,
            ref content,
            ..
        } if content == "event response"
    ));
    assert!(matches!(
        events[4],
        AgentEvent::AssistantMessage { step: 1, .. }
    ));
    assert!(matches!(
        events[5],
        AgentEvent::TurnFinished { step: 1, .. }
    ));
    assert!(matches!(
        events[6],
        AgentEvent::RunFinished {
            outcome: RunOutcome::Completed,
            ..
        }
    ));
}

#[tokio::test]
async fn run_with_context_preserves_external_identity_in_lifecycle_events() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured_events = Arc::clone(&events);
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("linked response".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("fake model agent should be valid")
    .with_event_sink(move |event| {
        captured_events
            .lock()
            .expect("event mutex should not be poisoned")
            .push(event);
    });
    let context = RunContext {
        session_id: Some(Uuid::from_u128(1)),
        run_id: Uuid::from_u128(2),
        turn_id: Uuid::from_u128(3),
    };

    agent
        .run_with_context("linked prompt", context)
        .await
        .expect("linked run should succeed");

    let events = events.lock().expect("event mutex should not be poisoned");
    assert!(matches!(
        events.first(),
        Some(AgentEvent::RunStarted { context: actual }) if *actual == context
    ));
    assert!(matches!(
        events.last(),
        Some(AgentEvent::RunFinished { context: actual, .. }) if *actual == context
    ));
}

#[tokio::test]
async fn run_with_context_rejects_a_mismatched_attached_session() {
    let directory = tempdir().expect("temporary directory should exist");
    let expected = Uuid::new_v4();
    let session = SessionStore::open(directory.path().join("session.jsonl"), expected)
        .await
        .expect("session should open");
    let agent = AgentLoop::new(
        FakeChatModel {
            response: Message::assistant(Some("must not run".to_string()), None),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("fake model agent should be valid")
    .with_session(session);
    let context = RunContext {
        session_id: Some(Uuid::new_v4()),
        run_id: Uuid::new_v4(),
        turn_id: Uuid::new_v4(),
    };

    let error = agent
        .run_with_context("mismatched session", context)
        .await
        .expect_err("a mismatched session identity must be rejected");

    assert!(matches!(
        error,
        AgentError::SessionContextMismatch {
            expected: actual,
            provided: Some(_),
        } if actual == expected
    ));
}

#[tokio::test]
async fn returns_final_assistant_message_without_tools() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(streaming_response_with_usage(
            Message::assistant(
                Some("KV cache stores attention keys and values.".to_string()),
                None,
            ),
            json!({
                "prompt_tokens": 17,
                "completion_tokens": 9,
                "total_tokens": 26
            }),
        ))
        .expect(1)
        .mount(&server)
        .await;

    let agent = AgentLoop::from_config(&test_config(server.uri()), ToolRegistry::new())
        .expect("test agent should be valid");
    let result = agent
        .run("What is KV cache?")
        .await
        .expect("run should succeed");

    assert_eq!(result.steps, 1);
    assert_eq!(result.messages.len(), 2);
    assert_eq!(
        result.final_message.content.as_deref(),
        Some("KV cache stores attention keys and values.")
    );
    assert_eq!(result.input_tokens, Some(17));
    assert_eq!(result.output_tokens, Some(9));
    assert_eq!(result.total_tokens, Some(26));
}

#[tokio::test]
async fn forwards_tool_choice_to_model_client() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_string_contains("\"tool_choice\":\"none\""))
        .respond_with(streaming_response(Message::assistant(
            Some("tools disabled".to_string()),
            None,
        )))
        .expect(1)
        .mount(&server)
        .await;

    let directory = tempdir().expect("temporary directory should exist");
    let mut config = test_config(server.uri());
    config.workspace = directory.path().to_path_buf();
    let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
        .expect("read_file should register");
    let agent = AgentLoop::from_config(&config, tools)
        .expect("test agent should be valid")
        .with_tool_choice(ToolChoice::None);

    let result = agent
        .run("Answer without calling tools")
        .await
        .expect("run should succeed");

    assert_eq!(
        result.final_message.content.as_deref(),
        Some("tools disabled")
    );
}

#[tokio::test]
async fn resumes_previous_session_messages_before_new_prompt() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_string_contains("previous answer"))
        .respond_with(streaming_response(Message::assistant(
            Some("new answer".to_string()),
            None,
        )))
        .expect(1)
        .mount(&server)
        .await;

    let directory = tempdir().expect("temporary directory should exist");
    let session = SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
        .await
        .expect("session should open");
    session
        .append(&Message::user("previous question"))
        .await
        .expect("previous user message should append");
    session
        .append(&Message::assistant(
            Some("previous answer".to_string()),
            None,
        ))
        .await
        .expect("previous assistant message should append");

    let agent = AgentLoop::from_config(&test_config(server.uri()), ToolRegistry::new())
        .expect("test agent should be valid")
        .with_session(session.clone());
    let result = agent
        .run("current question")
        .await
        .expect("resumed run should succeed");

    assert_eq!(result.messages.len(), 4);
    assert_eq!(
        result.messages[0].content.as_deref(),
        Some("previous question")
    );
    assert_eq!(
        result.messages[1].content.as_deref(),
        Some("previous answer")
    );
    assert_eq!(
        result.messages[2].content.as_deref(),
        Some("current question")
    );
    assert_eq!(result.messages[3].content.as_deref(), Some("new answer"));
    assert_eq!(
        session.load().await.expect("session should reload").len(),
        4
    );
}

#[tokio::test]
async fn forwards_thinking_setting_from_config() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_string_contains("\"enable_thinking\":false"))
        .respond_with(streaming_response(Message::assistant(
            Some("thinking disabled".to_string()),
            None,
        )))
        .expect(1)
        .mount(&server)
        .await;

    let mut config = test_config(server.uri());
    config.enable_thinking = Some(false);
    let agent =
        AgentLoop::from_config(&config, ToolRegistry::new()).expect("test agent should be valid");

    let result = agent
        .run("Answer directly")
        .await
        .expect("run should succeed");

    assert_eq!(
        result.final_message.content.as_deref(),
        Some("thinking disabled")
    );
}

#[tokio::test]
async fn executes_tool_then_returns_final_assistant_message() {
    let server = MockServer::start().await;
    let mut read_file_call = tool_call("read_file", "call-1");
    read_file_call.function.arguments = r#"{"path":"hello.txt"}"#.to_string();
    Mock::given(method("POST"))
        .respond_with(streaming_response(Message::assistant(
            None,
            Some(vec![read_file_call]),
        )))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(streaming_response(Message::assistant(
            Some("The file says: hello from the tool.".to_string()),
            None,
        )))
        .expect(1)
        .mount(&server)
        .await;

    let directory = tempdir().expect("temporary directory should exist");
    fs::write(directory.path().join("hello.txt"), "hello from the tool")
        .expect("fixture should be written");
    let mut config = test_config(server.uri());
    config.workspace = directory.path().to_path_buf();
    let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
        .expect("read_file should register");
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured_events = Arc::clone(&events);
    let agent = AgentLoop::from_config(&config, tools)
        .expect("test agent should be valid")
        .with_event_sink(move |event| {
            captured_events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event);
        });

    let result = agent
        .run("Read hello.txt")
        .await
        .expect("run should succeed");

    assert_eq!(result.steps, 2);
    assert_eq!(result.messages.len(), 4);
    assert_eq!(result.messages[0].role, "user");
    assert_eq!(result.messages[1].role, "assistant");
    assert_eq!(result.messages[2].role, "tool");
    assert_eq!(
        result.messages[2].content.as_deref(),
        Some("hello from the tool")
    );
    assert_eq!(result.messages[3].role, "assistant");

    let events = events.lock().expect("event mutex should not be poisoned");
    assert_eq!(events.len(), 14);
    assert!(matches!(events[0], AgentEvent::RunStarted { .. }));
    assert!(matches!(events[1], AgentEvent::TurnStarted { step: 1, .. }));
    assert!(matches!(
        events[2],
        AgentEvent::ModelStarted { step: 1, .. }
    ));
    assert!(matches!(
        events[3],
        AgentEvent::ToolCallDelta {
            step: 1,
            index: 0,
            ..
        }
    ));
    assert!(matches!(
        events[4],
        AgentEvent::AssistantMessage { step: 1, .. }
    ));
    assert!(matches!(events[5], AgentEvent::ToolStarted { step: 1, .. }));
    assert!(matches!(
        events[6],
        AgentEvent::ToolFinished {
            step: 1,
            is_error: false,
            ..
        }
    ));
    assert!(matches!(
        events[7],
        AgentEvent::TurnFinished { step: 1, .. }
    ));
    assert!(matches!(events[8], AgentEvent::TurnStarted { step: 2, .. }));
    assert!(matches!(
        events[9],
        AgentEvent::ModelStarted { step: 2, .. }
    ));
    assert!(matches!(
        events[10],
        AgentEvent::AssistantDelta {
            step: 2,
            ref content,
            ..
        } if content == "The file says: hello from the tool."
    ));
    assert!(matches!(
        events[11],
        AgentEvent::AssistantMessage { step: 2, .. }
    ));
    assert!(matches!(
        events[12],
        AgentEvent::TurnFinished { step: 2, .. }
    ));
    assert!(matches!(
        events[13],
        AgentEvent::RunFinished {
            outcome: RunOutcome::Completed,
            steps: 2,
            ..
        }
    ));
}

#[tokio::test]
async fn does_not_execute_tools_on_final_allowed_step() {
    let server = MockServer::start().await;
    let unknown_call = tool_call("unknown_tool", "call-1");
    Mock::given(method("POST"))
        .respond_with(streaming_response(Message::assistant(
            None,
            Some(vec![unknown_call]),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let mut config = test_config(server.uri());
    config.max_steps = 1;
    let agent =
        AgentLoop::from_config(&config, ToolRegistry::new()).expect("test agent should be valid");

    let error = agent
        .run("Call a tool")
        .await
        .expect_err("tool call on final step should stop the run");

    assert!(matches!(
        error,
        AgentError::MaxStepsExceeded { max_steps: 1 }
    ));
}

#[tokio::test]
async fn applies_tool_call_limit_from_config_policy() {
    let server = MockServer::start().await;
    let first_call = tool_call("first", "call-1");
    let second_call = tool_call("second", "call-2");
    Mock::given(method("POST"))
        .respond_with(streaming_response(Message::assistant(
            None,
            Some(vec![first_call, second_call]),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let mut config = test_config(server.uri());
    config.max_tool_calls_per_step = 1;
    let agent =
        AgentLoop::from_config(&config, ToolRegistry::new()).expect("test agent should be valid");

    let error = agent
        .run("Call tools")
        .await
        .expect_err("the configured tool limit should be enforced");

    assert!(matches!(
        error,
        AgentError::TooManyToolCalls {
            step: 1,
            count: 2,
            limit: 1
        }
    ));
}

#[tokio::test]
async fn persists_rejected_tool_results_for_an_invalid_tool_batch() {
    let directory = tempdir().expect("temporary directory should exist");
    let session = SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
        .await
        .expect("session should open");
    let responses = Arc::new(Mutex::new(VecDeque::from([Message::assistant(
        None,
        Some(vec![
            tool_call("echo_tool", "duplicate"),
            tool_call("echo_tool", "duplicate"),
        ]),
    )])));
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("agent should be valid")
    .with_session(session.clone());

    let error = agent
        .run("call the tool twice")
        .await
        .expect_err("duplicate tool IDs should be rejected");

    assert!(matches!(
        error,
        AgentError::InvalidToolCall { index: 1, .. }
    ));
    let records = session.load().await.expect("session should load");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].message.role, "user");
    let repaired = session
        .load_repaired()
        .await
        .expect("invalid assistant output must not corrupt the session");
    assert_eq!(repaired.len(), 1);
}

#[tokio::test]
async fn rejects_empty_prompt_before_calling_model() {
    let agent = AgentLoop::new(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("test agent should be valid");

    let error = agent.run("  ").await.expect_err("empty prompt should fail");

    assert!(matches!(error, AgentError::EmptyPrompt));
}

#[test]
fn rejects_zero_max_steps() {
    let error = AgentLoop::new(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        0,
    )
    .err()
    .expect("zero max_steps should fail");

    assert!(matches!(error, AgentError::InvalidMaxSteps));
}

#[test]
fn rejects_zero_tool_call_limit() {
    let error = AgentLoop::new_with_policy(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            max_tool_calls_per_step: 0,
            ..RunPolicy::default()
        },
    )
    .err()
    .expect("zero tool calls should fail validation");

    assert!(matches!(
        error,
        AgentError::InvalidRunPolicy {
            field: "max_tool_calls_per_step"
        }
    ));
}

#[test]
fn rejects_zero_parallel_tool_limit() {
    let error = AgentLoop::new_with_policy(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            max_parallel_tool_calls: 0,
            ..RunPolicy::default()
        },
    )
    .err()
    .expect("zero parallel tools should fail validation");

    assert!(matches!(
        error,
        AgentError::InvalidRunPolicy {
            field: "max_parallel_tool_calls"
        }
    ));
}

#[test]
fn rejects_zero_context_byte_limit() {
    let error = AgentLoop::new_with_policy(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            max_context_bytes: 0,
            ..RunPolicy::default()
        },
    )
    .err()
    .expect("zero context bytes should fail validation");

    assert!(matches!(
        error,
        AgentError::InvalidRunPolicy {
            field: "max_context_bytes"
        }
    ));
}

#[test]
fn model_retry_backoff_is_exponential_and_bounded() {
    assert_eq!(model_retry_delay_ms(250, 1), 250);
    assert_eq!(model_retry_delay_ms(250, 2), 500);
    assert_eq!(model_retry_delay_ms(250, 3), 1_000);
    assert_eq!(model_retry_delay_ms(10_000, 8), 30_000);
}

#[test]
fn stores_run_timeout_from_policy() {
    let agent = AgentLoop::new_with_policy(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            run_timeout_secs: Some(9),
            ..RunPolicy::default()
        },
    )
    .expect("run timeout should be valid");

    assert_eq!(agent.run_timeout, Some(Duration::from_secs(9)));
}

#[test]
fn from_config_stores_run_timeout() {
    let config = Config {
        base_url: "http://localhost:1".to_string(),
        run_timeout_secs: Some(12),
        ..Config::default()
    };

    let agent = AgentLoop::from_config(&config, ToolRegistry::new())
        .expect("configured agent should be valid");

    assert_eq!(agent.run_timeout, Some(Duration::from_secs(12)));
}

#[test]
fn rejects_zero_run_timeout() {
    let error = AgentLoop::new_with_policy(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            run_timeout_secs: Some(0),
            ..RunPolicy::default()
        },
    )
    .err()
    .expect("zero run timeout should fail validation");

    assert!(matches!(
        error,
        AgentError::InvalidRunPolicy {
            field: "run_timeout_secs"
        }
    ));
}

#[test]
fn aggregates_complete_model_usage() {
    let mut stats = RunStats::default();
    stats.record_model_usage(Some(ModelUsage {
        input_tokens: Some(7),
        output_tokens: Some(3),
        total_tokens: Some(10),
    }));
    stats.record_model_usage(Some(ModelUsage {
        input_tokens: Some(11),
        output_tokens: Some(5),
        total_tokens: Some(16),
    }));

    assert_eq!(stats.model_usage(), (Some(18), Some(8), Some(26)));
}

#[test]
fn incomplete_model_usage_is_not_reported_as_zero() {
    let mut stats = RunStats::default();
    stats.record_model_usage(Some(ModelUsage {
        input_tokens: Some(7),
        output_tokens: Some(3),
        total_tokens: Some(10),
    }));
    stats.record_model_usage(None);

    assert_eq!(stats.model_usage(), (None, None, None));
}

#[test]
fn preserves_usage_fields_when_one_provider_field_is_missing() {
    let mut stats = RunStats::default();
    stats.record_model_usage(Some(ModelUsage {
        input_tokens: Some(7),
        output_tokens: Some(3),
        total_tokens: None,
    }));

    assert_eq!(stats.model_usage(), (Some(7), Some(3), None));
}
