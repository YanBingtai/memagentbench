use super::*;

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
