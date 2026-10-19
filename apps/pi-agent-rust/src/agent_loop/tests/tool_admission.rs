use super::*;

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
