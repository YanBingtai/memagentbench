use super::*;

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
