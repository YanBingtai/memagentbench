use super::*;

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
