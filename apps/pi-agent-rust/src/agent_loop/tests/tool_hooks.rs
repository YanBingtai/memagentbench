use super::*;

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
