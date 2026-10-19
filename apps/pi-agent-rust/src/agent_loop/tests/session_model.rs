use super::*;

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
