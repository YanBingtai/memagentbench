use super::*;

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
