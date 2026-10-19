use super::*;

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
