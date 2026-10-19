use super::*;

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
