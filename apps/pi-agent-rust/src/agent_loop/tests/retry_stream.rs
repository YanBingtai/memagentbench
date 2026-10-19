use super::*;

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
