use super::*;

#[test]
fn memory_context_keeps_source_identity_and_evidence_references() {
    let context = format_memory_context(&[MemoryInjection {
        candidate_id: Uuid::from_u128(1),
        projection: MemoryProjection {
            projection_id: Uuid::from_u128(2),
            source_memory_id: Uuid::from_u128(3),
            source_version: 4,
            content: "视频中的红色方块".to_owned(),
            evidence_refs: vec![json!({
                "artifact_id": "artifact-7",
                "uri": "cas://episode-7.mp4",
                "frame_start": 360,
                "step_start": 30
            })],
            provenance: "video-memory:episode-7".to_owned(),
            token_cost: 12,
        },
    }]);

    assert!(context.contains("source_memory_id=00000000-0000-0000-0000-000000000003"));
    assert!(context.contains("version=4"));
    assert!(context.contains("artifact-7"));
    assert!(context.contains("frame_start"));
    assert!(context.contains("step_start"));
}

#[tokio::test]
async fn evaluates_and_injects_memory_before_the_model_request() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let emitted_events = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("answer with memory".to_owned()), None),
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
    .expect("memory hook agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone())
    .with_event_sink({
        let emitted_events = Arc::clone(&emitted_events);
        move |event| {
            emitted_events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event);
        }
    })
    .with_memory_hook(
        MemoryHookConfig {
            namespace: "bench/text".to_owned(),
            max_tokens: 32,
            ..MemoryHookConfig::default()
        },
        |request| {
            Box::pin(async move {
                Ok(Some(MemoryProposal {
                    proposal_id: Uuid::from_u128(7),
                    run_id: request.context.run_id,
                    namespace: request.context.namespace.clone(),
                    base_revision: request.context.current_revision,
                    memory_revision: request.context.memory_revision,
                    policy_revision: request.context.policy_revision,
                    candidates: vec![MemoryCandidate {
                        candidate_id: Uuid::from_u128(8),
                        projection: MemoryProjection {
                            projection_id: Uuid::from_u128(9),
                            source_memory_id: Uuid::from_u128(10),
                            source_version: 1,
                            content: "previous parser fix".to_owned(),
                            evidence_refs: Vec::new(),
                            provenance: "text-memory:record-1".to_owned(),
                            token_cost: 5,
                        },
                        score: 0.9,
                        confidence: 0.8,
                        expires_at_ms: None,
                    }],
                    created_at_ms: request.context.now_ms,
                    expires_at_ms: Some(request.context.now_ms + 1_000),
                    retrieval_summary: None,
                }))
            })
        },
    );

    agent
        .run("fix the parser")
        .await
        .expect("run should succeed");

    let requests = requests
        .lock()
        .expect("request mutex should not be poisoned");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].len(), 2);
    assert_eq!(requests[0][0].role, "system");
    assert!(requests[0][0]
        .content
        .as_deref()
        .is_some_and(|content| content.contains("previous parser fix")));
    assert_eq!(requests[0][1].content.as_deref(), Some("fix the parser"));

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("memory decision should be readable");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].draft.kind.as_str(), "memory_injected");

    let emitted_events = emitted_events
        .lock()
        .expect("event mutex should not be poisoned");
    assert!(emitted_events.iter().any(|event| matches!(
        event,
        AgentEvent::MemoryDecision {
            step: 1,
            proposal_id,
            event_sequence: 1,
            decision_kind,
            injection_count: 1,
            total_token_cost: 5,
            skip_reason: None,
            ..
        } if *proposal_id == Uuid::from_u128(7) && decision_kind == "injected"
    )));
}

#[tokio::test]
async fn rejects_memory_that_would_exceed_context_before_runtime_audit() {
    let directory = tempdir().expect("temporary directory should exist");
    let ledger_path = directory.path().join("runtime.jsonl");
    let agent = AgentLoop::new_with_policy(
        RecordingChatModel {
            response: Message::assistant(Some("unused".to_owned()), None),
            requests: Arc::new(Mutex::new(Vec::new())),
        },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            max_context_bytes: 128,
            ..RunPolicy::default()
        },
    )
    .expect("memory budget agent should be valid")
    .with_runtime_ledger_file(ledger_path.clone())
    .with_memory_hook(MemoryHookConfig::default(), |request| {
        Box::pin(async move {
            Ok(Some(MemoryProposal {
                proposal_id: Uuid::from_u128(17),
                run_id: request.context.run_id,
                namespace: request.context.namespace.clone(),
                base_revision: request.context.current_revision,
                memory_revision: request.context.memory_revision,
                policy_revision: request.context.policy_revision,
                candidates: vec![MemoryCandidate {
                    candidate_id: Uuid::from_u128(18),
                    projection: MemoryProjection {
                        projection_id: Uuid::from_u128(19),
                        source_memory_id: Uuid::from_u128(20),
                        source_version: 1,
                        content: "x".repeat(512),
                        evidence_refs: Vec::new(),
                        provenance: "oversized-memory".to_owned(),
                        token_cost: 128,
                    },
                    score: 1.0,
                    confidence: 1.0,
                    expires_at_ms: None,
                }],
                created_at_ms: request.context.now_ms,
                expires_at_ms: None,
                retrieval_summary: None,
            }))
        })
    });

    let error = agent
        .run("prompt")
        .await
        .expect_err("oversized memory should fail before the model request");
    assert!(
        matches!(error, AgentError::ContextTooLarge { .. }),
        "{error:?}"
    );

    let ledger = JsonlLedger::open(&ledger_path).expect("runtime ledger should reopen");
    let events = ledger
        .scan_after(EventSequence::INITIAL)
        .expect("runtime events should be readable");
    assert!(events.is_empty());
}

#[tokio::test]
async fn system_prompt_is_ephemeral_and_precedes_session_messages() {
    let directory = tempdir().expect("temporary directory should exist");
    let session = SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
        .await
        .expect("session should open");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("done".to_owned()), None),
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
    .expect("system prompt agent should be valid")
    .with_system_prompt("You are a careful coding agent.")
    .with_session(session.clone());

    agent
        .run("inspect the project")
        .await
        .expect("run should succeed");

    {
        let requests = requests
            .lock()
            .expect("request mutex should not be poisoned");
        assert_eq!(requests[0][0].role, "system");
        assert_eq!(
            requests[0][0].content.as_deref(),
            Some("You are a careful coding agent.")
        );
        assert_eq!(
            requests[0][1].content.as_deref(),
            Some("inspect the project")
        );
    }

    let records = session.load().await.expect("session should load");
    assert!(records.iter().all(|record| {
        record.message.content.as_deref() != Some("You are a careful coding agent.")
    }));
}

#[tokio::test]
async fn injects_skill_index_into_model_context() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut catalog = SkillCatalog::new();
    catalog
        .insert(
            Skill::new(
                "research",
                "Research tasks",
                "Use primary sources.",
                "/skills/research/SKILL.md",
            )
            .expect("test skill should be valid"),
        )
        .expect("test catalog should accept the skill");

    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("done".to_string()), None),
            requests: Arc::clone(&requests),
        },
        ToolRegistry::new(),
        "recording-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("recording model agent should be valid")
    .with_skills(catalog);

    assert!(agent
        .tools
        .model_definitions()
        .iter()
        .any(|definition| definition.function.name == "read_skill"));

    agent
        .run("find evidence")
        .await
        .expect("run should succeed");

    let requests = requests
        .lock()
        .expect("request mutex should not be poisoned");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0][0].role, "system");
    assert!(requests[0][0]
        .content
        .as_deref()
        .is_some_and(|content| content.contains("<name>research</name>")));
    assert_eq!(
        requests[0].last().map(|message| message.role.as_str()),
        Some("user")
    );
}

#[tokio::test]
async fn rejects_a_second_run_while_the_agent_is_active() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let agent = Arc::new(
        AgentLoop::new(
            BlockingChatModel {
                started: started.clone(),
                release: release.clone(),
                response: Message::assistant(Some("done".to_string()), None),
            },
            ToolRegistry::new(),
            "blocking-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("blocking agent should be valid"),
    );
    let first_agent = Arc::clone(&agent);
    let first = tokio::spawn(async move { first_agent.run("first").await });
    started.notified().await;

    let error = agent
        .run("second")
        .await
        .expect_err("a concurrent run should be rejected");
    assert!(matches!(error, AgentError::RunBusy));

    release.notify_one();
    first
        .await
        .expect("first run task should join")
        .expect("first run should complete");
}

#[tokio::test]
async fn steering_is_injected_before_the_next_model_request() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("answer".to_string()), None),
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
    .expect("steering agent should be valid");
    agent
        .steer("additional constraint")
        .expect("steering should queue");

    agent.run("initial task").await.expect("run should succeed");

    let requests = requests.lock().expect("request mutex should not poison");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0][0].content.as_deref(), Some("initial task"));
    assert_eq!(
        requests[0][1].content.as_deref(),
        Some("additional constraint")
    );
}

#[tokio::test]
async fn follow_up_starts_after_a_natural_assistant_stop() {
    let responses = Arc::new(Mutex::new(VecDeque::from([
        Message::assistant(Some("first answer".to_string()), None),
        Message::assistant(Some("follow-up answer".to_string()), None),
    ])));
    let agent = AgentLoop::new(
        SequenceChatModel { responses },
        ToolRegistry::new(),
        "fake-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        3,
    )
    .expect("follow-up agent should be valid");
    agent
        .follow_up("continue with a second task")
        .expect("follow-up should queue");

    let result = agent.run("first task").await.expect("run should succeed");

    assert_eq!(result.steps, 2);
    assert_eq!(
        result.final_message.content.as_deref(),
        Some("follow-up answer")
    );
    assert_eq!(result.messages.len(), 4);
    assert_eq!(
        result.messages[2].content.as_deref(),
        Some("continue with a second task")
    );
}

#[tokio::test]
async fn rejects_idle_control_messages_after_the_first_run() {
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
    .expect("agent should be valid");

    agent
        .run("first task")
        .await
        .expect("first run should succeed");

    assert!(matches!(
        agent.steer("stale steering"),
        Err(AgentError::ControlQueueRequiresActiveRun { kind: "steering" })
    ));
    assert!(matches!(
        agent.follow_up("stale follow-up"),
        Err(AgentError::ControlQueueRequiresActiveRun { kind: "follow-up" })
    ));
}

#[tokio::test]
async fn clears_unconsumed_control_messages_when_a_run_finishes() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("done".to_string()), None),
            requests: Arc::clone(&requests),
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
    .expect("agent should be valid");
    agent
        .follow_up("stale follow-up")
        .expect("initial follow-up should be queueable");

    agent
        .run("first task")
        .await
        .expect("first run should succeed");
    agent
        .run("second task")
        .await
        .expect("second run should succeed");

    let requests = requests.lock().expect("request mutex should not poison");
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].len(), 1);
    assert_eq!(requests[1].len(), 1);
    assert_eq!(requests[1][0].content.as_deref(), Some("second task"));
}

#[tokio::test]
async fn steering_prompt_is_injected_before_the_next_model_request() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("done".to_owned()), None),
            requests: Arc::clone(&requests),
        },
        ToolRegistry::new(),
        "recording-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("recording model agent should be valid");
    agent
        .steer("stop and inspect the result")
        .expect("steering prompt should be queued");

    agent
        .run("original request")
        .await
        .expect("run should succeed");

    let requests = requests
        .lock()
        .expect("request mutex should not be poisoned");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0][0].content.as_deref(), Some("original request"));
    assert_eq!(
        requests[0][1].content.as_deref(),
        Some("stop and inspect the result")
    );
}

#[tokio::test]
async fn follow_up_prompt_runs_after_the_current_assistant_turn() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = AgentLoop::new(
        RecordingChatModel {
            response: Message::assistant(Some("done".to_owned()), None),
            requests: Arc::clone(&requests),
        },
        ToolRegistry::new(),
        "recording-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        2,
    )
    .expect("recording model agent should be valid");
    agent
        .follow_up("now summarize it")
        .expect("follow-up prompt should be queued");

    let result = agent
        .run("original request")
        .await
        .expect("run should succeed");

    let requests = requests
        .lock()
        .expect("request mutex should not be poisoned");
    assert_eq!(result.steps, 2);
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].len(), 1);
    assert_eq!(requests[0][0].content.as_deref(), Some("original request"));
    assert_eq!(requests[1].len(), 3);
    assert_eq!(requests[1][2].content.as_deref(), Some("now summarize it"));
}
