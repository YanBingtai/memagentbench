use super::*;

#[tokio::test]
async fn rejects_empty_prompt_before_calling_model() {
    let agent = AgentLoop::new(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        1,
    )
    .expect("test agent should be valid");

    let error = agent.run("  ").await.expect_err("empty prompt should fail");

    assert!(matches!(error, AgentError::EmptyPrompt));
}

#[test]
fn rejects_zero_max_steps() {
    let error = AgentLoop::new(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        0,
    )
    .err()
    .expect("zero max_steps should fail");

    assert!(matches!(error, AgentError::InvalidMaxSteps));
}

#[test]
fn rejects_zero_tool_call_limit() {
    let error = AgentLoop::new_with_policy(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            max_tool_calls_per_step: 0,
            ..RunPolicy::default()
        },
    )
    .err()
    .expect("zero tool calls should fail validation");

    assert!(matches!(
        error,
        AgentError::InvalidRunPolicy {
            field: "max_tool_calls_per_step"
        }
    ));
}

#[test]
fn rejects_zero_parallel_tool_limit() {
    let error = AgentLoop::new_with_policy(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            max_parallel_tool_calls: 0,
            ..RunPolicy::default()
        },
    )
    .err()
    .expect("zero parallel tools should fail validation");

    assert!(matches!(
        error,
        AgentError::InvalidRunPolicy {
            field: "max_parallel_tool_calls"
        }
    ));
}

#[test]
fn rejects_zero_context_byte_limit() {
    let error = AgentLoop::new_with_policy(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            max_context_bytes: 0,
            ..RunPolicy::default()
        },
    )
    .err()
    .expect("zero context bytes should fail validation");

    assert!(matches!(
        error,
        AgentError::InvalidRunPolicy {
            field: "max_context_bytes"
        }
    ));
}

#[test]
fn model_retry_backoff_is_exponential_and_bounded() {
    assert_eq!(model_retry_delay_ms(250, 1), 250);
    assert_eq!(model_retry_delay_ms(250, 2), 500);
    assert_eq!(model_retry_delay_ms(250, 3), 1_000);
    assert_eq!(model_retry_delay_ms(10_000, 8), 30_000);
}

#[test]
fn stores_run_timeout_from_policy() {
    let agent = AgentLoop::new_with_policy(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            run_timeout_secs: Some(9),
            ..RunPolicy::default()
        },
    )
    .expect("run timeout should be valid");

    assert_eq!(agent.run_timeout, Some(Duration::from_secs(9)));
}

#[test]
fn from_config_stores_run_timeout() {
    let config = Config {
        base_url: "http://localhost:1".to_string(),
        run_timeout_secs: Some(12),
        ..Config::default()
    };

    let agent = AgentLoop::from_config(&config, ToolRegistry::new())
        .expect("configured agent should be valid");

    assert_eq!(agent.run_timeout, Some(Duration::from_secs(12)));
}

#[test]
fn rejects_zero_run_timeout() {
    let error = AgentLoop::new_with_policy(
        ModelClient::new("http://localhost:1", Duration::from_secs(1))
            .expect("endpoint should be valid"),
        ToolRegistry::new(),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        RunPolicy {
            run_timeout_secs: Some(0),
            ..RunPolicy::default()
        },
    )
    .err()
    .expect("zero run timeout should fail validation");

    assert!(matches!(
        error,
        AgentError::InvalidRunPolicy {
            field: "run_timeout_secs"
        }
    ));
}

#[test]
fn aggregates_complete_model_usage() {
    let mut stats = RunStats::default();
    stats.record_model_usage(Some(ModelUsage {
        input_tokens: Some(7),
        output_tokens: Some(3),
        total_tokens: Some(10),
    }));
    stats.record_model_usage(Some(ModelUsage {
        input_tokens: Some(11),
        output_tokens: Some(5),
        total_tokens: Some(16),
    }));

    assert_eq!(stats.model_usage(), (Some(18), Some(8), Some(26)));
}

#[test]
fn incomplete_model_usage_is_not_reported_as_zero() {
    let mut stats = RunStats::default();
    stats.record_model_usage(Some(ModelUsage {
        input_tokens: Some(7),
        output_tokens: Some(3),
        total_tokens: Some(10),
    }));
    stats.record_model_usage(None);

    assert_eq!(stats.model_usage(), (None, None, None));
}

#[test]
fn preserves_usage_fields_when_one_provider_field_is_missing() {
    let mut stats = RunStats::default();
    stats.record_model_usage(Some(ModelUsage {
        input_tokens: Some(7),
        output_tokens: Some(3),
        total_tokens: None,
    }));

    assert_eq!(stats.model_usage(), (Some(7), Some(3), None));
}
