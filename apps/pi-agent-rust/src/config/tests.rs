use std::fs;

use tempfile::tempdir;

use super::{Config, RunPolicy};

#[test]
fn thinking_setting_loads_with_defaults_and_explicit_overrides() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("config.toml");

    for (raw, expected) in [
        ("", Some(true)),
        ("enable_thinking = true", Some(true)),
        ("enable_thinking = false", Some(false)),
    ] {
        fs::write(&path, raw).expect("config fixture should be written");

        let config = Config::load(&path).expect("config should load");

        assert_eq!(config.enable_thinking, expected, "config: {raw:?}");
    }
}

#[test]
fn thinking_setting_rejects_non_boolean_values() {
    let error = toml::from_str::<Config>("enable_thinking = \"false\"")
        .expect_err("thinking must be a TOML boolean, not a string");

    assert!(error.message().contains("boolean"));
}

#[test]
fn system_prompt_is_optional_and_loads_from_toml() {
    let config: Config = toml::from_str(r#"system_prompt = "Act as a careful coding agent.""#)
        .expect("system prompt should parse");

    assert_eq!(
        config.system_prompt.as_deref(),
        Some("Act as a careful coding agent.")
    );
    assert_eq!(Config::default().system_prompt, None);
}

#[test]
fn run_policy_uses_safe_defaults() {
    let policy = Config::default().run_policy();

    assert_eq!(policy.max_steps, 8);
    assert_eq!(policy.max_tool_calls_per_step, 16);
    assert_eq!(policy.max_parallel_tool_calls, 4);
    assert_eq!(policy.max_model_retries, 2);
    assert_eq!(policy.model_retry_backoff_ms, 250);
    assert_eq!(policy.tool_timeout_secs, 120);
    assert_eq!(policy.run_timeout_secs, None);
    assert_eq!(policy.max_context_messages, 128);
    assert_eq!(policy.max_context_bytes, 1_048_576);
}

#[test]
fn run_policy_loads_explicit_toml_values() {
    let config: Config = toml::from_str(
        r#"
            max_steps = 4
            max_tool_calls_per_step = 6
            max_parallel_tool_calls = 3
            max_model_retries = 5
            model_retry_backoff_ms = 400
            tool_timeout_secs = 30
            run_timeout_secs = 300
            max_context_messages = 64
            max_context_bytes = 65536
        "#,
    )
    .expect("runtime policy should parse");
    let policy = config.run_policy();

    assert_eq!(
        policy,
        RunPolicy {
            max_steps: 4,
            max_tool_calls_per_step: 6,
            max_parallel_tool_calls: 3,
            max_model_retries: 5,
            model_retry_backoff_ms: 400,
            tool_timeout_secs: 30,
            run_timeout_secs: Some(300),
            max_context_messages: 64,
            max_context_bytes: 65_536,
        }
    );
}
