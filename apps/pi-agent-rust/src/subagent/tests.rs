use super::progress::{
    child_details, child_error_status, child_error_status_with_outcome, subagent_progress,
    truncate_output,
};
use super::*;
use crate::{
    agent_loop::AgentError,
    config::Config,
    events::{AgentEvent, RunOutcome},
    model::{ChatOptions, ModelClient},
    tools::{ToolContext, ToolError, ToolExecutionContext, ToolRegistry},
};
use pi_agent_runtime::RiskClass;
use serde_json::{json, Value};
use uuid::Uuid;
use wiremock::{matchers::method, Mock, MockServer, ResponseTemplate};

#[test]
fn advertises_a_bounded_delegation_tool() {
    let config = Config::default();
    let mut tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
        .expect("read_file should register");
    tools
        .register_write_file(ToolContext::from_config(&config))
        .expect("write_file should register");
    tools
        .register_edit_file(ToolContext::from_config(&config))
        .expect("edit_file should register");
    let client = ModelClient::new("http://localhost:1", std::time::Duration::from_secs(1))
        .expect("endpoint should be valid");
    let tool = SubagentTool::new(
        client,
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        tools,
        config.run_policy(),
        30,
        1024,
    );
    let definition = tool.definition();

    assert_eq!(definition.function.name, "spawn_subagent");
    assert_eq!(definition.function.parameters["required"], json!(["task"]));
    assert_eq!(
        definition.function.parameters["additionalProperties"],
        false
    );
    assert_eq!(
        tool.tools.risk_class("read_file"),
        Some(RiskClass::ReadOnly)
    );
    assert_eq!(tool.tools.risk_class("write_file"), None);
    assert_eq!(tool.tools.risk_class("edit_file"), None);
}

#[test]
fn truncation_respects_small_utf8_limits() {
    assert_eq!(truncate_output("你好".to_string(), 0), "");
    assert_eq!(truncate_output("你好".to_string(), 1), "");
    assert_eq!(truncate_output("你好".to_string(), 3), "…");
    assert!(truncate_output("a long result".to_string(), 5).len() <= 5);
}

#[test]
fn child_details_preserve_parent_lineage() {
    let parent_run_id = Uuid::from_u128(1);
    let parent_turn_id = Uuid::from_u128(2);
    let child_run_id = Uuid::from_u128(3);
    let context = ToolExecutionContext::new(crate::CancellationToken::new())
        .with_parent_identity(parent_run_id, parent_turn_id, 4, "call-child")
        .with_delegation_depth(2);

    let details = child_details(
        Some(&context),
        Some(child_run_id),
        Some("researcher"),
        "test-model",
        3,
        3,
        "completed",
    );

    assert_eq!(details["parent_run_id"], json!(parent_run_id));
    assert_eq!(details["parent_turn_id"], json!(parent_turn_id));
    assert_eq!(details["parent_step"], json!(4));
    assert_eq!(details["parent_tool_call_id"], json!("call-child"));
    assert_eq!(details["child_run_id"], json!(child_run_id));
    assert_eq!(details["depth"], json!(3));
}

#[test]
fn rejects_invalid_arguments_before_execution() {
    let config = Config::default();
    let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
        .expect("read_file should register");
    let tool = SubagentTool::new(
        ModelClient::new("http://localhost:1", std::time::Duration::from_secs(1))
            .expect("endpoint should be valid"),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        tools,
        config.run_policy(),
        30,
        1024,
    );

    let empty_task = tool
        .validate_arguments(&json!({"task": "   "}))
        .expect_err("empty task should fail preflight");
    assert!(matches!(
        empty_task,
        ToolError::Subagent { message } if message.contains("must not be empty")
    ));

    let zero_steps = tool
        .validate_arguments(&json!({"task": "inspect", "max_steps": 0}))
        .expect_err("zero max_steps should fail preflight");
    assert!(matches!(
        zero_steps,
        ToolError::Subagent { message } if message.contains("at least one model step")
    ));
}

#[tokio::test]
async fn role_and_task_are_sent_in_separate_messages() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"done\"}}]}\n\n"
                            .to_owned()
                            + "data: [DONE]\n\n",
                    ),
            )
            .expect(1)
            .mount(&server)
            .await;

    let config = Config::default();
    let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
        .expect("read_file should register");
    let tool = SubagentTool::new(
        ModelClient::new(&server.uri(), std::time::Duration::from_secs(5))
            .expect("endpoint should be valid"),
        "test-model",
        None,
        ChatOptions {
            temperature: 0.0,
            enable_thinking: Some(false),
        },
        tools,
        config.run_policy(),
        5,
        1024,
    );

    tool.execute_result(json!({
        "task": "inspect the repository",
        "role": "researcher"
    }))
    .await
    .expect("child should complete");

    let requests = server
        .received_requests()
        .await
        .expect("wiremock should return received requests");
    assert_eq!(requests.len(), 1);
    let body: Value =
        serde_json::from_slice(&requests[0].body).expect("model request should be valid JSON");
    assert_eq!(body["messages"][0]["role"], "system");
    assert!(body["messages"][0]["content"]
        .as_str()
        .is_some_and(|content| content.contains("researcher")));
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"][1]["content"], "inspect the repository");
}

#[test]
fn summarizes_child_lifecycle_without_forwarding_transcript() {
    let context = crate::events::RunContext::new(None);
    let update = subagent_progress(&AgentEvent::ModelStarted {
        context,
        step: 2,
        model: "qwen3.8".to_string(),
    })
    .expect("model start should produce progress");

    assert_eq!(update.details["status"], "model_started");
    assert_eq!(update.details["step"], 2);
    assert!(update.content.contains("qwen3.8"));
    assert!(subagent_progress(&AgentEvent::AssistantMessage {
        context,
        step: 2,
        message: crate::message::Message::assistant(Some("private answer".to_string()), None),
    })
    .is_none());
}

#[test]
fn classifies_child_terminal_errors_for_structured_results() {
    assert_eq!(child_error_status(&AgentError::RunCancelled), "cancelled");
    assert_eq!(
        child_error_status(&AgentError::RunTimedOut { timeout_ms: 10 }),
        "timed_out"
    );
    assert_eq!(
        child_error_status(&AgentError::RuntimeActionUnknown {
            action_id: uuid::Uuid::new_v4(),
            tool_name: "write_file".to_owned(),
            reason: "executor disconnected".to_owned(),
        }),
        "unknown"
    );
    assert_eq!(child_error_status(&AgentError::EmptyPrompt), "failed");
    assert_eq!(
        child_error_status_with_outcome(&AgentError::RunCancelled, Some(RunOutcome::Unknown)),
        "unknown"
    );
    assert_eq!(
        child_error_status_with_outcome(&AgentError::RunCancelled, Some(RunOutcome::Cancelled)),
        "cancelled"
    );
}
