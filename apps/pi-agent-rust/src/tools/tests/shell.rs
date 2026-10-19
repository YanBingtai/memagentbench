use super::*;

#[tokio::test]
async fn bash_executes_in_workspace_and_reports_command_metadata() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_shell(context)
        .expect("bash should register");

    let result = registry
        .execute_result(&shell_call(json!({
            "command": "printf 'hello\\n'; pwd"
        })))
        .await
        .expect("bash should execute");

    assert!(!result.is_error);
    assert!(result.content.contains("hello"));
    assert!(result
        .content
        .contains(&directory.path().to_string_lossy().to_string()));
    assert_eq!(result.details["status"], "completed");
    assert_eq!(result.details["sandboxed"], false);
    assert_eq!(
        registry.concurrency("bash"),
        Some(ToolConcurrency::Exclusive)
    );
    assert_eq!(
        registry.replay_policy("bash"),
        Some(ToolReplayPolicy::Never)
    );
    assert_eq!(registry.risk_class("bash"), Some(RiskClass::Irreversible));
}

#[tokio::test]
#[ignore = "requires bubblewrap user and network namespaces"]
async fn bash_executes_inside_the_managed_sandbox() {
    let directory = tempdir().expect("temporary directory should exist");
    let tool = ShellTool::with_sandbox_mode(
        ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        },
        SandboxMode::Managed,
    )
    .expect("managed sandbox should be available");

    let result = tool
        .execute_result(json!({
            "command": "test -e /workspace && test ! -e /etc/shadow && printf 'sandbox-ok'"
        }))
        .await
        .expect("managed bash should execute");

    assert!(!result.is_error);
    assert_eq!(result.content, "sandbox-ok");
    assert_eq!(result.details["sandboxed"], true);
}

#[test]
fn bash_rejects_empty_and_out_of_range_arguments() {
    let tool = ShellTool::new(ToolContext {
        workspace: PathBuf::from("/tmp"),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    });

    for arguments in [
        json!({"command": ""}),
        json!({"command": "echo ok", "timeout": 0}),
    ] {
        let error = tool
            .validate_arguments(&arguments)
            .expect_err("invalid bash arguments should fail");
        assert!(matches!(
            error,
            ToolError::InvalidArgumentValue { name, .. } if name == "bash"
        ));
    }
}

#[tokio::test]
async fn bash_reports_nonzero_exit_and_bounded_output() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 128,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_shell(context)
        .expect("bash should register");

    let failed = registry
        .execute_result(&shell_call(json!({
            "command": "printf 'failure\\n' >&2; exit 7"
        })))
        .await
        .expect("nonzero exit should be a structured tool result");
    assert!(failed.is_error);
    assert!(failed.content.contains("failure"));
    assert_eq!(failed.details["status"], "exited_nonzero");
    assert_eq!(failed.details["status_code"], 7);

    let bounded = registry
        .execute_result(&shell_call(json!({
            "command": "printf '%*s' 256 x"
        })))
        .await
        .expect("bounded output should be a structured tool result");
    assert!(bounded.is_error);
    assert_eq!(bounded.details["status"], "truncated");
}

#[tokio::test]
async fn bash_honors_the_per_call_timeout() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_shell(context)
        .expect("bash should register");

    let result = registry
        .execute_result(&shell_call(json!({
            "command": "sleep 2",
            "timeout": 1
        })))
        .await
        .expect("timeout should be a structured tool result");

    assert!(result.is_error);
    assert_eq!(result.details["status"], "timed_out");
    assert_eq!(result.details["timeout_ms"], 1_000);
}
