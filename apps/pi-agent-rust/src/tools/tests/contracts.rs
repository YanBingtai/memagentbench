use super::*;

#[test]
fn execution_context_reports_cancellation_and_updates() {
    let cancellation = CancellationToken::new();
    let updates = Arc::new(Mutex::new(Vec::new()));
    let context = ToolExecutionContext::new(cancellation.clone()).with_update_sink({
        let updates = Arc::clone(&updates);
        move |update| {
            updates
                .lock()
                .expect("update mutex should not poison")
                .push(update)
        }
    });

    context.update(ToolUpdate::text("started"));
    assert!(!context.is_cancelled());
    let child_token = context.cancellation_token();
    cancellation.cancel();
    assert!(context.is_cancelled());
    assert!(child_token.is_cancelled());
    assert_eq!(
        updates
            .lock()
            .expect("update mutex should not poison")
            .len(),
        1
    );
}

#[test]
fn execution_context_bounds_progress_updates() {
    let updates = Arc::new(Mutex::new(Vec::new()));
    let context = ToolExecutionContext::new(CancellationToken::new())
        .with_output_limit(32)
        .with_update_sink({
            let updates = Arc::clone(&updates);
            move |update| {
                updates
                    .lock()
                    .expect("update mutex should not poison")
                    .push(update)
            }
        });

    context.update(ToolUpdate::text("你好".repeat(32)).with_details(json!({
        "payload": "x".repeat(128)
    })));

    let updates = updates.lock().expect("update mutex should not poison");
    assert_eq!(updates.len(), 1);
    assert!(updates[0].content.len() <= 32);
    assert!(updates[0].details.is_null());
}

#[tokio::test]
async fn registry_closes_progress_updates_after_tool_completion() {
    let retained = Arc::new(Mutex::new(None));
    let updates = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry
        .register(RetainedContextTool {
            retained: Arc::clone(&retained),
        })
        .expect("retained context tool should register");
    let call = ToolCall {
        id: "retained-call".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "retained_context".to_string(),
            arguments: "{}".to_string(),
        },
    };
    let context = ToolExecutionContext::new(CancellationToken::new()).with_update_sink({
        let updates = Arc::clone(&updates);
        move |update| {
            updates
                .lock()
                .expect("updates mutex should not be poisoned")
                .push(update.content);
        }
    });

    registry
        .execute_result_with_context(&call, context)
        .await
        .expect("tool should complete");
    retained
        .lock()
        .expect("retained context mutex should not be poisoned")
        .as_ref()
        .expect("tool should retain its context")
        .update(ToolUpdate::text("late"));

    assert_eq!(
        *updates
            .lock()
            .expect("updates mutex should not be poisoned"),
        vec!["during"]
    );
}

#[tokio::test]
async fn registry_bounds_custom_tool_results() {
    let mut registry = ToolRegistry::new().with_output_limit(4_096);
    registry
        .register(LargeResultTool)
        .expect("large result tool should register");
    let call = ToolCall {
        id: "large-call".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "large_result".to_string(),
            arguments: "{}".to_string(),
        },
    };

    let result = registry
        .execute_result(&call)
        .await
        .expect("large result tool should execute");

    assert!(result.content.len() <= 4_096);
    assert_eq!(result.details["status"], "details_truncated");
    assert!(result.is_error);
    assert!(result.terminate);
}

#[tokio::test]
async fn registry_preserves_uncertain_status_when_bounding_details() {
    let mut registry = ToolRegistry::new().with_output_limit(256);
    registry
        .register(LargeUnknownResultTool)
        .expect("uncertain result tool should register");
    let call = ToolCall {
        id: "unknown-call".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "large_unknown_result".to_string(),
            arguments: "{}".to_string(),
        },
    };

    let result = registry
        .execute_result(&call)
        .await
        .expect("uncertain result tool should execute");

    assert_eq!(result.details["status"], "unknown");
    assert_eq!(result.details["details_truncated"], true);
}

#[test]
fn tool_result_bounding_preserves_reconciled_status() {
    let result = ToolResult::text("confirmed")
        .with_details(json!({
            "status": "reconciled",
            "evidence": "x".repeat(1_000),
        }))
        .bounded(64);

    assert_eq!(result.details["status"], "reconciled");
}

#[test]
fn bounded_details_falls_back_to_minimal_uncertain_status() {
    let result = ToolResult::error("uncertain")
        .with_details(json!({
            "status": "unknown",
            "payload": "x".repeat(128),
        }))
        .bounded(20);

    assert_eq!(result.details, json!({"status": "unknown"}));
}
