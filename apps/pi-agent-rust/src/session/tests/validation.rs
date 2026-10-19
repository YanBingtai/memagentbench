use super::*;

#[tokio::test]
async fn rejects_duplicate_tool_results() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("duplicate-tool-result.jsonl");
    let session_id = Uuid::new_v4();
    let store = SessionStore::open(&path, session_id)
        .await
        .expect("session should open");
    store
        .append(&Message::assistant(None, Some(vec![tool_call("call-1")])))
        .await
        .expect("assistant call should append");
    store
        .append(&Message::tool("call-1", "first"))
        .await
        .expect("first result should append");
    store
        .append(&Message::tool("call-1", "duplicate"))
        .await
        .expect("duplicate fixture should append");

    let error = store
        .load_repaired()
        .await
        .expect_err("duplicate tool results must be rejected");
    assert!(matches!(
        error,
        SessionError::DuplicateToolResult { call_id } if call_id == "call-1"
    ));
}

#[tokio::test]
async fn rejects_tool_results_without_a_known_call() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("unknown-tool-result.jsonl");
    let session_id = Uuid::new_v4();
    let store = SessionStore::open(&path, session_id)
        .await
        .expect("session should open");
    store
        .append(&Message::tool("unknown-call", "unexpected"))
        .await
        .expect("unexpected result fixture should append");

    let error = store
        .load_repaired()
        .await
        .expect_err("unknown tool results must be rejected");
    assert!(matches!(
        error,
        SessionError::UnexpectedToolResult { call_id } if call_id == "unknown-call"
    ));
}

#[tokio::test]
async fn rejects_a_tool_result_before_its_assistant_call() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("out-of-order-tool-result.jsonl");
    let session_id = Uuid::new_v4();
    let store = SessionStore::open(&path, session_id)
        .await
        .expect("session should open");
    store
        .append(&Message::tool("call-1", "result before call"))
        .await
        .expect("out-of-order result fixture should append");
    store
        .append(&Message::assistant(None, Some(vec![tool_call("call-1")])))
        .await
        .expect("assistant call fixture should append");

    let error = store
        .load_repaired()
        .await
        .expect_err("out-of-order tool result must be rejected immediately");
    assert!(matches!(
        error,
        SessionError::UnexpectedToolResult { call_id } if call_id == "call-1"
    ));
}

#[tokio::test]
async fn rejects_tool_results_without_call_ids() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("missing-tool-result-id.jsonl");
    let session_id = Uuid::new_v4();
    let store = SessionStore::open(&path, session_id)
        .await
        .expect("session should open");
    store
        .append(&Message {
            role: "tool".to_string(),
            content: Some("missing id".to_string()),
            reasoning_content: None,
            stop_reason: None,
            name: None,
            tool_call_id: None,
            tool_result: None,
            tool_calls: None,
        })
        .await
        .expect("malformed result fixture should append");

    let error = store
        .load_repaired()
        .await
        .expect_err("tool results without ids must be rejected");
    assert!(matches!(error, SessionError::ToolResultMissingCallId));
}
