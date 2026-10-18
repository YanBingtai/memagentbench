use super::limits::MAX_SESSION_RECORD_BYTES;
use super::*;
use crate::message::{FunctionCall, Message, ToolCall};
use tempfile::tempdir;
use tokio::fs;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

fn tool_call(id: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "read_file".to_string(),
            arguments: r#"{"path":"note.txt"}"#.to_string(),
        },
    }
}

#[tokio::test]
async fn appends_and_reloads_messages_in_sequence() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("sessions/conversation.jsonl");
    let session_id = Uuid::new_v4();
    let store = SessionStore::open(&path, session_id)
        .await
        .expect("session should open");

    let first = store
        .append(&Message::user("hello"))
        .await
        .expect("first message should append");
    let second = store
        .append(&Message::assistant(Some("hi".to_string()), None))
        .await
        .expect("second message should append");
    let records = store.load().await.expect("session should load");

    assert_eq!(first.sequence, 0);
    assert_eq!(second.sequence, 1);
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].message, Message::user("hello"));
    assert_eq!(records[1].message.role, "assistant");
}

#[tokio::test]
async fn separate_sessions_share_a_file_without_mixing_records() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("conversation.jsonl");
    let first_id = Uuid::new_v4();
    let second_id = Uuid::new_v4();
    let first = SessionStore::open(&path, first_id)
        .await
        .expect("first session should open");
    let second = SessionStore::open(&path, second_id)
        .await
        .expect("second session should open");

    first
        .append(&Message::user("first"))
        .await
        .expect("first message should append");
    second
        .append(&Message::user("second"))
        .await
        .expect("second message should append");

    assert_eq!(first.load().await.expect("first should load").len(), 1);
    assert_eq!(second.load().await.expect("second should load").len(), 1);
}

#[tokio::test]
async fn separate_stores_serialize_same_session_sequences() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("conversation.jsonl");
    let session_id = Uuid::new_v4();
    let first = SessionStore::open(&path, session_id)
        .await
        .expect("first session should open");
    let second = SessionStore::open(&path, session_id)
        .await
        .expect("second session should open");

    let first_message = Message::user("first");
    let second_message = Message::user("second");
    let (first_result, second_result) =
        tokio::join!(first.append(&first_message), second.append(&second_message),);
    let first_record = first_result.expect("first append should succeed");
    let second_record = second_result.expect("second append should succeed");
    let mut sequences = [first_record.sequence, second_record.sequence];
    sequences.sort_unstable();

    assert_eq!(sequences, [0, 1]);
    assert_eq!(
        SessionStore::open(&path, session_id)
            .await
            .expect("session should reopen")
            .load()
            .await
            .expect("session should load")
            .len(),
        2
    );
}

#[tokio::test]
async fn malformed_json_reports_the_line_number() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("broken.jsonl");
    fs::write(&path, "\nnot-json\n")
        .await
        .expect("fixture should be written");

    let error = SessionStore::open(&path, Uuid::new_v4())
        .await
        .expect_err("malformed JSON should fail");

    assert!(matches!(error, SessionError::Decode { line: 2, .. }));
}

#[tokio::test]
async fn rejects_an_oversized_session_record() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("oversized.jsonl");
    fs::write(&path, vec![b'x'; MAX_SESSION_RECORD_BYTES + 1])
        .await
        .expect("oversized fixture should be written");

    let error = SessionStore::open(&path, Uuid::new_v4())
        .await
        .expect_err("oversized record should be rejected");

    assert!(matches!(
        error,
        SessionError::RecordTooLarge {
            line: 1,
            limit: MAX_SESSION_RECORD_BYTES,
            ..
        }
    ));
}

#[tokio::test]
async fn recovers_from_an_incomplete_final_record() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("recoverable.jsonl");
    let session_id = Uuid::new_v4();
    let store = SessionStore::open(&path, session_id)
        .await
        .expect("session should open");
    store
        .append(&Message::user("complete"))
        .await
        .expect("complete message should append");

    let mut file = OpenOptions::new()
        .append(true)
        .open(&path)
        .await
        .expect("session file should open");
    file.write_all(b"{\"session_id\":")
        .await
        .expect("partial record should append");
    file.flush().await.expect("partial record should flush");

    let reopened = SessionStore::open(&path, session_id)
        .await
        .expect("incomplete tail should be recoverable");
    assert_eq!(reopened.load().await.expect("session should load").len(), 1);

    reopened
        .append(&Message::assistant(Some("recovered".to_string()), None))
        .await
        .expect("new message should append after recovery");
    assert_eq!(
        reopened.load().await.expect("session should reload").len(),
        2
    );
}

#[tokio::test]
async fn repairs_a_first_incomplete_record_before_append() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("first-incomplete.jsonl");
    fs::write(&path, b"{\"session_id\":")
        .await
        .expect("partial record should be written");

    let session_id = Uuid::new_v4();
    let store = SessionStore::open(&path, session_id)
        .await
        .expect("incomplete first record should be recoverable");
    assert!(store.load().await.expect("session should load").is_empty());

    let record = store
        .append(&Message::user("after repair"))
        .await
        .expect("append should succeed after repair");
    assert_eq!(record.sequence, 0);
    assert_eq!(store.load().await.expect("session should reload").len(), 1);
}

#[tokio::test]
async fn appends_after_a_complete_record_without_trailing_newline() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("missing-newline.jsonl");
    let session_id = Uuid::new_v4();
    let store = SessionStore::open(&path, session_id)
        .await
        .expect("session should open");
    store
        .append(&Message::user("first"))
        .await
        .expect("first message should append");

    let file = OpenOptions::new()
        .write(true)
        .open(&path)
        .await
        .expect("session file should open");
    let length = file
        .metadata()
        .await
        .expect("session metadata should be readable")
        .len();
    file.set_len(length - 1)
        .await
        .expect("trailing newline should be removable");
    file.sync_data()
        .await
        .expect("session truncation should be durable");

    let reopened = SessionStore::open(&path, session_id)
        .await
        .expect("complete JSON without a newline should load");
    let second = reopened
        .append(&Message::assistant(Some("second".to_string()), None))
        .await
        .expect("append should restore the JSONL separator");
    let records = reopened.load().await.expect("session should reload");

    assert_eq!(second.sequence, 1);
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].message.content.as_deref(), Some("first"));
    assert_eq!(records[1].message.content.as_deref(), Some("second"));
}

#[tokio::test]
async fn repairs_all_tail_tool_calls_once_in_source_order() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("orphan.jsonl");
    let session_id = Uuid::new_v4();
    let store = SessionStore::open(&path, session_id)
        .await
        .expect("session should open");
    store
        .append(&Message::user("read notes"))
        .await
        .expect("user message should append");
    store
        .append(&Message::assistant(
            None,
            Some(vec![tool_call("call-b"), tool_call("call-a")]),
        ))
        .await
        .expect("assistant call should append");

    let repaired = store
        .load_repaired()
        .await
        .expect("tail calls should be repaired");
    assert_eq!(repaired.len(), 4);
    assert_eq!(repaired[2].message.tool_call_id.as_deref(), Some("call-b"));
    assert_eq!(repaired[3].message.tool_call_id.as_deref(), Some("call-a"));
    assert!(repaired[2]
        .message
        .content
        .as_deref()
        .is_some_and(|content| content.contains("outcome is unknown")));
    for record in &repaired[2..] {
        let result = record
            .message
            .tool_result
            .as_ref()
            .expect("recovered tool result should have metadata");
        assert_eq!(result.details, serde_json::json!({"status": "unknown"}));
        assert!(result.is_error);
        assert!(!result.terminate);
    }

    let second_load = store
        .load_repaired()
        .await
        .expect("repaired transcript should remain stable");
    assert_eq!(second_load.len(), 4);
}

#[tokio::test]
async fn repairs_only_missing_tail_call_after_partial_execution() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("partial-orphan.jsonl");
    let session_id = Uuid::new_v4();
    let store = SessionStore::open(&path, session_id)
        .await
        .expect("session should open");
    store
        .append(&Message::assistant(
            None,
            Some(vec![tool_call("call-1"), tool_call("call-2")]),
        ))
        .await
        .expect("assistant call should append");
    store
        .append(&Message::tool("call-1", "done"))
        .await
        .expect("completed tool should append");

    let repaired = store
        .load_repaired()
        .await
        .expect("missing tail call should be repaired");
    assert_eq!(repaired.len(), 3);
    assert_eq!(repaired[2].message.tool_call_id.as_deref(), Some("call-2"));
    let result = repaired[2]
        .message
        .tool_result
        .as_ref()
        .expect("recovered tool result should have metadata");
    assert_eq!(result.details, serde_json::json!({"status": "unknown"}));
    assert!(result.is_error);
    assert!(!result.terminate);
}

#[tokio::test]
async fn rejects_an_orphan_that_is_not_at_the_transcript_tail() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("middle-orphan.jsonl");
    let session_id = Uuid::new_v4();
    let store = SessionStore::open(&path, session_id)
        .await
        .expect("session should open");
    store
        .append(&Message::assistant(None, Some(vec![tool_call("call-1")])))
        .await
        .expect("assistant call should append");
    store
        .append(&Message::user("continue"))
        .await
        .expect("following user message should append");

    let error = store
        .load_repaired()
        .await
        .expect_err("middle orphan must not be appended out of order");
    assert!(matches!(
        error,
        SessionError::OrphanToolCall { call_ids } if call_ids == vec!["call-1"]
    ));
}

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
