use super::*;

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
