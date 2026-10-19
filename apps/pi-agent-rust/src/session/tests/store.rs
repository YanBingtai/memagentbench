use super::*;

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
