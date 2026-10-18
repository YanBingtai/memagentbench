//! Public contract tests for artifact-store implementations.
//!
//! The assertions use only `ArtifactStore`, so another backend can reuse the
//! same observable storage contract without depending on filesystem helpers.

use std::io::{self, Read};
use std::{env, fs, path::PathBuf};

use pi_agent_artifacts::{
    ArtifactStore, ArtifactStoreError, FsArtifactStore, PutArtifactRequest,
    PutArtifactStreamRequest,
};

fn path(name: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "pi-agent-artifact-contract-{name}-{}",
        uuid::Uuid::new_v4()
    ))
}

fn request(namespace: &str, key: &str, data: &[u8]) -> PutArtifactRequest {
    PutArtifactRequest {
        namespace: namespace.to_string(),
        idempotency_key: Some(key.to_string()),
        media_type: "text/plain".to_string(),
        data: data.to_vec(),
        codec: None,
        schema: Some("plain-v1".to_string()),
        stream_key: None,
    }
}

fn exercise_store<S: ArtifactStore>(store: &S) {
    let first = store
        .put(request("contract/run-1", "step-1", b"hello artifact"))
        .expect("artifact should be stored");
    assert_eq!(first.bytes, 14);
    assert!(first.uri.starts_with("cas://contract/run-1/sha256:"));
    assert_eq!(store.stat(&first).expect("artifact should stat"), first);

    let retry = store
        .put(request("contract/run-1", "step-1", b"hello artifact"))
        .expect("same request should be idempotent");
    assert_eq!(retry, first);

    let chunk = store
        .read_range(&first, 6..14)
        .expect("range should be readable");
    assert_eq!(chunk.data, b"artifact");
    assert_eq!(chunk.offset, 6);
    assert_eq!(chunk.total_bytes, 14);

    let error = store
        .put(request("contract/run-1", "step-1", b"changed"))
        .expect_err("idempotency key reuse must be rejected");
    assert!(matches!(error, ArtifactStoreError::Conflict(_)));
}

#[test]
fn filesystem_store_satisfies_public_artifact_contract() {
    let root = path("fs");
    let store = FsArtifactStore::new(&root).expect("store should open");
    exercise_store(&store);
    drop(store);

    let reopened = FsArtifactStore::new(&root).expect("store should reopen");
    let artifact = reopened
        .put(request("contract/run-1", "step-2", b"after restart"))
        .expect("reopened store should accept writes");
    assert_eq!(reopened.read_range(&artifact, 0..5).unwrap().data, b"after");
    drop(reopened);
    let _ = fs::remove_dir_all(root);
}

struct ChunkedReader {
    data: Vec<u8>,
    offset: usize,
    chunk_size: usize,
}

impl Read for ChunkedReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.offset == self.data.len() {
            return Ok(0);
        }
        let length = self
            .chunk_size
            .min(buffer.len())
            .min(self.data.len() - self.offset);
        buffer[..length].copy_from_slice(&self.data[self.offset..self.offset + length]);
        self.offset += length;
        Ok(length)
    }
}

struct FailingReader {
    emitted: bool,
}

impl Read for FailingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if !self.emitted {
            self.emitted = true;
            buffer[..7].copy_from_slice(b"partial");
            return Ok(7);
        }
        Err(io::Error::other("test reader failed"))
    }
}

#[test]
fn stream_put_matches_vec_hash_stat_and_range_contract() {
    let root = path("stream");
    let store = FsArtifactStore::new(&root).expect("store should open");
    let data = b"streamed artifact payload".to_vec();
    let expected = store
        .put(request("contract/stream", "vec", &data))
        .expect("vec payload should be stored");

    let mut stream = PutArtifactStreamRequest::from_reader(
        "contract/stream",
        Some("stream".to_string()),
        "text/plain",
        ChunkedReader {
            data: data.clone(),
            offset: 0,
            chunk_size: 3,
        },
    );
    stream.schema = Some("plain-v1".to_string());
    let actual = store
        .put_stream(stream)
        .expect("stream payload should be stored");

    assert_eq!(actual, expected);
    assert_eq!(actual.bytes, data.len() as u64);
    assert_eq!(store.stat(&actual).expect("artifact should stat"), expected);
    assert_eq!(store.read_range(&actual, 2..11).unwrap().data, &data[2..11]);
    assert_eq!(
        fs::read_dir(root.join("contract/stream/.tmp"))
            .expect("temporary directory should be readable")
            .count(),
        0
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn stream_reader_failure_cleans_temp_file_without_metadata() {
    let root = path("stream-failure");
    let store = FsArtifactStore::new(&root).expect("store should open");
    let stream = PutArtifactStreamRequest::from_reader(
        "contract/failure",
        Some("failed".to_string()),
        "application/octet-stream",
        FailingReader { emitted: false },
    );

    assert!(matches!(
        store.put_stream(stream),
        Err(ArtifactStoreError::Io(_))
    ));
    let namespace = root.join("contract/failure");
    let temporary = namespace.join(".tmp");
    assert!(temporary.is_dir());
    assert_eq!(
        fs::read_dir(&temporary)
            .expect("temporary directory should be readable")
            .count(),
        0
    );
    assert!(!namespace.join("objects").exists());
    assert!(!namespace.join("idempotency").exists());
    let _ = fs::remove_dir_all(root);
}
