//! Content-addressed storage for media and structured artifacts.
//!
//! The store keeps payload bytes outside memory records. A successful `put`
//! returns the shared [`pi_agent_memory::ArtifactRef`], which is the only
//! value that memory methods need to retain.

use pi_agent_memory::ArtifactRef;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use uuid::Uuid;

const DEFAULT_MAX_OBJECT_BYTES: u64 = 1 << 30;
const DEFAULT_MAX_READ_BYTES: u64 = 64 << 20;

/// A request to persist one complete artifact.
#[derive(Debug, Clone)]
pub struct PutArtifactRequest {
    /// Logical isolation boundary. It becomes the namespace path in the
    /// `cas://` URI; each path segment must be safe and non-empty.
    pub namespace: String,
    /// A caller supplied key that makes retries safe. The key is scoped to
    /// `namespace` and must identify the same payload and metadata on retry.
    pub idempotency_key: Option<String>,
    pub media_type: String,
    pub data: Vec<u8>,
    pub codec: Option<String>,
    pub schema: Option<String>,
    pub stream_key: Option<String>,
}

/// A bounded result from [`ArtifactStore::read_range`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactChunk {
    pub artifact: ArtifactRef,
    pub offset: u64,
    pub data: Vec<u8>,
    pub total_bytes: u64,
}

/// Errors returned by an artifact store.
#[derive(Debug, Error)]
pub enum ArtifactStoreError {
    #[error("invalid artifact request: {0}")]
    InvalidRequest(String),
    #[error("artifact not found")]
    NotFound,
    #[error("artifact request conflicts with existing state: {0}")]
    Conflict(String),
    #[error("artifact metadata is corrupt: {0}")]
    CorruptMetadata(String),
    #[error("artifact content hash does not match metadata")]
    HashMismatch,
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("metadata encoding error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("artifact store lock is poisoned")]
    LockPoisoned,
}

/// The storage boundary used by memory methods and benchmark runners.
pub trait ArtifactStore: Send + Sync {
    /// Persist bytes and return their content-addressed reference.
    fn put(&self, request: PutArtifactRequest) -> Result<ArtifactRef, ArtifactStoreError>;

    /// Read canonical metadata for an artifact reference.
    fn stat(&self, artifact: &ArtifactRef) -> Result<ArtifactRef, ArtifactStoreError>;

    /// Read a half-open byte range `[start, end)` with a configured upper
    /// bound. The range must be within the artifact.
    fn read_range(
        &self,
        artifact: &ArtifactRef,
        range: Range<u64>,
    ) -> Result<ArtifactChunk, ArtifactStoreError>;
}

/// Filesystem implementation using SHA-256 content addressing.
///
/// Each namespace has an `objects/` tree and an idempotency index. Payload and
/// metadata writes use a temporary file followed by `rename`, so readers never
/// observe a partially written file.
#[derive(Debug, Clone)]
pub struct FsArtifactStore {
    root: Arc<PathBuf>,
    max_object_bytes: u64,
    max_read_bytes: u64,
    write_lock: Arc<Mutex<()>>,
}

impl FsArtifactStore {
    /// Create a store rooted at `root`, using the default 1 GiB object and
    /// 64 MiB range limits.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, ArtifactStoreError> {
        Self::with_limits(root, DEFAULT_MAX_OBJECT_BYTES, DEFAULT_MAX_READ_BYTES)
    }

    /// Create a store with explicit payload and range limits.
    pub fn with_limits(
        root: impl Into<PathBuf>,
        max_object_bytes: u64,
        max_read_bytes: u64,
    ) -> Result<Self, ArtifactStoreError> {
        if max_object_bytes == 0 {
            return Err(ArtifactStoreError::InvalidRequest(
                "max_object_bytes must be greater than zero".to_string(),
            ));
        }
        if max_read_bytes == 0 {
            return Err(ArtifactStoreError::InvalidRequest(
                "max_read_bytes must be greater than zero".to_string(),
            ));
        }
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self {
            root: Arc::new(root),
            max_object_bytes,
            max_read_bytes,
            write_lock: Arc::new(Mutex::new(())),
        })
    }

    pub fn root(&self) -> &Path {
        self.root.as_path()
    }

    fn put_sync(&self, request: PutArtifactRequest) -> Result<ArtifactRef, ArtifactStoreError> {
        validate_namespace(&request.namespace)?;
        validate_idempotency_key(request.idempotency_key.as_deref())?;
        validate_media_type(&request.media_type)?;
        let byte_len = u64::try_from(request.data.len()).map_err(|_| {
            ArtifactStoreError::InvalidRequest("payload length overflows u64".to_string())
        })?;
        if byte_len > self.max_object_bytes {
            return Err(ArtifactStoreError::InvalidRequest(format!(
                "payload is {byte_len} bytes, limit is {}",
                self.max_object_bytes
            )));
        }

        let digest = digest_bytes(&request.data);
        let fingerprint = request_fingerprint(&request, &digest);
        let namespace_dir = self.namespace_dir(&request.namespace);
        let idempotency_path = request
            .idempotency_key
            .as_deref()
            .map(|key| self.idempotency_path(&request.namespace, key));

        let _guard = self
            .write_lock
            .lock()
            .map_err(|_| ArtifactStoreError::LockPoisoned)?;

        if let Some(path) = idempotency_path.as_ref() {
            if path.exists() {
                let entry: IdempotencyEntry = read_json(path)?;
                if entry.request_fingerprint != fingerprint {
                    return Err(ArtifactStoreError::Conflict(
                        "idempotency key was reused for a different request".to_string(),
                    ));
                }
                return self.stat(&entry.artifact);
            }
        }

        let artifact = ArtifactRef {
            artifact_id: Uuid::new_v4(),
            uri: format!("cas://{}/{}", request.namespace, digest),
            media_type: request.media_type,
            sha256: digest.clone(),
            bytes: byte_len,
            codec: request.codec,
            schema: request.schema,
            stream_key: request.stream_key,
            storage_revision: Some("fs-v1".to_string()),
        };
        let (data_path, metadata_path) = self.object_paths(&request.namespace, &digest);
        fs::create_dir_all(data_path.parent().ok_or_else(|| {
            ArtifactStoreError::InvalidRequest("invalid store root".to_string())
        })?)?;
        fs::create_dir_all(namespace_dir.join("idempotency"))?;

        if data_path.exists() || metadata_path.exists() {
            if !data_path.is_file() || !metadata_path.is_file() {
                return Err(ArtifactStoreError::CorruptMetadata(
                    "payload and metadata must exist together".to_string(),
                ));
            }
            let existing: ArtifactRef = read_json(&metadata_path)?;
            if existing.sha256 != artifact.sha256
                || existing.bytes != artifact.bytes
                || existing.media_type != artifact.media_type
                || existing.codec != artifact.codec
                || existing.schema != artifact.schema
                || existing.stream_key != artifact.stream_key
            {
                return Err(ArtifactStoreError::Conflict(
                    "content address already has different metadata".to_string(),
                ));
            }
            self.stat_sync(&existing)?;
            if existing != artifact {
                // The UUID is metadata, not identity. A retry without an
                // idempotency key reuses the canonical existing reference.
                return self.write_idempotency(idempotency_path.clone(), fingerprint, existing);
            }
        } else {
            atomic_write(&data_path, &request.data)?;
            atomic_write_json(&metadata_path, &artifact)?;
        }

        self.write_idempotency(idempotency_path, fingerprint, artifact.clone())?;
        Ok(artifact)
    }

    fn write_idempotency(
        &self,
        path: Option<PathBuf>,
        fingerprint: String,
        artifact: ArtifactRef,
    ) -> Result<ArtifactRef, ArtifactStoreError> {
        if let Some(path) = path {
            let entry = IdempotencyEntry {
                request_fingerprint: fingerprint,
                artifact: artifact.clone(),
            };
            if !path.exists() {
                atomic_write_json(&path, &entry)?;
            }
        }
        Ok(artifact)
    }

    fn stat_sync(&self, artifact: &ArtifactRef) -> Result<ArtifactRef, ArtifactStoreError> {
        let (namespace, digest) = parse_uri(artifact)?;
        let (data_path, metadata_path) = self.object_paths(&namespace, &digest);
        if !data_path.is_file() || !metadata_path.is_file() {
            return Err(ArtifactStoreError::NotFound);
        }
        let stored: ArtifactRef = read_json(&metadata_path)?;
        if stored.sha256 != digest
            || stored.uri != artifact.uri
            || stored.bytes == 0 && fs::metadata(&data_path)?.len() != 0
        {
            return Err(ArtifactStoreError::CorruptMetadata(
                "stored reference does not match its content address".to_string(),
            ));
        }
        let actual_bytes = fs::metadata(&data_path)?.len();
        if stored.bytes != actual_bytes {
            return Err(ArtifactStoreError::CorruptMetadata(
                "stored byte count does not match payload".to_string(),
            ));
        }
        if stored.sha256 != digest_bytes_file(&data_path)? {
            return Err(ArtifactStoreError::HashMismatch);
        }
        Ok(stored)
    }

    fn read_range_sync(
        &self,
        artifact: &ArtifactRef,
        range: Range<u64>,
    ) -> Result<ArtifactChunk, ArtifactStoreError> {
        if range.start > range.end {
            return Err(ArtifactStoreError::InvalidRequest(
                "range start must not exceed end".to_string(),
            ));
        }
        let length = range.end - range.start;
        if length > self.max_read_bytes {
            return Err(ArtifactStoreError::InvalidRequest(format!(
                "range is {length} bytes, limit is {}",
                self.max_read_bytes
            )));
        }
        let stored = self.stat_sync(artifact)?;
        if range.end > stored.bytes {
            return Err(ArtifactStoreError::InvalidRequest(format!(
                "range end {} exceeds artifact size {}",
                range.end, stored.bytes
            )));
        }
        let (namespace, digest) = parse_uri(&stored)?;
        let (data_path, _) = self.object_paths(&namespace, &digest);
        let mut file = File::open(data_path)?;
        file.seek(SeekFrom::Start(range.start))?;
        let capacity = usize::try_from(length).map_err(|_| {
            ArtifactStoreError::InvalidRequest("range is too large for this platform".to_string())
        })?;
        let mut data = Vec::with_capacity(capacity);
        file.take(length).read_to_end(&mut data)?;
        Ok(ArtifactChunk {
            artifact: stored.clone(),
            offset: range.start,
            data,
            total_bytes: stored.bytes,
        })
    }

    fn namespace_dir(&self, namespace: &str) -> PathBuf {
        self.root.join(namespace)
    }

    fn object_paths(&self, namespace: &str, digest: &str) -> (PathBuf, PathBuf) {
        let dir = self
            .namespace_dir(namespace)
            .join("objects")
            .join(&digest[..2]);
        (
            dir.join(format!("{digest}.bin")),
            dir.join(format!("{digest}.json")),
        )
    }

    fn idempotency_path(&self, namespace: &str, key: &str) -> PathBuf {
        self.namespace_dir(namespace)
            .join("idempotency")
            .join(format!("{}.json", digest_bytes(key.as_bytes())))
    }
}

impl ArtifactStore for FsArtifactStore {
    fn put(&self, request: PutArtifactRequest) -> Result<ArtifactRef, ArtifactStoreError> {
        self.put_sync(request)
    }

    fn stat(&self, artifact: &ArtifactRef) -> Result<ArtifactRef, ArtifactStoreError> {
        self.stat_sync(artifact)
    }

    fn read_range(
        &self,
        artifact: &ArtifactRef,
        range: Range<u64>,
    ) -> Result<ArtifactChunk, ArtifactStoreError> {
        self.read_range_sync(artifact, range)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct IdempotencyEntry {
    request_fingerprint: String,
    artifact: ArtifactRef,
}

fn validate_namespace(namespace: &str) -> Result<(), ArtifactStoreError> {
    let valid = !namespace.is_empty()
        && namespace.len() <= 128
        && namespace.split('/').all(|segment| {
            !segment.is_empty()
                && segment != "."
                && segment != ".."
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        });
    if !valid {
        return Err(ArtifactStoreError::InvalidRequest(
            "namespace must be 1-128 safe ASCII path segments".to_string(),
        ));
    }
    Ok(())
}

fn validate_idempotency_key(key: Option<&str>) -> Result<(), ArtifactStoreError> {
    if let Some(key) = key {
        if key.is_empty() || key.len() > 256 || key.bytes().any(|byte| byte.is_ascii_control()) {
            return Err(ArtifactStoreError::InvalidRequest(
                "idempotency key must be 1-256 non-control characters".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_media_type(media_type: &str) -> Result<(), ArtifactStoreError> {
    if media_type.is_empty() || media_type.len() > 256 || !media_type.contains('/') {
        return Err(ArtifactStoreError::InvalidRequest(
            "media_type must be a non-empty MIME-like value".to_string(),
        ));
    }
    Ok(())
}

fn parse_uri(artifact: &ArtifactRef) -> Result<(String, String), ArtifactStoreError> {
    let value = artifact.uri.strip_prefix("cas://").ok_or_else(|| {
        ArtifactStoreError::InvalidRequest("artifact URI must use cas://".to_string())
    })?;
    let (namespace, digest) = value.rsplit_once('/').ok_or_else(|| {
        ArtifactStoreError::InvalidRequest("artifact URI is missing digest".to_string())
    })?;
    validate_namespace(namespace)?;
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ArtifactStoreError::InvalidRequest(
            "artifact URI contains an invalid SHA-256 digest".to_string(),
        ));
    }
    let digest = digest.to_ascii_lowercase();
    if artifact.sha256 != digest {
        return Err(ArtifactStoreError::InvalidRequest(
            "artifact URI and sha256 field disagree".to_string(),
        ));
    }
    Ok((namespace.to_string(), digest))
}

fn request_fingerprint(request: &PutArtifactRequest, digest: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(digest.as_bytes());
    hasher.update([0]);
    hasher.update(request.media_type.as_bytes());
    for value in [&request.codec, &request.schema, &request.stream_key] {
        hasher.update([0]);
        if let Some(value) = value {
            hasher.update(value.as_bytes());
        }
    }
    to_hex(&hasher.finalize())
}

fn digest_bytes(data: &[u8]) -> String {
    to_hex(&Sha256::digest(data))
}

fn digest_bytes_file(path: &Path) -> Result<String, ArtifactStoreError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let bytes = file.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    Ok(to_hex(&hasher.finalize()))
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, ArtifactStoreError> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| ArtifactStoreError::CorruptMetadata(error.to_string()))
}

fn atomic_write(path: &Path, data: &[u8]) -> Result<(), ArtifactStoreError> {
    let parent = path
        .parent()
        .ok_or_else(|| ArtifactStoreError::InvalidRequest("path has no parent".to_string()))?;
    fs::create_dir_all(parent)?;
    let temp = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name().unwrap().to_string_lossy(),
        Uuid::new_v4()
    ));
    let result = (|| -> Result<(), ArtifactStoreError> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(data)?;
        file.sync_all()?;
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), ArtifactStoreError> {
    atomic_write(path, &serde_json::to_vec_pretty(value)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn request(namespace: &str, key: Option<&str>, data: &[u8]) -> PutArtifactRequest {
        PutArtifactRequest {
            namespace: namespace.to_string(),
            idempotency_key: key.map(str::to_string),
            media_type: "text/plain".to_string(),
            data: data.to_vec(),
            codec: None,
            schema: Some("plain-v1".to_string()),
            stream_key: None,
        }
    }

    #[test]
    fn put_stat_and_read_range_round_trip() {
        let directory = tempdir().unwrap();
        let store = FsArtifactStore::with_limits(directory.path(), 1024, 8).unwrap();
        let artifact = store
            .put(request("benchmark/text", Some("step-1"), b"hello world"))
            .unwrap();
        assert_eq!(artifact.sha256, digest_bytes(b"hello world"));
        assert_eq!(
            artifact.uri,
            format!("cas://benchmark/text/{}", artifact.sha256)
        );
        assert_eq!(store.stat(&artifact).unwrap(), artifact);
        let chunk = store.read_range(&artifact, 2..6).unwrap();
        assert_eq!(chunk.data, b"llo ");
        assert_eq!(chunk.offset, 2);
        assert_eq!(chunk.total_bytes, 11);
    }

    #[test]
    fn idempotency_reuses_reference_and_rejects_changed_payload() {
        let directory = tempdir().unwrap();
        let store = FsArtifactStore::new(directory.path()).unwrap();
        let first = store.put(request("run", Some("same"), b"one")).unwrap();
        let retry = store.put(request("run", Some("same"), b"one")).unwrap();
        assert_eq!(first, retry);
        let error = store.put(request("run", Some("same"), b"two")).unwrap_err();
        assert!(matches!(error, ArtifactStoreError::Conflict(_)));
    }

    #[test]
    fn idempotency_is_scoped_to_namespace() {
        let directory = tempdir().unwrap();
        let store = FsArtifactStore::new(directory.path()).unwrap();
        let first = store.put(request("a", Some("same"), b"one")).unwrap();
        let second = store.put(request("b", Some("same"), b"two")).unwrap();
        assert_ne!(first.uri, second.uri);
    }

    #[test]
    fn range_is_bounded_and_must_fit_artifact() {
        let directory = tempdir().unwrap();
        let store = FsArtifactStore::with_limits(directory.path(), 1024, 3).unwrap();
        let artifact = store.put(request("run", None, b"hello")).unwrap();
        assert!(matches!(
            store.read_range(&artifact, 0..4),
            Err(ArtifactStoreError::InvalidRequest(_))
        ));
        assert!(matches!(
            store.read_range(&artifact, 4..6),
            Err(ArtifactStoreError::InvalidRequest(_))
        ));
        assert!(store.read_range(&artifact, 3..3).is_ok());
    }

    #[test]
    fn rejects_unsafe_namespace() {
        let directory = tempdir().unwrap();
        let store = FsArtifactStore::new(directory.path()).unwrap();
        let error = store.put(request("../escape", None, b"x")).unwrap_err();
        assert!(matches!(error, ArtifactStoreError::InvalidRequest(_)));
    }

    #[test]
    fn restart_keeps_content_and_metadata() {
        let directory = tempdir().unwrap();
        let artifact = {
            let store = FsArtifactStore::new(directory.path()).unwrap();
            store
                .put(request("run", Some("step"), b"persisted"))
                .unwrap()
        };
        let store = FsArtifactStore::new(directory.path()).unwrap();
        assert_eq!(store.stat(&artifact).unwrap(), artifact);
        assert_eq!(
            store.read_range(&artifact, 0..9).unwrap().data,
            b"persisted"
        );
    }
}
