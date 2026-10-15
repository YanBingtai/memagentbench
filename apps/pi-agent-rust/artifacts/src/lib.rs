//! Content-addressed storage for media and structured artifacts.
//!
//! The store keeps payload bytes outside memory records. A successful `put`
//! returns the shared [`pi_agent_memory::ArtifactRef`], which is the only
//! value that memory methods need to retain.

use pi_agent_memory::ArtifactRef;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use uuid::Uuid;

mod validated_memory;

pub use validated_memory::ArtifactValidatedMemoryMethod;

#[cfg(unix)]
use std::os::fd::AsRawFd;

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

/// A request to persist an artifact read incrementally from a source.
///
/// The reader is consumed exactly once. The store hashes and bounds the
/// stream while copying it to a temporary file, then atomically publishes the
/// payload and metadata after the complete stream has been read.
pub struct PutArtifactStreamRequest {
    pub namespace: String,
    pub idempotency_key: Option<String>,
    pub media_type: String,
    pub reader: Box<dyn Read + Send>,
    pub codec: Option<String>,
    pub schema: Option<String>,
    pub stream_key: Option<String>,
}

impl PutArtifactStreamRequest {
    /// Build a stream request from any owned, sendable reader.
    pub fn from_reader<R>(
        namespace: impl Into<String>,
        idempotency_key: Option<String>,
        media_type: impl Into<String>,
        reader: R,
    ) -> Self
    where
        R: Read + Send + 'static,
    {
        Self {
            namespace: namespace.into(),
            idempotency_key,
            media_type: media_type.into(),
            reader: Box::new(reader),
            codec: None,
            schema: None,
            stream_key: None,
        }
    }
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
    #[error("artifact store lock is already held: {0}")]
    LockHeld(PathBuf),
}

/// The storage boundary used by memory methods and benchmark runners.
pub trait ArtifactStore: Send + Sync {
    /// Persist bytes and return their content-addressed reference.
    fn put(&self, request: PutArtifactRequest) -> Result<ArtifactRef, ArtifactStoreError>;

    /// Persist bytes read incrementally from a source and return their
    /// content-addressed reference.
    fn put_stream(
        &self,
        request: PutArtifactStreamRequest,
    ) -> Result<ArtifactRef, ArtifactStoreError>;

    /// Read canonical metadata for an artifact reference.
    fn stat(&self, artifact: &ArtifactRef) -> Result<ArtifactRef, ArtifactStoreError>;

    /// Validate the identity fields carried by an inline memory blob.
    ///
    /// Implementations must read canonical metadata (and, for local stores,
    /// verify the payload hash) before returning success.  The optional
    /// provenance fields on [`ArtifactRef`] are intentionally not part of a
    /// `ContentRef::Blob`, so this helper compares the four fields that are
    /// available at that boundary.
    fn validate_blob(
        &self,
        uri: &str,
        media_type: &str,
        sha256: &str,
        size_bytes: u64,
    ) -> Result<ArtifactRef, ArtifactStoreError> {
        let artifact = ArtifactRef {
            artifact_id: Uuid::nil(),
            uri: uri.to_string(),
            media_type: media_type.to_string(),
            sha256: sha256.to_string(),
            bytes: size_bytes,
            codec: None,
            schema: None,
            stream_key: None,
            storage_revision: None,
        };
        let canonical = self.stat(&artifact)?;
        if canonical.uri != uri
            || canonical.media_type != media_type
            || canonical.sha256 != sha256
            || canonical.bytes != size_bytes
        {
            return Err(ArtifactStoreError::InvalidRequest(
                "artifact reference does not match canonical metadata".to_string(),
            ));
        }
        Ok(canonical)
    }

    /// Read a half-open byte range `[start, end)` with a configured upper
    /// bound. The range must be within the artifact.
    fn read_range(
        &self,
        artifact: &ArtifactRef,
        range: Range<u64>,
    ) -> Result<ArtifactChunk, ArtifactStoreError>;
}

impl<T> ArtifactStore for Arc<T>
where
    T: ArtifactStore + ?Sized,
{
    fn put(&self, request: PutArtifactRequest) -> Result<ArtifactRef, ArtifactStoreError> {
        self.as_ref().put(request)
    }

    fn put_stream(
        &self,
        request: PutArtifactStreamRequest,
    ) -> Result<ArtifactRef, ArtifactStoreError> {
        self.as_ref().put_stream(request)
    }

    fn stat(&self, artifact: &ArtifactRef) -> Result<ArtifactRef, ArtifactStoreError> {
        self.as_ref().stat(artifact)
    }

    fn validate_blob(
        &self,
        uri: &str,
        media_type: &str,
        sha256: &str,
        size_bytes: u64,
    ) -> Result<ArtifactRef, ArtifactStoreError> {
        self.as_ref()
            .validate_blob(uri, media_type, sha256, size_bytes)
    }

    fn read_range(
        &self,
        artifact: &ArtifactRef,
        range: Range<u64>,
    ) -> Result<ArtifactChunk, ArtifactStoreError> {
        self.as_ref().read_range(artifact, range)
    }
}

/// Filesystem implementation using SHA-256 content addressing.
///
/// Each namespace has an `objects/` tree and an idempotency index. Payload and
/// metadata writes use a temporary file followed by `rename`, so readers never
/// observe a partially written file. A root-level file lock also coordinates
/// writers created from separate store instances or processes.
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
        let PutArtifactRequest {
            namespace,
            idempotency_key,
            media_type,
            data,
            codec,
            schema,
            stream_key,
        } = request;
        self.put_stream_sync(PutArtifactStreamRequest {
            namespace,
            idempotency_key,
            media_type,
            reader: Box::new(Cursor::new(data)),
            codec,
            schema,
            stream_key,
        })
    }

    fn put_stream_sync(
        &self,
        request: PutArtifactStreamRequest,
    ) -> Result<ArtifactRef, ArtifactStoreError> {
        let PutArtifactStreamRequest {
            namespace,
            idempotency_key,
            media_type,
            mut reader,
            codec,
            schema,
            stream_key,
        } = request;
        validate_namespace(&namespace)?;
        validate_idempotency_key(idempotency_key.as_deref())?;
        validate_media_type(&media_type)?;

        let temp_path = self.stream_temp_path(&namespace)?;
        let stream_result = (|| -> Result<(u64, String), ArtifactStoreError> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)?;
            let mut hasher = Sha256::new();
            let mut total = 0_u64;
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                let read = reader.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                let read = u64::try_from(read).map_err(|_| {
                    ArtifactStoreError::InvalidRequest("payload length overflows u64".to_string())
                })?;
                total = total.checked_add(read).ok_or_else(|| {
                    ArtifactStoreError::InvalidRequest("payload length overflows u64".to_string())
                })?;
                if total > self.max_object_bytes {
                    return Err(ArtifactStoreError::InvalidRequest(format!(
                        "payload is {total} bytes, limit is {}",
                        self.max_object_bytes
                    )));
                }
                let read = usize::try_from(read).map_err(|_| {
                    ArtifactStoreError::InvalidRequest("payload chunk is too large".to_string())
                })?;
                file.write_all(&buffer[..read])?;
                hasher.update(&buffer[..read]);
            }
            file.sync_all()?;
            Ok((total, to_hex(&hasher.finalize())))
        })();
        let (byte_len, digest_hex) = match stream_result {
            Ok(value) => value,
            Err(error) => {
                let _ = fs::remove_file(&temp_path);
                return Err(error);
            }
        };

        let digest = format!("sha256:{digest_hex}");
        let fingerprint = request_fingerprint(
            &digest,
            &media_type,
            codec.as_deref(),
            schema.as_deref(),
            stream_key.as_deref(),
        );
        let namespace_dir = self.namespace_dir(&namespace);
        let idempotency_path = idempotency_key
            .as_deref()
            .map(|key| self.idempotency_path(&namespace, key));

        let result = self.commit_streamed(
            &namespace,
            idempotency_path,
            &media_type,
            codec,
            schema,
            stream_key,
            byte_len,
            &digest,
            &digest_hex,
            fingerprint,
            temp_path.clone(),
            namespace_dir,
        );
        // The commit may reuse an existing object or idempotency entry, so
        // the temporary payload can remain even on a successful result.
        // Remove it for every outcome; after a successful rename this is a
        // harmless NotFound result.
        let _ = fs::remove_file(temp_path);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_streamed(
        &self,
        namespace: &str,
        idempotency_path: Option<PathBuf>,
        media_type: &str,
        codec: Option<String>,
        schema: Option<String>,
        stream_key: Option<String>,
        byte_len: u64,
        digest: &str,
        digest_hex: &str,
        fingerprint: String,
        temp_path: PathBuf,
        namespace_dir: PathBuf,
    ) -> Result<ArtifactRef, ArtifactStoreError> {
        let _guard = self
            .write_lock
            .lock()
            .map_err(|_| ArtifactStoreError::LockPoisoned)?;
        let _file_lock = acquire_write_lock(self.root.as_path())?;

        let artifact = ArtifactRef {
            artifact_id: Uuid::new_v4(),
            uri: format!("cas://{namespace}/{digest}"),
            media_type: media_type.to_string(),
            sha256: digest.to_string(),
            bytes: byte_len,
            codec,
            schema,
            stream_key,
            storage_revision: Some("fs-v1".to_string()),
        };

        if let Some(path) = idempotency_path.as_ref() {
            if path.exists() {
                let entry: IdempotencyEntry = read_json(path)?;
                let canonical = self.stat(&entry.artifact)?;
                if entry.request_fingerprint != fingerprint
                    && !same_artifact_identity(&canonical, &artifact)
                {
                    return Err(ArtifactStoreError::Conflict(
                        "idempotency key was reused for a different request".to_string(),
                    ));
                }
                return Ok(canonical);
            }
        }
        let (data_path, metadata_path) = self.object_paths(namespace, digest_hex);
        fs::create_dir_all(data_path.parent().ok_or_else(|| {
            ArtifactStoreError::InvalidRequest("invalid store root".to_string())
        })?)?;
        fs::create_dir_all(namespace_dir.join("idempotency"))?;

        if data_path.exists() != metadata_path.exists() {
            if data_path.exists() {
                fs::remove_file(&data_path)?;
            }
            if metadata_path.exists() {
                fs::remove_file(&metadata_path)?;
            }
        }

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
                return self.write_idempotency(idempotency_path, fingerprint, existing);
            }
        } else {
            fs::rename(&temp_path, &data_path)?;
            atomic_write_json(&metadata_path, &artifact)?;
        }

        self.write_idempotency(idempotency_path, fingerprint, artifact.clone())?;
        Ok(artifact)
    }

    fn stream_temp_path(&self, namespace: &str) -> Result<PathBuf, ArtifactStoreError> {
        let directory = self.namespace_dir(namespace).join(".tmp");
        fs::create_dir_all(&directory)?;
        Ok(directory.join(format!("artifact-{}.tmp", Uuid::new_v4())))
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
        if stored.sha256 != format!("sha256:{}", digest)
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
        if stored.sha256 != format!("sha256:{}", digest_bytes_file(&data_path)?) {
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

    fn put_stream(
        &self,
        request: PutArtifactStreamRequest,
    ) -> Result<ArtifactRef, ArtifactStoreError> {
        self.put_stream_sync(request)
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
    let digest_hex = digest.strip_prefix("sha256:").ok_or_else(|| {
        ArtifactStoreError::InvalidRequest(
            "artifact URI digest must use the sha256:<hex> format".to_string(),
        )
    })?;
    if digest_hex.len() != 64 || !digest_hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ArtifactStoreError::InvalidRequest(
            "artifact URI contains an invalid SHA-256 digest".to_string(),
        ));
    }
    let digest = digest_hex.to_ascii_lowercase();
    if artifact.sha256 != format!("sha256:{}", digest) {
        return Err(ArtifactStoreError::InvalidRequest(
            "artifact URI and sha256 field disagree".to_string(),
        ));
    }
    Ok((namespace.to_string(), digest))
}

fn request_fingerprint(
    digest: &str,
    media_type: &str,
    codec: Option<&str>,
    schema: Option<&str>,
    stream_key: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    update_fingerprint_text(&mut hasher, digest);
    update_fingerprint_text(&mut hasher, media_type);
    for value in [codec, schema, stream_key] {
        update_fingerprint_option(&mut hasher, value);
    }
    to_hex(&hasher.finalize())
}

fn same_artifact_identity(left: &ArtifactRef, right: &ArtifactRef) -> bool {
    left.uri == right.uri
        && left.media_type == right.media_type
        && left.sha256 == right.sha256
        && left.bytes == right.bytes
        && left.codec == right.codec
        && left.schema == right.schema
        && left.stream_key == right.stream_key
        && left.storage_revision == right.storage_revision
}

fn update_fingerprint_option(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            update_fingerprint_text(hasher, value);
        }
        None => hasher.update([0]),
    }
}

fn update_fingerprint_text(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
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

struct WriteLock {
    #[allow(dead_code)]
    path: PathBuf,
    file: File,
}

impl Drop for WriteLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            // The kernel releases this lock when the process exits. Keep the
            // path so a crashed writer cannot leave a permanently stale lock.
            let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        }
        #[cfg(not(unix))]
        {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn acquire_write_lock(root: &Path) -> Result<WriteLock, ArtifactStoreError> {
    let path = root.join(".write.lock");
    let file = OpenOptions::new()
        .write(true)
        .read(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(ArtifactStoreError::Io)?;

    #[cfg(unix)]
    {
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result != 0 {
            let source = io::Error::last_os_error();
            if source
                .raw_os_error()
                .is_some_and(|code| code == libc::EWOULDBLOCK || code == libc::EAGAIN)
            {
                return Err(ArtifactStoreError::LockHeld(path));
            }
            return Err(ArtifactStoreError::Io(source));
        }
    }

    #[cfg(not(unix))]
    {
        if file.metadata().map_err(ArtifactStoreError::Io)?.len() != 0 {
            return Err(ArtifactStoreError::LockHeld(path));
        }
        writeln!(&file, "pid={}", std::process::id()).map_err(ArtifactStoreError::Io)?;
    }

    Ok(WriteLock { path, file })
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
        assert_eq!(
            artifact.sha256,
            format!("sha256:{}", digest_bytes(b"hello world"))
        );
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
    fn idempotency_retries_remain_compatible_with_legacy_fingerprints() {
        let directory = tempdir().unwrap();
        let store = FsArtifactStore::new(directory.path()).unwrap();
        let original = request("run", Some("same"), b"one");
        let first = store.put(original.clone()).unwrap();
        let idempotency_path = store.idempotency_path("run", "same");
        let mut entry: IdempotencyEntry = read_json(&idempotency_path).unwrap();
        entry.request_fingerprint = legacy_request_fingerprint(
            &first.sha256,
            &original.media_type,
            original.codec.as_deref(),
            original.schema.as_deref(),
            original.stream_key.as_deref(),
        );
        atomic_write_json(&idempotency_path, &entry).unwrap();

        assert_eq!(store.put(original).unwrap(), first);
    }

    #[test]
    fn idempotency_fingerprint_distinguishes_nul_and_field_boundaries() {
        let directory = tempdir().unwrap();
        let store = FsArtifactStore::new(directory.path()).unwrap();

        let mut first = request("run", Some("same"), b"one");
        first.schema = Some("a".to_string());
        first.stream_key = Some("\0".to_string());
        store.put(first).unwrap();

        let mut different_metadata = request("run", Some("same"), b"one");
        different_metadata.schema = Some("a\0".to_string());
        different_metadata.stream_key = None;
        let error = store.put(different_metadata).unwrap_err();
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

    #[test]
    fn retries_recover_an_orphan_payload() {
        let directory = tempdir().unwrap();
        let store = FsArtifactStore::new(directory.path()).unwrap();
        let request = request("run", None, b"orphan");
        let digest = digest_bytes(&request.data);
        let (data_path, metadata_path) = store.object_paths(&request.namespace, &digest);
        fs::create_dir_all(data_path.parent().unwrap()).unwrap();
        fs::write(&data_path, &request.data).unwrap();

        let artifact = store.put(request).unwrap();
        assert_eq!(store.stat(&artifact).unwrap().bytes, 6);
        assert!(metadata_path.is_file());
    }

    #[test]
    fn write_lock_rejects_another_store_instance() {
        let directory = tempdir().unwrap();
        let store = FsArtifactStore::new(directory.path()).unwrap();
        let lock = acquire_write_lock(directory.path()).unwrap();
        let error = store.put(request("run", None, b"locked")).unwrap_err();
        assert!(matches!(error, ArtifactStoreError::LockHeld(_)));
        drop(lock);
        assert!(store.put(request("run", None, b"unlocked")).is_ok());
    }

    fn legacy_request_fingerprint(
        digest: &str,
        media_type: &str,
        codec: Option<&str>,
        schema: Option<&str>,
        stream_key: Option<&str>,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(digest.as_bytes());
        hasher.update([0]);
        hasher.update(media_type.as_bytes());
        for value in [codec, schema, stream_key] {
            hasher.update([0]);
            if let Some(value) = value {
                hasher.update(value.as_bytes());
            }
        }
        to_hex(&hasher.finalize())
    }
}
