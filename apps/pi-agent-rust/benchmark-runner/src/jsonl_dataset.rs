//! Streaming JSONL dataset sources.
//!
//! A JSONL file is treated as an ordered sequence of independent records. The
//! source keeps only one physical line in memory, so large datasets can be
//! replayed without loading the complete file before a run starts.

use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    marker::PhantomData,
    path::{Path, PathBuf},
};

use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

use super::{
    ContinualDatasetSample, ContinualDatasetSource, DatasetError, DatasetSample, DatasetSource,
    OfflineDatasetSample, ReplayableOfflineDatasetSource,
};

/// Maximum raw bytes in one physical line, including its line ending.
pub const DEFAULT_MAX_JSONL_LINE_BYTES: usize = 16 * 1024 * 1024;

/// A streaming JSONL source for any owned serde-deserializable value.
pub struct JsonlSource<T> {
    path: PathBuf,
    reader: BufReader<File>,
    line_number: usize,
    max_line_bytes: usize,
    expected_digest: String,
    observed_digest: Sha256,
    marker: PhantomData<fn() -> T>,
}

impl<T> std::fmt::Debug for JsonlSource<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JsonlSource")
            .field("path", &self.path)
            .field("line_number", &self.line_number)
            .finish_non_exhaustive()
    }
}

impl<T> JsonlSource<T> {
    /// Opens and hashes a JSONL file without loading its records into memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DatasetError> {
        Self::open_with_limits(path, DEFAULT_MAX_JSONL_LINE_BYTES)
    }

    /// Opens a source with a maximum raw byte count per physical line.
    ///
    /// The initial digest and subsequent records come from the same file
    /// handle. The digest is verified when streaming reaches EOF.
    pub fn open_with_limits(
        path: impl AsRef<Path>,
        max_line_bytes: usize,
    ) -> Result<Self, DatasetError> {
        let path = path.as_ref().to_path_buf();
        if max_line_bytes == 0 {
            return Err(DatasetError::Source(format!(
                "JSONL dataset {} max_line_bytes must be greater than zero",
                path.display()
            )));
        }
        let mut file = File::open(&path).map_err(|error| {
            DatasetError::Source(format!(
                "failed to open JSONL dataset {}: {error}",
                path.display()
            ))
        })?;

        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let bytes = file.read(&mut buffer).map_err(|error| {
                DatasetError::Source(format!(
                    "failed to read JSONL dataset {} for digest: {error}",
                    path.display()
                ))
            })?;
            if bytes == 0 {
                break;
            }
            hasher.update(&buffer[..bytes]);
        }
        let expected_digest = format!("sha256:{:x}", hasher.finalize());
        file.seek(SeekFrom::Start(0)).map_err(|error| {
            DatasetError::Source(format!(
                "failed to rewind JSONL dataset {}: {error}",
                path.display()
            ))
        })?;

        Ok(Self {
            path,
            reader: BufReader::new(file),
            line_number: 0,
            max_line_bytes,
            expected_digest,
            observed_digest: Sha256::new(),
            marker: PhantomData,
        })
    }

    /// Returns the path supplied when this source was opened.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the last physical line read from the file.
    pub fn line_number(&self) -> usize {
        self.line_number
    }

    /// Returns the initial digest of the open file, without reopening its path.
    pub fn content_digest(&self) -> Result<String, DatasetError> {
        Ok(self.expected_digest.clone())
    }

    /// Rewinds this source without reopening its path.
    ///
    /// Seeking the already-open file preserves the snapshot captured by
    /// [`Self::open`], even when another process replaces the pathname while a
    /// benchmark is running. The observed digest belongs to one pass, so it
    /// must be reset together with the reader and physical line counter.
    pub fn rewind(&mut self) -> Result<(), DatasetError> {
        self.reader.seek(SeekFrom::Start(0)).map_err(|error| {
            DatasetError::Source(format!(
                "failed to rewind JSONL dataset {}: {error}",
                self.path.display()
            ))
        })?;
        self.line_number = 0;
        self.observed_digest = Sha256::new();
        Ok(())
    }

    fn read_line(&mut self) -> Result<Option<Vec<u8>>, DatasetError> {
        let line_number = self.line_number + 1;
        let mut line = Vec::new();
        loop {
            let buffer = self.reader.fill_buf().map_err(|error| {
                DatasetError::Source(format!(
                    "failed to read JSONL dataset {} at line {line_number}: {error}",
                    self.path.display()
                ))
            })?;
            if buffer.is_empty() {
                let actual_digest = format!("sha256:{:x}", self.observed_digest.clone().finalize());
                if actual_digest != self.expected_digest {
                    return Err(DatasetError::Source(format!(
                        "JSONL dataset {} changed at line {line_number}: expected {}, got {actual_digest}",
                        self.path.display(), self.expected_digest
                    )));
                }
                if line.is_empty() {
                    return Ok(None);
                }
                break;
            }
            let newline = buffer.iter().position(|byte| *byte == b'\n');
            let bytes = newline.map_or(buffer.len(), |index| index + 1);
            if bytes > self.max_line_bytes - line.len() {
                return Err(DatasetError::Source(format!(
                    "JSONL dataset {} at line {line_number} exceeds max_line_bytes {}",
                    self.path.display(),
                    self.max_line_bytes
                )));
            }
            self.observed_digest.update(&buffer[..bytes]);
            line.extend_from_slice(&buffer[..bytes]);
            self.reader.consume(bytes);
            if newline.is_some() {
                break;
            }
        }
        self.line_number = line_number;
        Ok(Some(line))
    }
}

impl<T> JsonlSource<T>
where
    T: DeserializeOwned,
{
    /// Reads the next non-empty JSONL record.
    pub fn next_value(&mut self) -> Result<Option<T>, DatasetError> {
        loop {
            let Some(bytes) = self.read_line()? else {
                return Ok(None);
            };
            let line_number = self.line_number;
            let line = std::str::from_utf8(&bytes).map_err(|error| {
                DatasetError::Source(format!(
                    "invalid UTF-8 in JSONL dataset {} at line {line_number}: {error}",
                    self.path.display()
                ))
            })?;
            if line.trim().is_empty() {
                continue;
            }

            return serde_json::from_str(line.trim())
                .map(Some)
                .map_err(|error| {
                    DatasetError::Source(format!(
                        "invalid JSON in dataset {} at line {line_number}: {error}",
                        self.path.display()
                    ))
                });
        }
    }
}

/// JSONL source for retrieval-only [`DatasetSample`] records.
pub type JsonlDataset = JsonlSource<DatasetSample>;

impl DatasetSource for JsonlDataset {
    fn dataset_digest(&self) -> Option<&str> {
        Some(&self.expected_digest)
    }

    fn next_sample(&mut self) -> Result<Option<DatasetSample>, DatasetError> {
        self.next_value()
    }
}

/// JSONL source for online ingest-and-retrieve records.
pub type JsonlContinualDataset = JsonlSource<ContinualDatasetSample>;

impl ContinualDatasetSource for JsonlContinualDataset {
    fn dataset_digest(&self) -> Option<&str> {
        Some(&self.expected_digest)
    }

    fn next_sample(&mut self) -> Result<Option<ContinualDatasetSample>, DatasetError> {
        self.next_value()
    }
}

/// JSONL source for two-phase offline build-then-query records.
pub type JsonlOfflineDataset = JsonlSource<OfflineDatasetSample>;

impl ReplayableOfflineDatasetSource for JsonlOfflineDataset {
    fn next_sample(&mut self) -> Result<Option<OfflineDatasetSample>, DatasetError> {
        self.next_value()
    }

    fn rewind(&mut self) -> Result<(), DatasetError> {
        JsonlSource::rewind(self)
    }

    fn dataset_digest(&self) -> Option<&str> {
        Some(&self.expected_digest)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write};

    use pi_agent_memory::{
        ContentRef, IngestRequest, MemoryBudget, MemoryItem, MemoryKind, MemoryProvenance,
        MemoryQuery, MemoryRecord, MemoryScope, Modality, RecordIngestRequest,
    };
    use tempfile::NamedTempFile;

    use super::*;

    fn sample(sample_id: &str) -> DatasetSample {
        DatasetSample {
            sample_id: sample_id.to_owned(),
            observation: super::super::AgentObservation { parts: Vec::new() },
            query: MemoryQuery {
                input: Some(ContentRef::Text {
                    text: format!("query-{sample_id}"),
                }),
                namespace: "dataset".to_owned(),
                scope: Some(MemoryScope::Shared),
                subject: None,
                episode_id: None,
                kinds: vec![MemoryKind::Episodic],
                modalities: vec![Modality::Text],
                time_range: None,
                visibility_cutoff_ms: None,
                graph: None,
                budget: MemoryBudget::default(),
            },
            session_id: None,
            episode_id: None,
            terminal: true,
            terminal_reason: Some(super::super::TerminalReason::Success),
        }
    }

    fn continual_sample(sample_id: &str) -> ContinualDatasetSample {
        ContinualDatasetSample {
            sample: sample(sample_id),
            ingest: IngestRequest {
                items: vec![MemoryItem {
                    id: uuid::Uuid::new_v4(),
                    version: 1,
                    kind: MemoryKind::Episodic,
                    scope: MemoryScope::Shared,
                    modality: Modality::Text,
                    content: ContentRef::Text {
                        text: format!("memory-{sample_id}"),
                    },
                    namespace: "dataset".to_owned(),
                    episode_id: None,
                    subject: None,
                    source: "fixture".to_owned(),
                    metadata: Default::default(),
                    abstraction_level: 0,
                    event_time: None,
                    ingest_time_ms: 1,
                    schema_version: 1,
                }],
                relations: Vec::new(),
                idempotency_key: format!("key-{sample_id}"),
            },
        }
    }

    fn write_jsonl<T: serde::Serialize>(lines: &[T]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("temporary dataset");
        for line in lines {
            serde_json::to_writer(&mut file, line).expect("serialize fixture");
            file.write_all(b"\n").expect("newline");
        }
        file
    }

    #[test]
    fn streams_samples_and_skips_empty_physical_lines() {
        let first = sample("first");
        let second = sample("second");
        let file = NamedTempFile::new().expect("temporary dataset");
        let path = file.path().to_path_buf();
        fs::write(
            &path,
            format!(
                "\n{}\n  \n{}\n",
                serde_json::to_string(&first).unwrap(),
                serde_json::to_string(&second).unwrap()
            ),
        )
        .unwrap();

        let mut source = JsonlDataset::open(&path).unwrap();
        assert_eq!(source.path(), path.as_path());
        assert_eq!(source.next_sample().unwrap(), Some(first));
        assert_eq!(source.line_number(), 2);
        assert_eq!(source.next_sample().unwrap(), Some(second));
        assert_eq!(source.line_number(), 4);
        assert_eq!(source.next_sample().unwrap(), None);
        assert_eq!(source.line_number(), 4);
    }

    #[test]
    fn reports_physical_line_for_invalid_json() {
        let first = sample("first");
        let file = NamedTempFile::new().expect("temporary dataset");
        let path = file.path().to_path_buf();
        fs::write(
            &path,
            format!("{}\n{{broken\n", serde_json::to_string(&first).unwrap()),
        )
        .unwrap();

        let mut source = JsonlDataset::open(&path).unwrap();
        assert_eq!(source.next_sample().unwrap(), Some(first));
        let error = source.next_sample().unwrap_err();
        let DatasetError::Source(message) = error;
        assert!(message.contains("line 2"), "{message}");
        assert!(
            message.contains(path.to_string_lossy().as_ref()),
            "{message}"
        );
    }

    #[test]
    fn replays_continual_samples() {
        let expected = continual_sample("episode-1");
        let file = write_jsonl(std::slice::from_ref(&expected));
        let mut source = JsonlContinualDataset::open(file.path()).unwrap();

        assert_eq!(source.next_sample().unwrap(), Some(expected));
        assert_eq!(source.next_sample().unwrap(), None);
    }

    #[test]
    fn rewinds_offline_samples_and_restarts_digest_pass() {
        let first = OfflineDatasetSample {
            sample: sample("offline-1"),
            build: RecordIngestRequest {
                records: vec![MemoryRecord {
                    item: MemoryItem {
                        id: uuid::Uuid::new_v4(),
                        version: 1,
                        kind: MemoryKind::Episodic,
                        scope: MemoryScope::Shared,
                        modality: Modality::Text,
                        content: ContentRef::Text {
                            text: "offline memory".to_owned(),
                        },
                        namespace: "dataset".to_owned(),
                        episode_id: None,
                        subject: None,
                        source: "fixture".to_owned(),
                        metadata: Default::default(),
                        abstraction_level: 0,
                        event_time: None,
                        ingest_time_ms: 1,
                        schema_version: 1,
                    },
                    episode: None,
                    evidence: Vec::new(),
                    provenance: MemoryProvenance {
                        producer: "fixture".to_owned(),
                        source_id: "fixture".to_owned(),
                        source_event_id: None,
                        parent_record_ids: Vec::new(),
                        method_version: None,
                    },
                }],
                relations: Vec::new(),
                idempotency_key: "unused".to_owned(),
            },
        };
        let file = write_jsonl(std::slice::from_ref(&first));
        let mut source = JsonlOfflineDataset::open(file.path()).unwrap();

        assert_eq!(source.next_sample().unwrap(), Some(first.clone()));
        assert_eq!(source.next_sample().unwrap(), None);
        source.rewind().unwrap();
        assert_eq!(source.line_number(), 0);
        assert_eq!(source.next_sample().unwrap(), Some(first));
        assert_eq!(source.next_sample().unwrap(), None);
    }

    #[test]
    fn reports_open_errors_with_path() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("missing.jsonl");
        let error = JsonlDataset::open(&path).unwrap_err();
        let DatasetError::Source(message) = error;
        assert!(
            message.contains(path.to_string_lossy().as_ref()),
            "{message}"
        );
    }

    #[test]
    fn content_digest_is_stable_without_consuming_the_source() {
        let expected = sample("digest");
        let file = write_jsonl(std::slice::from_ref(&expected));
        let mut source = JsonlDataset::open(file.path()).unwrap();
        let first_digest = source.content_digest().unwrap();
        assert_eq!(first_digest, source.content_digest().unwrap());
        assert_eq!(source.line_number(), 0);
        assert_eq!(source.next_sample().unwrap(), Some(expected));
        assert!(first_digest.starts_with("sha256:"));
    }

    #[test]
    fn handles_crlf_and_a_final_line_without_newline() {
        let file = NamedTempFile::new().unwrap();
        fs::write(file.path(), b"\r\n1\r\n \t\r\n2").unwrap();
        let mut source = JsonlSource::<u32>::open_with_limits(file.path(), 4).unwrap();

        assert_eq!(source.next_value().unwrap(), Some(1));
        assert_eq!(source.line_number(), 2);
        assert_eq!(source.next_value().unwrap(), Some(2));
        assert_eq!(source.line_number(), 4);
        assert_eq!(source.next_value().unwrap(), None);
        assert_eq!(source.next_value().unwrap(), None);
        assert_eq!(source.line_number(), 4);
    }

    #[test]
    fn rejects_a_line_over_the_limit_before_parsing() {
        let file = NamedTempFile::new().unwrap();
        // Exceed BufReader's buffer as well as the configured line limit.
        let line = format!("\n\"{}\"\n", "x".repeat(32 * 1024));
        fs::write(file.path(), line).unwrap();
        let mut source = JsonlSource::<String>::open_with_limits(file.path(), 16 * 1024).unwrap();

        let message = source.next_value().unwrap_err().to_string();
        assert!(message.contains("line 2"), "{message}");
        assert!(message.contains("max_line_bytes 16384"), "{message}");
        assert!(message.contains(file.path().to_string_lossy().as_ref()));
    }

    #[test]
    fn accepts_exact_raw_line_limit() {
        let file = NamedTempFile::new().unwrap();
        fs::write(file.path(), b"1\r\n").unwrap();
        let mut source = JsonlSource::<u32>::open_with_limits(file.path(), 3).unwrap();

        assert_eq!(source.next_value().unwrap(), Some(1));
        assert_eq!(source.next_value().unwrap(), None);
    }

    #[test]
    fn reports_invalid_utf8_at_its_physical_line() {
        let file = NamedTempFile::new().unwrap();
        fs::write(file.path(), b"\n\xff\n").unwrap();
        let mut source = JsonlSource::<u32>::open(file.path()).unwrap();

        let message = source.next_value().unwrap_err().to_string();
        assert!(message.contains("invalid UTF-8"), "{message}");
        assert!(message.contains("line 2"), "{message}");
        assert!(message.contains(file.path().to_string_lossy().as_ref()));
    }

    #[test]
    fn detects_in_place_changes_at_eof_and_keeps_reporting_them() {
        let file = NamedTempFile::new().unwrap();
        fs::write(file.path(), b"1\n").unwrap();
        let mut source = JsonlSource::<u32>::open(file.path()).unwrap();
        let expected_digest = source.content_digest().unwrap();
        fs::write(file.path(), b"2\n").unwrap();

        assert_eq!(source.content_digest().unwrap(), expected_digest);
        assert_eq!(source.next_value().unwrap(), Some(2));
        let message = source.next_value().unwrap_err().to_string();
        assert!(message.contains("changed"), "{message}");
        assert!(message.contains(&expected_digest), "{message}");
        assert!(source.next_value().is_err());
    }

    #[test]
    fn pathname_replacement_does_not_change_the_open_dataset() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dataset.jsonl");
        fs::write(&path, b"1\n").unwrap();
        let mut source = JsonlSource::<u32>::open(&path).unwrap();
        let expected_digest = source.content_digest().unwrap();
        fs::rename(&path, directory.path().join("original.jsonl")).unwrap();
        fs::write(&path, b"2\n").unwrap();

        assert_eq!(source.content_digest().unwrap(), expected_digest);
        assert_eq!(source.next_value().unwrap(), Some(1));
        assert_eq!(source.next_value().unwrap(), None);
    }
}
