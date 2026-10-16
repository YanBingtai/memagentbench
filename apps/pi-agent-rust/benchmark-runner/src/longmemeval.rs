//! LongMemEval's raw sample adapter.
//!
//! The public benchmark runner deliberately knows nothing about question
//! answers or dataset-specific labels. This module keeps those labels in a
//! benchmark-side reference object and exposes one question at a time as an
//! [`OfflineDatasetSample`]. A separate run per question matches the
//! LightMem reference script, which creates one collection for every question.

use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
};

use pi_agent_benchmark::{AgentObservation, ObservationPart, TerminalReason};
use pi_agent_memory::{
    ContentRef, EpisodeRef, MemoryBudget, MemoryItem, MemoryKind, MemoryProvenance, MemoryQuery,
    MemoryRecord, MemoryScope, Modality, RecordIngestRequest,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

use crate::{DatasetError, DatasetSample, OfflineDatasetSample, ReplayableOfflineDatasetSource};

const DATASET_SOURCE: &str = "longmemeval";
const DATASET_NAMESPACE: &str = "benchmark/longmemeval";

/// One message from the raw LongMemEval JSON file.
///
/// Unknown fields are retained because the published data uses fields such as
/// `has_answer` that are useful to the evaluator but must never be sent to the
/// memory provider as agent-visible content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LongMemEvalMessage {
    pub role: String,
    pub content: String,
    #[serde(flatten)]
    pub metadata: BTreeMap<String, Value>,
}

impl LongMemEvalMessage {
    fn has_answer(&self) -> bool {
        self.metadata
            .get("has_answer")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }
}

/// A raw session with its source timestamp and opaque dataset identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LongMemEvalSession {
    pub source_index: usize,
    pub session_id: String,
    pub date: String,
    pub date_ms: i64,
    pub messages: Vec<LongMemEvalMessage>,
}

/// One valid user/assistant pair after applying LightMem's pairing rules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LongMemEvalPair {
    pub source_record_id: Uuid,
    pub session_index: usize,
    pub session_id: String,
    pub session_date_ms: i64,
    pub user_index: usize,
    pub assistant_index: usize,
    pub pair_index: usize,
    pub user_content: String,
    pub assistant_content: String,
    /// This is evaluator metadata. It is not included in the build record.
    pub has_answer: bool,
    /// These flags reproduce the final-turn flush decision in the reference
    /// LightMem script. The current generic offline runner records them in
    /// metadata for a provider-specific adapter to consume later.
    pub force_segment: bool,
    pub force_extract: bool,
}

/// Evaluator-only metadata for one LongMemEval question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LongMemEvalReference {
    pub sample_id: String,
    pub question_type: String,
    pub question: String,
    pub question_date: String,
    pub question_date_ms: i64,
    pub answer: String,
    pub answer_session_ids: Vec<String>,
    pub relevant_record_ids: BTreeSet<String>,
}

/// One question and its parsed raw/session view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LongMemEvalCase {
    pub question_id: String,
    pub question_type: String,
    pub question: String,
    pub question_date: String,
    pub question_date_ms: i64,
    pub answer: String,
    pub answer_session_ids: Vec<String>,
    /// Unknown top-level fields from the source sample. They stay on the
    /// evaluator side and are never copied into provider-visible records.
    pub source_metadata: BTreeMap<String, Value>,
    pub sessions: Vec<LongMemEvalSession>,
    pub pairs: Vec<LongMemEvalPair>,
    pub reference: LongMemEvalReference,
    pub offline_sample: OfflineDatasetSample,
}

impl LongMemEvalCase {
    /// Returns a one-question source. A separate source/run per question is
    /// required because the official LightMem script uses one collection per
    /// question and the generic runner builds all samples before querying.
    fn source(&self, dataset_digest: impl Into<String>) -> LongMemEvalSource {
        LongMemEvalSource {
            sample: self.offline_sample.clone(),
            consumed: false,
            dataset_digest: dataset_digest.into(),
        }
    }
}

/// A replayable one-question source for [`crate::OfflineBuildThenQueryRunner`].
#[derive(Debug, Clone)]
pub struct LongMemEvalSource {
    sample: OfflineDatasetSample,
    consumed: bool,
    dataset_digest: String,
}

impl ReplayableOfflineDatasetSource for LongMemEvalSource {
    fn next_sample(&mut self) -> Result<Option<OfflineDatasetSample>, DatasetError> {
        if self.consumed {
            Ok(None)
        } else {
            self.consumed = true;
            Ok(Some(self.sample.clone()))
        }
    }

    fn rewind(&mut self) -> Result<(), DatasetError> {
        self.consumed = false;
        Ok(())
    }

    fn dataset_digest(&self) -> Option<&str> {
        Some(&self.dataset_digest)
    }
}

/// Parsed LongMemEval corpus. Cases are intentionally exposed separately so a
/// caller can create one isolated benchmark run per question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LongMemEvalCorpus {
    pub dataset_digest: String,
    pub cases: Vec<LongMemEvalCase>,
}

impl LongMemEvalCorpus {
    /// Reads the original JSON bytes, hashes those bytes, and parses the full
    /// top-level sample array without mutating it.
    pub fn from_reader(mut reader: impl Read) -> Result<Self, LongMemEvalError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let dataset_digest = format!("sha256:{:x}", Sha256::digest(&bytes));
        let raw: Vec<RawLongMemEvalSample> = serde_json::from_slice(&bytes)?;
        if raw.is_empty() {
            return Err(LongMemEvalError::Invalid(
                "LongMemEval JSON must contain at least one question".to_owned(),
            ));
        }
        let mut question_ids = BTreeSet::new();
        let cases = raw
            .into_iter()
            .map(|sample| {
                if !question_ids.insert(sample.question_id.clone()) {
                    return Err(LongMemEvalError::Invalid(format!(
                        "duplicate question_id {:?}",
                        sample.question_id
                    )));
                }
                build_case(sample)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            dataset_digest,
            cases,
        })
    }

    pub fn case(&self, question_id: &str) -> Option<&LongMemEvalCase> {
        self.cases
            .iter()
            .find(|case| case.question_id == question_id)
    }

    /// Returns the isolated one-question source bound to this corpus's raw
    /// byte digest.
    pub fn source(&self, question_id: &str) -> Option<LongMemEvalSource> {
        self.case(question_id)
            .map(|case| case.source(self.dataset_digest.clone()))
    }
}

#[derive(Debug, Error)]
pub enum LongMemEvalError {
    #[error("failed to read LongMemEval data: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid LongMemEval JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid LongMemEval sample: {0}")]
    Invalid(String),
}

#[derive(Debug, Deserialize)]
struct RawLongMemEvalSample {
    question_id: String,
    question_type: String,
    question: String,
    question_date: String,
    answer: Value,
    answer_session_ids: Vec<String>,
    haystack_dates: Vec<String>,
    haystack_session_ids: Vec<String>,
    haystack_sessions: Vec<Vec<LongMemEvalMessage>>,
    #[serde(flatten)]
    metadata: BTreeMap<String, Value>,
}

fn build_case(raw: RawLongMemEvalSample) -> Result<LongMemEvalCase, LongMemEvalError> {
    require_non_empty("question_id", &raw.question_id)?;
    require_non_empty("question_type", &raw.question_type)?;
    require_non_empty("question", &raw.question)?;
    require_non_empty("question_date", &raw.question_date)?;
    if raw.haystack_dates.len() != raw.haystack_sessions.len()
        || raw.haystack_session_ids.len() != raw.haystack_sessions.len()
    {
        return Err(LongMemEvalError::Invalid(
            "haystack_dates, haystack_session_ids, and haystack_sessions must have equal lengths"
                .to_owned(),
        ));
    }
    validate_answer_session_ids(
        &raw.question_id,
        &raw.answer_session_ids,
        &raw.haystack_session_ids,
    )?;
    let question_date_ms = parse_timestamp(&raw.question_date)?;
    let answer = answer_text(&raw.answer)?;
    if question_date_ms < 0 {
        return Err(LongMemEvalError::Invalid(
            "question_date must not precede the Unix epoch".to_owned(),
        ));
    }
    let session_count = raw.haystack_sessions.len();
    let mut sessions = Vec::with_capacity(session_count);
    let mut pairs = Vec::new();
    for (session_index, messages) in raw.haystack_sessions.into_iter().enumerate() {
        let session_id = raw.haystack_session_ids[session_index].clone();
        let date = raw.haystack_dates[session_index].clone();
        require_non_empty("haystack_session_id", &session_id)?;
        let date_ms = parse_timestamp(&date)?;
        if date_ms < 0 {
            return Err(LongMemEvalError::Invalid(
                "haystack session dates must not precede the Unix epoch".to_owned(),
            ));
        }
        sessions.push(LongMemEvalSession {
            source_index: session_index,
            session_id: session_id.clone(),
            date: date.clone(),
            date_ms,
            messages: messages.clone(),
        });

        // This exactly mirrors the reference script: leading non-user
        // messages are removed, then floor(len/2) adjacent pairs are checked.
        let first_user = messages
            .iter()
            .position(|message| message.role == "user")
            .unwrap_or(messages.len());
        let normalized = &messages[first_user..];
        let turn_count = normalized.len() / 2;
        for pair_index in 0..turn_count {
            let user_index = first_user + pair_index * 2;
            let assistant_index = user_index + 1;
            let user = &messages[user_index];
            let assistant = &messages[assistant_index];
            if user.role != "user" || assistant.role != "assistant" {
                continue;
            }
            let final_reference_turn =
                session_index + 1 == session_count && pair_index + 1 == turn_count;
            let source_record_id = stable_record_id(
                &raw.question_id,
                session_index,
                &session_id,
                user_index,
                assistant_index,
            );
            pairs.push(LongMemEvalPair {
                source_record_id,
                session_index,
                session_id: session_id.clone(),
                session_date_ms: date_ms,
                user_index,
                assistant_index,
                pair_index,
                user_content: user.content.clone(),
                assistant_content: assistant.content.clone(),
                has_answer: user.has_answer() || assistant.has_answer(),
                force_segment: final_reference_turn,
                force_extract: final_reference_turn,
            });
        }
    }
    if pairs.is_empty() {
        return Err(LongMemEvalError::Invalid(format!(
            "question {:?} has no valid user/assistant pair",
            raw.question_id
        )));
    }

    let relevant_record_ids = relevant_record_ids(&pairs);
    if relevant_record_ids.is_empty() && !is_abstention_question(&raw.question_id) {
        return Err(LongMemEvalError::Invalid(format!(
            "question {:?} has no answer-labeled pair",
            raw.question_id
        )));
    }
    let reference = LongMemEvalReference {
        sample_id: raw.question_id.clone(),
        question_type: raw.question_type.clone(),
        question: raw.question.clone(),
        question_date: raw.question_date.clone(),
        question_date_ms,
        answer: answer.clone(),
        answer_session_ids: raw.answer_session_ids.clone(),
        relevant_record_ids: relevant_record_ids.clone(),
    };
    let records = pairs
        .iter()
        .enumerate()
        .map(|(index, pair)| {
            // The benchmark has no wall-clock ingest event. Use a stable
            // replay clock immediately before the question cutoff so the
            // generic visibility contract remains in the same Unix-ms domain
            // as `question_date_ms`; keep the source order separately.
            let ingest_time_ms =
                question_date_ms.saturating_sub((pairs.len() - index) as i64) as u64;
            build_record(&raw.question_id, pair, ingest_time_ms, index as u64 + 1)
        })
        .collect::<Vec<_>>();
    let offline_sample = OfflineDatasetSample {
        sample: DatasetSample {
            sample_id: raw.question_id.clone(),
            observation: AgentObservation {
                parts: vec![ObservationPart::Text(format!(
                    "Question time: {}\nQuestion: {}",
                    raw.question_date, raw.question
                ))],
            },
            query: MemoryQuery {
                input: Some(ContentRef::Text {
                    text: raw.question.clone(),
                }),
                // The runner binds an empty namespace to the isolated run
                // namespace before calling the provider.
                namespace: String::new(),
                scope: Some(MemoryScope::Personalized),
                subject: None,
                episode_id: None,
                kinds: vec![MemoryKind::Episodic],
                modalities: vec![Modality::Text],
                time_range: None,
                // The official LightMem experiment does not enforce a
                // provider visibility cutoff; question_date remains in the
                // reference for QA and temporal diagnostics.
                visibility_cutoff_ms: None,
                graph: None,
                budget: MemoryBudget {
                    max_hits: 20,
                    ..MemoryBudget::default()
                },
            },
            session_id: None,
            episode_id: None,
            terminal: true,
            terminal_reason: Some(TerminalReason::Success),
        },
        build: RecordIngestRequest {
            records,
            relations: Vec::new(),
            idempotency_key: format!("longmemeval/build/{}", raw.question_id),
        },
    };

    Ok(LongMemEvalCase {
        question_id: raw.question_id,
        question_type: raw.question_type,
        question: raw.question,
        question_date: raw.question_date,
        question_date_ms,
        answer,
        answer_session_ids: raw.answer_session_ids,
        source_metadata: raw.metadata,
        sessions,
        pairs,
        reference,
        offline_sample,
    })
}

fn build_record(
    question_id: &str,
    pair: &LongMemEvalPair,
    ingest_time_ms: u64,
    ingest_ordinal: u64,
) -> MemoryRecord {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "longmemeval.dataset_namespace".to_owned(),
        json!(DATASET_NAMESPACE),
    );
    metadata.insert("longmemeval.question_id".to_owned(), json!(question_id));
    metadata.insert("longmemeval.session_id".to_owned(), json!(pair.session_id));
    metadata.insert(
        "longmemeval.session_index".to_owned(),
        json!(pair.session_index),
    );
    metadata.insert(
        "longmemeval.session_date_ms".to_owned(),
        json!(pair.session_date_ms),
    );
    metadata.insert("longmemeval.user_index".to_owned(), json!(pair.user_index));
    metadata.insert(
        "longmemeval.assistant_index".to_owned(),
        json!(pair.assistant_index),
    );
    metadata.insert("longmemeval.pair_index".to_owned(), json!(pair.pair_index));
    metadata.insert(
        "longmemeval.ingest_ordinal".to_owned(),
        json!(ingest_ordinal),
    );
    metadata.insert(
        "longmemeval.ingest_clock".to_owned(),
        json!("replay_before_question_unix_ms"),
    );
    metadata.insert(
        "lightmem.force_segment".to_owned(),
        json!(pair.force_segment),
    );
    metadata.insert(
        "lightmem.force_extract".to_owned(),
        json!(pair.force_extract),
    );
    // The public memory content follows the official `messages_use=user_only`
    // profile. Pair-aware providers can consume `LongMemEvalCase::pairs`
    // directly; assistant text is deliberately excluded from this generic
    // record so a provider cannot index it accidentally.
    MemoryRecord {
        item: MemoryItem {
            id: pair.source_record_id,
            version: 1,
            kind: MemoryKind::Episodic,
            scope: MemoryScope::Personalized,
            modality: Modality::Text,
            content: ContentRef::Text {
                text: pair.user_content.clone(),
            },
            namespace: String::new(),
            episode_id: Some(pair.session_id.clone()),
            subject: None,
            source: DATASET_SOURCE.to_owned(),
            metadata,
            abstraction_level: 0,
            event_time: Some(pi_agent_memory::TimeRange {
                clock_domain: "unix_ms".to_owned(),
                start_ms: pair.session_date_ms,
                end_ms: None,
            }),
            // Event time describes when the conversation happened. Ingest
            // time describes when offline replay accepted the record, in the
            // same Unix-ms domain as the question cutoff. The source order is
            // retained separately in `longmemeval.ingest_ordinal`.
            ingest_time_ms,
            schema_version: 1,
        },
        episode: Some(EpisodeRef {
            dataset_or_run_id: DATASET_SOURCE.to_owned(),
            episode_id: pair.session_id.clone(),
            task: question_id.to_owned(),
            time_range: None,
            dataset_revision: "raw".to_owned(),
            source: DATASET_SOURCE.to_owned(),
        }),
        evidence: Vec::new(),
        provenance: MemoryProvenance {
            producer: "longmemeval.adapter".to_owned(),
            source_id: pair.session_id.clone(),
            source_event_id: Some(format!(
                "session:{}:pair:{}:{}",
                pair.session_index, pair.user_index, pair.assistant_index
            )),
            parent_record_ids: Vec::new(),
            method_version: Some("1".to_owned()),
        },
    }
}

fn relevant_record_ids(pairs: &[LongMemEvalPair]) -> BTreeSet<String> {
    // Canonical retrieval metrics need pair-level labels. A session-level
    // fallback would silently widen the reference set and make methods look
    // better than the dataset annotation supports.
    pairs
        .iter()
        .filter(|pair| pair.has_answer)
        .map(|pair| pair.source_record_id.to_string())
        .collect::<BTreeSet<_>>()
}

fn is_abstention_question(question_id: &str) -> bool {
    question_id.ends_with("_abs")
}

fn stable_record_id(
    question_id: &str,
    session_index: usize,
    session_id: &str,
    user_index: usize,
    assistant_index: usize,
) -> Uuid {
    // Hash a structured tuple instead of delimiter-joining opaque dataset
    // strings. Question and session IDs are external data and may contain the
    // delimiter, so textual concatenation would make distinct source tuples
    // collide before SHA-256 is applied.
    let identity = (
        DATASET_SOURCE,
        1_u32,
        question_id,
        session_index,
        session_id,
        user_index,
        assistant_index,
    );
    let identity = serde_json::to_vec(&identity).expect("source identity serialization");
    let digest = Sha256::digest(identity);
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    // Use the UUID version/variant bits for recognizable serialization while
    // deriving the remaining bytes from the stable source tuple with SHA-256.
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

fn require_non_empty(field: &str, value: &str) -> Result<(), LongMemEvalError> {
    if value.trim().is_empty() {
        return Err(LongMemEvalError::Invalid(format!(
            "{field} must be non-empty"
        )));
    }
    Ok(())
}

fn validate_answer_session_ids(
    question_id: &str,
    answer_session_ids: &[String],
    haystack_session_ids: &[String],
) -> Result<(), LongMemEvalError> {
    if answer_session_ids.is_empty() && !is_abstention_question(question_id) {
        return Err(LongMemEvalError::Invalid(format!(
            "question {question_id:?} must provide answer_session_ids unless it is an abstention"
        )));
    }

    let haystack_ids = haystack_session_ids.iter().collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    for session_id in answer_session_ids {
        require_non_empty("answer_session_id", session_id)?;
        if !seen.insert(session_id) {
            return Err(LongMemEvalError::Invalid(format!(
                "question {question_id:?} contains duplicate answer_session_id {session_id:?}"
            )));
        }
        if !haystack_ids.contains(&session_id) {
            return Err(LongMemEvalError::Invalid(format!(
                "question {question_id:?} references answer_session_id {session_id:?} that is not present in haystack_session_ids"
            )));
        }
    }
    Ok(())
}

fn answer_text(value: &Value) -> Result<String, LongMemEvalError> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string()),
        Value::Bool(value) => Ok(value.to_string()),
        _ => Err(LongMemEvalError::Invalid(
            "answer must be a string, number, or boolean".to_owned(),
        )),
    }
}

fn parse_timestamp(value: &str) -> Result<i64, LongMemEvalError> {
    let (date, rest) = value.split_once(" (").ok_or_else(|| {
        LongMemEvalError::Invalid(format!(
            "timestamp {value:?} must use YYYY/MM/DD (Day) HH:MM"
        ))
    })?;
    let time = rest.split_once(") ").map(|(_, time)| time).ok_or_else(|| {
        LongMemEvalError::Invalid(format!("timestamp {value:?} must include a weekday"))
    })?;
    let date_parts = date.split('/').collect::<Vec<_>>();
    let time_parts = time.split(':').collect::<Vec<_>>();
    if date_parts.len() != 3 || time_parts.len() != 2 {
        return Err(LongMemEvalError::Invalid(format!(
            "timestamp {value:?} must use YYYY/MM/DD (Day) HH:MM"
        )));
    }
    let year = parse_component(date_parts[0], "year", value)?;
    let month = parse_component(date_parts[1], "month", value)?;
    let day = parse_component(date_parts[2], "day", value)?;
    let hour = parse_component(time_parts[0], "hour", value)?;
    let minute = parse_component(time_parts[1], "minute", value)?;
    if !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
        || hour > 23
        || minute > 59
    {
        return Err(LongMemEvalError::Invalid(format!(
            "timestamp {value:?} contains an invalid date or time"
        )));
    }
    let days = days_from_civil(year, month, day);
    Ok(days
        .saturating_mul(86_400_000)
        .saturating_add(i64::from(hour) * 3_600_000)
        .saturating_add(i64::from(minute) * 60_000))
}

fn parse_component(component: &str, field: &str, value: &str) -> Result<i32, LongMemEvalError> {
    component.parse::<i32>().map_err(|_| {
        LongMemEvalError::Invalid(format!("timestamp {value:?} has an invalid {field}"))
    })
}

fn days_in_month(year: i32, month: i32) -> i32 {
    match month {
        2 if year % 400 == 0 || (year % 4 == 0 && year % 100 != 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

// Howard Hinnant's civil-date conversion, expressed without a timezone or
// platform dependency. LongMemEval timestamps are naive benchmark timestamps;
// this adapter treats them as UTC for a stable visibility/reference value.
fn days_from_civil(year: i32, month: i32, day: i32) -> i64 {
    let year = i64::from(year) - i64::from(month <= 2);
    let era = (if year >= 0 { year } else { year - 399 }) / 400;
    let year_of_era = year - era * 400;
    let month_prime = i64::from(month) + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * month_prime + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn fixture() -> &'static str {
        r#"[
          {
            "question_id": "q-1",
            "question_type": "single-session-user",
            "question": "What degree did I graduate with?",
            "question_date": "2023/05/30 (Tue) 23:40",
            "answer": "Business Administration",
            "answer_session_ids": ["answer-session"],
            "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 02:21"],
            "haystack_session_ids": ["session-1", "answer-session"],
            "haystack_sessions": [
              [
                {"role":"assistant", "content":"leading assistant"},
                {"role":"user", "content":"first question"},
                {"role":"assistant", "content":"first answer"},
                {"role":"user", "content":"dangling user"}
              ],
              [
                {"role":"user", "content":"degree?", "has_answer":true},
                {"role":"assistant", "content":"Business Administration", "has_answer":true}
              ]
            ]
          }
        ]"#
    }

    #[test]
    fn adapter_preserves_raw_sessions_and_builds_only_valid_pairs() {
        let mut value: Value = serde_json::from_str(fixture()).unwrap();
        value[0]["benchmark_extension"] = json!({"variant": "fixture-v1"});
        let corpus =
            LongMemEvalCorpus::from_reader(Cursor::new(serde_json::to_vec(&value).unwrap()))
                .unwrap();
        let case = &corpus.cases[0];

        assert_eq!(case.sessions.len(), 2);
        assert_eq!(case.sessions[0].messages.len(), 4);
        assert_eq!(case.pairs.len(), 2);
        assert_eq!(case.pairs[0].session_index, 0);
        assert_eq!(case.pairs[1].session_index, 1);
        assert_eq!(case.pairs[0].user_index, 1);
        assert_eq!(case.pairs[0].assistant_index, 2);
        assert!(case.pairs[1].has_answer);
        assert_eq!(case.reference.relevant_record_ids.len(), 1);
        assert_eq!(case.offline_sample.build.records.len(), 2);
        assert!(case.offline_sample.sample.terminal);
        assert!(matches!(
            &case.offline_sample.sample.observation.parts[0],
            ObservationPart::Text(text) if text.contains("What degree did I graduate with?")
        ));
        assert!(case.offline_sample.sample.query.namespace.is_empty());
        assert!(case
            .offline_sample
            .build
            .records
            .iter()
            .all(|record| record.item.namespace.is_empty()));
        assert!(!case.offline_sample.build.records[0]
            .item
            .metadata
            .contains_key("longmemeval.assistant_content"));
        assert_eq!(
            case.source_metadata["benchmark_extension"]["variant"],
            "fixture-v1"
        );
        assert!(!case.offline_sample.build.records[0]
            .item
            .metadata
            .contains_key("benchmark_extension"));
    }

    #[test]
    fn only_the_final_valid_reference_turn_gets_flush_flags() {
        let corpus = LongMemEvalCorpus::from_reader(Cursor::new(fixture())).unwrap();
        let pairs = &corpus.cases[0].pairs;

        assert!(!pairs[0].force_segment);
        assert!(pairs[1].force_segment);
        assert!(pairs[1].force_extract);
    }

    #[test]
    fn event_time_and_ingest_time_are_kept_separate() {
        let corpus = LongMemEvalCorpus::from_reader(Cursor::new(fixture())).unwrap();
        let records = &corpus.cases[0].offline_sample.build.records;
        let question_date_ms = corpus.cases[0].question_date_ms;

        assert_eq!(
            records[0].item.ingest_time_ms,
            (question_date_ms - 2) as u64
        );
        assert_eq!(
            records[1].item.ingest_time_ms,
            (question_date_ms - 1) as u64
        );
        assert_eq!(
            records[0].item.metadata["longmemeval.ingest_ordinal"],
            json!(1)
        );
        assert_eq!(
            records[1].item.metadata["longmemeval.ingest_ordinal"],
            json!(2)
        );
        assert_eq!(
            records[0].item.metadata["longmemeval.ingest_clock"],
            json!("replay_before_question_unix_ms")
        );
        assert_eq!(
            records[0]
                .item
                .event_time
                .as_ref()
                .expect("event time")
                .start_ms,
            corpus.cases[0].pairs[0].session_date_ms
        );
        assert_ne!(
            records[0].item.ingest_time_ms as i64,
            records[0]
                .item
                .event_time
                .as_ref()
                .expect("event time")
                .start_ms
        );
    }

    #[test]
    fn abstention_questions_allow_an_empty_relevance_set() {
        let mut value: Value = serde_json::from_str(fixture()).unwrap();
        value[0]["question_id"] = json!("q-1_abs");
        value[0]["answer"] = json!("I do not know");
        value[0]["answer_session_ids"] = json!([]);
        for message in value[0]["haystack_sessions"].as_array_mut().unwrap() {
            for message in message.as_array_mut().unwrap() {
                message.as_object_mut().unwrap().remove("has_answer");
            }
        }

        let corpus =
            LongMemEvalCorpus::from_reader(Cursor::new(serde_json::to_vec(&value).unwrap()))
                .unwrap();
        let case = &corpus.cases[0];
        assert!(case.reference.relevant_record_ids.is_empty());
        assert_eq!(case.pairs.len(), 2);
    }

    #[test]
    fn source_ids_are_stable_and_independent_of_run_identity() {
        let first = LongMemEvalCorpus::from_reader(Cursor::new(fixture())).unwrap();
        let second = LongMemEvalCorpus::from_reader(Cursor::new(fixture())).unwrap();

        let left = first.cases[0]
            .pairs
            .iter()
            .map(|pair| pair.source_record_id)
            .collect::<Vec<_>>();
        let right = second.cases[0]
            .pairs
            .iter()
            .map(|pair| pair.source_record_id)
            .collect::<Vec<_>>();
        assert_eq!(left, right);
        assert!(left.iter().all(|id| id.get_version_num() == 5));
        assert_ne!(
            stable_record_id("q-1", 0, "same-session-id", 0, 1),
            stable_record_id("q-1", 1, "same-session-id", 0, 1)
        );
        assert_ne!(
            stable_record_id("q:0", 1, "s", 2, 3),
            stable_record_id("q", 0, "1:s", 2, 3)
        );
    }

    #[test]
    fn duplicate_session_ids_still_have_distinct_pair_ids() {
        let mut value: Value = serde_json::from_str(fixture()).unwrap();
        value[0]["answer_session_ids"] = json!(["same-session"]);
        value[0]["haystack_session_ids"] = json!(["same-session", "same-session"]);

        let corpus =
            LongMemEvalCorpus::from_reader(Cursor::new(serde_json::to_vec(&value).unwrap()))
                .unwrap();
        let ids = corpus.cases[0]
            .pairs
            .iter()
            .map(|pair| pair.source_record_id)
            .collect::<BTreeSet<_>>();

        assert_eq!(ids.len(), corpus.cases[0].pairs.len());
    }

    #[test]
    fn one_question_source_replays_after_rewind() {
        let corpus = LongMemEvalCorpus::from_reader(Cursor::new(fixture())).unwrap();
        let mut source = corpus.source("q-1").unwrap();

        assert_eq!(
            source.dataset_digest(),
            Some(corpus.dataset_digest.as_str())
        );
        let first = source.next_sample().unwrap().unwrap();
        assert_eq!(first.sample.sample_id, "q-1");
        assert!(source.next_sample().unwrap().is_none());

        source.rewind().unwrap();
        let replay = source.next_sample().unwrap().unwrap();
        assert_eq!(replay, first);
    }

    #[test]
    fn timestamp_parser_uses_stable_utc_milliseconds() {
        assert_eq!(parse_timestamp("1970/01/01 (Thu) 00:00").unwrap(), 0);
        assert_eq!(parse_timestamp("1970/01/01 (Thu) 00:01").unwrap(), 60_000);
        assert!(parse_timestamp("1970/02/30 (Mon) 00:00").is_err());
    }

    #[test]
    fn answer_must_not_be_an_object() {
        let mut value: Value = serde_json::from_str(fixture()).unwrap();
        value[0]["answer"] = json!({"secret": "answer"});
        let error =
            LongMemEvalCorpus::from_reader(Cursor::new(serde_json::to_vec(&value).unwrap()))
                .unwrap_err();
        assert!(error.to_string().contains("answer must be"));
    }

    #[test]
    fn missing_answer_labels_are_rejected_instead_of_expanding_a_session() {
        let mut value: Value = serde_json::from_str(fixture()).unwrap();
        for session in value[0]["haystack_sessions"].as_array_mut().unwrap() {
            for message in session.as_array_mut().unwrap() {
                message.as_object_mut().unwrap().remove("has_answer");
            }
        }
        let error =
            LongMemEvalCorpus::from_reader(Cursor::new(serde_json::to_vec(&value).unwrap()))
                .unwrap_err();
        assert!(error.to_string().contains("no answer-labeled pair"));
    }

    #[test]
    fn answer_session_ids_must_bind_to_the_haystack() {
        let mut value: Value = serde_json::from_str(fixture()).unwrap();
        value[0]["answer_session_ids"] = json!(["missing-session"]);
        let error =
            LongMemEvalCorpus::from_reader(Cursor::new(serde_json::to_vec(&value).unwrap()))
                .unwrap_err();
        assert!(error
            .to_string()
            .contains("not present in haystack_session_ids"));
    }

    #[test]
    fn answer_session_ids_must_not_repeat() {
        let mut value: Value = serde_json::from_str(fixture()).unwrap();
        value[0]["answer_session_ids"] = json!(["answer-session", "answer-session"]);
        let error =
            LongMemEvalCorpus::from_reader(Cursor::new(serde_json::to_vec(&value).unwrap()))
                .unwrap_err();
        assert!(error.to_string().contains("duplicate answer_session_id"));
    }

    #[test]
    fn non_abstention_questions_require_answer_session_ids() {
        let mut value: Value = serde_json::from_str(fixture()).unwrap();
        value[0]["answer_session_ids"] = json!([]);
        let error =
            LongMemEvalCorpus::from_reader(Cursor::new(serde_json::to_vec(&value).unwrap()))
                .unwrap_err();
        assert!(error
            .to_string()
            .contains("must provide answer_session_ids"));
    }
}
