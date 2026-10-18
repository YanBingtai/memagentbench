//! Provider retrieval and deterministic result aggregation.
//!
//! This module performs the first side-effecting step of the memory agent: it
//! invokes the methods selected by the capability router. It keeps provider
//! failures in the report so one unavailable method does not hide results from
//! the other methods.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use pi_agent_memory::{MemoryError, MemoryHit, MemoryRegistry, OperationContext};
use tokio::{sync::Semaphore, task::JoinSet, time::timeout};
use uuid::Uuid;

use crate::{contracts::MemoryNeed, router::ProviderRouter, RouteError};

/// A provider error retained alongside successful retrievals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderFailure {
    pub method_id: String,
    pub error: MemoryError,
}

/// Aggregated retrieval output for one planned memory need.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalReport {
    pub need_id: Uuid,
    pub hits: Vec<MemoryHit>,
    /// Response-level snapshot when every successful provider observed the
    /// same generation. Mixed-provider responses leave this empty while each
    /// hit retains its provider snapshot.
    pub snapshot_id: Option<String>,
    pub provider_failures: Vec<ProviderFailure>,
    pub truncated: bool,
}

const MAX_PROVIDER_HITS: usize = 4_096;
const MAX_MERGED_HITS: usize = 16_384;
const MAX_PROVIDER_CONCURRENCY: usize = 16;

/// Errors that prevent the coordinator from producing a retrieval report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetrievalError {
    Route(RouteError),
    NamespaceMismatch {
        query_namespace: String,
        context_namespace: String,
    },
    ProviderUnavailable {
        method_id: String,
    },
    ProviderRejected {
        method_id: String,
        error: MemoryError,
    },
    ProviderContractViolation {
        method_id: String,
        message: String,
    },
    ProviderResponseTooLarge {
        method_id: String,
        hits: usize,
    },
    ExpiredNeed,
}

impl fmt::Display for RetrievalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Route(error) => write!(formatter, "memory retrieval routing failed: {error}"),
            Self::NamespaceMismatch {
                query_namespace,
                context_namespace,
            } => write!(
                formatter,
                "memory query namespace {query_namespace:?} does not match operation namespace {context_namespace:?}"
            ),
            Self::ProviderUnavailable { method_id } => {
                write!(formatter, "selected memory provider is unavailable: {method_id}")
            }
            Self::ProviderRejected { method_id, error } => {
                write!(formatter, "memory provider rejected the request ({method_id}): {error}")
            }
            Self::ProviderContractViolation { method_id, message } => {
                write!(formatter, "memory provider contract violation ({method_id}): {message}")
            }
            Self::ProviderResponseTooLarge { method_id, hits } => write!(
                formatter,
                "memory provider {method_id} returned too many hits: {hits}"
            ),
            Self::ExpiredNeed => formatter.write_str("memory need expired before retrieval"),
        }
    }
}

impl Error for RetrievalError {}

impl From<RouteError> for RetrievalError {
    fn from(error: RouteError) -> Self {
        Self::Route(error)
    }
}

/// Calls selected providers and merges their hits under the query budget.
pub struct RetrievalCoordinator<'a> {
    registry: &'a MemoryRegistry,
    router: ProviderRouter<'a>,
}

impl<'a> RetrievalCoordinator<'a> {
    /// Creates a coordinator over an existing provider registry.
    pub const fn new(registry: &'a MemoryRegistry) -> Self {
        Self {
            registry,
            router: ProviderRouter::new(registry),
        }
    }

    /// Retrieves from all compatible providers in deterministic route order.
    ///
    /// Provider failures are returned in [`RetrievalReport`] so callers can
    /// choose fail-soft behavior. A routing error or namespace mismatch is a
    /// hard error because no safe retrieval plan exists in those cases.
    pub async fn retrieve(
        &self,
        need: &MemoryNeed,
        context: OperationContext,
    ) -> Result<RetrievalReport, RetrievalError> {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        self.retrieve_at(need, context, now_ms).await
    }

    /// Retrieves with a caller-provided wall-clock timestamp used only for
    /// checking absolute need expiry. Elapsed time is measured with a
    /// monotonic clock while provider calls are in flight.
    pub async fn retrieve_at(
        &self,
        need: &MemoryNeed,
        context: OperationContext,
        now_ms: u64,
    ) -> Result<RetrievalReport, RetrievalError> {
        if need.query.namespace != context.namespace {
            return Err(RetrievalError::NamespaceMismatch {
                query_namespace: need.query.namespace.clone(),
                context_namespace: context.namespace,
            });
        }
        if need.expires_at_ms.is_some_and(|expiry| now_ms >= expiry) {
            return Err(RetrievalError::ExpiredNeed);
        }

        let selections = self.router.route(need)?;
        if need.query.budget.max_hits == 0 || need.query.budget.max_evidence_bytes == 0 {
            return Ok(RetrievalReport {
                need_id: need.need_id,
                hits: Vec::new(),
                snapshot_id: None,
                provider_failures: Vec::new(),
                truncated: false,
            });
        }
        let started = Instant::now();
        let deadline_ms = retrieval_deadline_ms(need, &context, now_ms);
        let mut hits = Vec::new();
        let mut provider_failures = Vec::new();
        let mut provider_truncated = false;
        let mut response_snapshot = None;
        let mut observed_provider_response = false;
        let mut snapshots_match = true;

        let semaphore = Arc::new(Semaphore::new(MAX_PROVIDER_CONCURRENCY));
        let mut attempts = JoinSet::new();
        for (order, selection) in selections.into_iter().enumerate() {
            let Some(method) = self.registry.get(&selection.method_id) else {
                return Err(RetrievalError::ProviderUnavailable {
                    method_id: selection.method_id,
                });
            };

            let mut provider_query = need.query.clone();
            // Every provider receives the same bounded top-k request. The
            // coordinator ranks the merged result globally afterwards, so an
            // early provider cannot hide a better hit from a later provider.
            provider_query.budget.max_hits = need.query.budget.max_hits;
            provider_query.budget.max_evidence_bytes = need.query.budget.max_evidence_bytes;
            let method_id = selection.method_id;
            let provider_context_base = context.clone();
            let semaphore = Arc::clone(&semaphore);
            attempts.spawn(async move {
                let permit = semaphore
                    .acquire_owned()
                    .await
                    .expect("provider concurrency semaphore should remain open");
                let remaining = remaining_ms(deadline_ms, now_ms, started.elapsed());
                if remaining == Some(0) {
                    drop(permit);
                    return (order, method_id, Err(()));
                }
                let provider_context =
                    context_for_provider(&provider_context_base, &method_id, remaining);
                let response = match remaining {
                    Some(remaining_ms) => match timeout(
                        Duration::from_millis(remaining_ms),
                        method.retrieve(provider_query, provider_context),
                    )
                    .await
                    {
                        Ok(result) => Ok(result),
                        Err(_) => Err(()),
                    },
                    None => Ok(method.retrieve(provider_query, provider_context).await),
                };
                drop(permit);
                (order, method_id, response)
            });
        }

        let mut completed = Vec::new();
        while let Some(result) = attempts.join_next().await {
            match result {
                Ok(attempt) => completed.push(attempt),
                Err(error) => {
                    return Err(RetrievalError::ProviderContractViolation {
                        method_id: "provider-task".to_owned(),
                        message: format!("provider task failed: {error}"),
                    });
                }
            }
        }
        completed.sort_by_key(|(order, _, _)| *order);

        for (_, method_id, response) in completed {
            let provider_max_hits = need.query.budget.max_hits;
            match response {
                Err(_) => {
                    provider_truncated = true;
                    provider_failures.push(ProviderFailure {
                        method_id,
                        error: MemoryError::Timeout,
                    });
                }
                Ok(Err(error)) if is_fail_closed(&error) => {
                    return Err(RetrievalError::ProviderRejected { method_id, error });
                }
                Ok(Err(error)) => provider_failures.push(ProviderFailure { method_id, error }),
                Ok(Ok(response)) => {
                    if observed_provider_response {
                        if response_snapshot != response.snapshot_id {
                            snapshots_match = false;
                        }
                    } else {
                        response_snapshot = response.snapshot_id.clone();
                        observed_provider_response = true;
                    }
                    if response.hits.len() > MAX_PROVIDER_HITS {
                        return Err(RetrievalError::ProviderResponseTooLarge {
                            method_id,
                            hits: response.hits.len(),
                        });
                    }
                    provider_truncated |= response.truncated;
                    let mut provider_ranks = BTreeSet::new();
                    for hit in response.hits {
                        if hit.method_id != method_id {
                            return Err(RetrievalError::ProviderContractViolation {
                                method_id,
                                message: format!(
                                    "hit {} was attributed to {}",
                                    hit.item_id, hit.method_id
                                ),
                            });
                        }
                        if let Err(message) = validate_provider_hit(
                            &hit,
                            &method_id,
                            provider_max_hits,
                            &mut provider_ranks,
                        ) {
                            return Err(RetrievalError::ProviderContractViolation {
                                method_id,
                                message,
                            });
                        }
                        let hit_bytes = hit_size_bytes(&hit);
                        if hit_bytes > need.query.budget.max_evidence_bytes {
                            provider_truncated = true;
                            continue;
                        }
                        hits.push(hit);
                        if hits.len() >= MAX_MERGED_HITS {
                            provider_truncated = true;
                            break;
                        }
                    }
                }
            }
        }

        let mut unique_hits = BTreeMap::new();
        for hit in hits {
            let key = (hit.item_id, hit.version);
            if let Some(existing) = unique_hits.get_mut(&key) {
                merge_hit_evidence(existing, hit);
            } else {
                unique_hits.insert(key, hit);
            }
        }
        let mut hits = unique_hits.into_values().collect::<Vec<_>>();
        hits.sort_by(compare_hits);
        let mut selected = Vec::with_capacity(need.query.budget.max_hits);
        let mut evidence_bytes = 0_usize;
        for mut hit in hits {
            if selected.len() >= need.query.budget.max_hits {
                provider_truncated = true;
                break;
            }
            let hit_bytes = hit_size_bytes(&hit);
            if hit_bytes
                > need
                    .query
                    .budget
                    .max_evidence_bytes
                    .saturating_sub(evidence_bytes)
            {
                provider_truncated = true;
                continue;
            }
            evidence_bytes = evidence_bytes.saturating_add(hit_bytes);
            hit.rank = selected.len() + 1;
            selected.push(hit);
        }

        Ok(RetrievalReport {
            need_id: need.need_id,
            hits: selected,
            snapshot_id: snapshots_match.then_some(response_snapshot).flatten(),
            provider_failures,
            truncated: provider_truncated,
        })
    }
}

fn validate_provider_hit(
    hit: &MemoryHit,
    method_id: &str,
    max_hits: usize,
    ranks: &mut BTreeSet<usize>,
) -> Result<(), String> {
    if hit.rank == 0 {
        return Err(format!("provider {method_id} returned rank 0"));
    }
    if hit.rank > max_hits {
        return Err(format!(
            "provider {method_id} returned rank {} above requested max_hits {max_hits}",
            hit.rank
        ));
    }
    if hit.score.is_some_and(|score| !score.is_finite()) {
        return Err(format!("provider {method_id} returned a non-finite score"));
    }
    if !ranks.insert(hit.rank) {
        return Err(format!(
            "provider {method_id} returned duplicate rank {}",
            hit.rank
        ));
    }
    Ok(())
}

fn merge_hit_evidence(existing: &mut MemoryHit, candidate: MemoryHit) {
    for source_record_id in candidate.source_record_ids {
        if !existing.source_record_ids.contains(&source_record_id) {
            existing.source_record_ids.push(source_record_id);
        }
    }
    existing.source_record_ids.sort_unstable();
    for span in candidate.evidence {
        if !existing.evidence.contains(&span) {
            existing.evidence.push(span);
        }
    }
    for reference in candidate.evidence_refs {
        if !existing.evidence_refs.contains(&reference) {
            existing.evidence_refs.push(reference);
        }
    }
    if existing.projection.is_none() {
        existing.projection = candidate.projection;
    }
}

fn context_for_provider(
    context: &OperationContext,
    method_id: &str,
    remaining_ms: Option<u64>,
) -> OperationContext {
    let mut provider_context = context.clone();
    provider_context.timeout_ms = match (provider_context.timeout_ms, remaining_ms) {
        (Some(timeout), Some(remaining)) => Some(timeout.min(remaining)),
        (None, Some(remaining)) => Some(remaining),
        (timeout, None) => timeout,
    };
    if let Some(base_key) = context.idempotency_key.as_ref() {
        provider_context.idempotency_key = Some(format!("{base_key}:{method_id}"));
    }
    provider_context
}

fn is_fail_closed(error: &MemoryError) -> bool {
    matches!(
        error,
        MemoryError::Unauthorized | MemoryError::InvalidRequest(_)
    )
}

fn retrieval_deadline_ms(
    need: &MemoryNeed,
    context: &OperationContext,
    now_ms: u64,
) -> Option<u64> {
    [
        context
            .timeout_ms
            .map(|timeout| now_ms.saturating_add(timeout)),
        need.query
            .budget
            .max_latency_ms
            .map(|latency| now_ms.saturating_add(latency)),
        need.expires_at_ms,
    ]
    .into_iter()
    .flatten()
    .min()
}

fn remaining_ms(deadline_ms: Option<u64>, now_ms: u64, elapsed: Duration) -> Option<u64> {
    deadline_ms.map(|deadline| {
        deadline.saturating_sub(
            now_ms.saturating_add(elapsed.as_millis().try_into().unwrap_or(u64::MAX)),
        )
    })
}

fn hit_size_bytes(hit: &MemoryHit) -> usize {
    serde_json::to_vec(&(&hit.evidence, &hit.evidence_refs, &hit.projection))
        .map_or(usize::MAX, |bytes| bytes.len())
}

/// Orders higher finite scores first and puts missing or non-finite scores at
/// the end. Remaining fields provide deterministic tie-breakers across methods.
fn compare_hits(left: &MemoryHit, right: &MemoryHit) -> Ordering {
    compare_scores(left.score, right.score)
        .then_with(|| left.method_id.cmp(&right.method_id))
        .then_with(|| left.item_id.cmp(&right.item_id))
        .then_with(|| left.version.cmp(&right.version))
        .then_with(|| left.rank.cmp(&right.rank))
}

fn compare_scores(left: Option<f32>, right: Option<f32>) -> Ordering {
    match (score_class(left), score_class(right)) {
        (ScoreClass::Finite(left), ScoreClass::Finite(right)) => {
            right.partial_cmp(&left).unwrap_or(Ordering::Equal)
        }
        (ScoreClass::Finite(_), _) => Ordering::Less,
        (_, ScoreClass::Finite(_)) => Ordering::Greater,
        (ScoreClass::Missing, ScoreClass::Invalid) => Ordering::Less,
        (ScoreClass::Invalid, ScoreClass::Missing) => Ordering::Greater,
        _ => Ordering::Equal,
    }
}

#[derive(Debug, Clone, Copy)]
enum ScoreClass {
    Finite(f32),
    Missing,
    Invalid,
}

fn score_class(score: Option<f32>) -> ScoreClass {
    match score {
        Some(value) if value.is_finite() => ScoreClass::Finite(value),
        Some(_) => ScoreClass::Invalid,
        None => ScoreClass::Missing,
    }
}

#[cfg(test)]
mod tests {
    use pi_agent_memory::ContentRef;

    use super::*;

    fn hit(item_id: u128, method_id: &str, score: Option<f32>, rank: usize) -> MemoryHit {
        MemoryHit {
            item_id: Uuid::from_u128(item_id),
            source_record_ids: Vec::new(),
            version: 1,
            method_id: method_id.to_string(),
            rank,
            score,
            evidence: Vec::new(),
            evidence_refs: Vec::new(),
            projection: None,
            provenance: method_id.to_string(),
            snapshot_id: Some(format!("snapshot:{method_id}")),
            latency_ms: 1,
        }
    }

    #[test]
    fn merge_order_is_deterministic_and_missing_scores_are_last() {
        let mut hits = [
            hit(3, "text", None, 0),
            hit(2, "video", Some(0.8), 1),
            hit(1, "text", Some(0.8), 0),
            hit(4, "broken", Some(f32::NAN), 0),
        ];

        hits.sort_by(compare_hits);

        assert_eq!(hits[0].item_id, Uuid::from_u128(1));
        assert_eq!(hits[1].item_id, Uuid::from_u128(2));
        assert_eq!(hits[2].item_id, Uuid::from_u128(3));
        assert_eq!(hits[3].item_id, Uuid::from_u128(4));
    }

    #[test]
    fn merging_provider_hits_preserves_unique_source_lineage() {
        let mut existing = hit(1, "text", Some(0.8), 1);
        existing.source_record_ids = vec![Uuid::from_u128(3), Uuid::from_u128(1)];
        let mut candidate = hit(1, "video", Some(0.7), 1);
        candidate.source_record_ids = vec![Uuid::from_u128(2), Uuid::from_u128(1)];

        merge_hit_evidence(&mut existing, candidate);

        assert_eq!(
            existing.source_record_ids,
            vec![Uuid::from_u128(1), Uuid::from_u128(2), Uuid::from_u128(3)]
        );
    }

    #[test]
    fn blob_budget_counts_reference_metadata_not_declared_payload_size() {
        let content = ContentRef::Blob {
            uri: "cas://episode/video.mp4".to_owned(),
            media_type: "video/mp4".to_owned(),
            sha256: "sha256:digest".to_owned(),
            size_bytes: u64::MAX,
        };

        assert!(serde_json::to_vec(&content).unwrap().len() < 256);
    }

    #[test]
    fn provider_idempotency_keys_are_scoped_per_method() {
        let context = OperationContext {
            namespace: "benchmark/code".to_string(),
            run_id: pi_agent_runtime::RunId::new().as_uuid(),
            session_id: None,
            episode_id: None,
            timeout_ms: Some(100),
            idempotency_key: Some("run-1/need-1".to_string()),
        };

        let provider_context = context_for_provider(&context, "semantic-text", Some(100));

        assert_eq!(
            provider_context.idempotency_key.as_deref(),
            Some("run-1/need-1:semantic-text")
        );
        assert_eq!(provider_context.namespace, context.namespace);
    }

    #[test]
    fn provider_hit_contract_rejects_invalid_rank_score_and_duplicates() {
        let mut ranks = BTreeSet::new();
        let mut hit = hit(1, "text", Some(0.5), 0);
        assert!(validate_provider_hit(&hit, "text", 4, &mut ranks)
            .unwrap_err()
            .contains("rank 0"));

        hit.rank = 5;
        assert!(validate_provider_hit(&hit, "text", 4, &mut ranks)
            .unwrap_err()
            .contains("above requested"));

        hit.rank = 1;
        hit.score = Some(f32::NAN);
        assert!(validate_provider_hit(&hit, "text", 4, &mut ranks)
            .unwrap_err()
            .contains("non-finite"));

        hit.score = Some(0.5);
        assert!(validate_provider_hit(&hit, "text", 4, &mut ranks).is_ok());
        assert!(validate_provider_hit(&hit, "text", 4, &mut ranks)
            .unwrap_err()
            .contains("duplicate rank"));
    }
}
