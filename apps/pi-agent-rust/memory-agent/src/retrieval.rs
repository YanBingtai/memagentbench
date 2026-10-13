//! Provider retrieval and deterministic result aggregation.
//!
//! This module performs the first side-effecting step of the memory agent: it
//! invokes the methods selected by the capability router. It keeps provider
//! failures in the report so one unavailable method does not hide results from
//! the other methods.

use std::{
    cmp::Ordering,
    error::Error,
    fmt,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use pi_agent_memory::{
    ContentRef, EvidenceSpan, MemoryError, MemoryHit, MemoryRegistry, OperationContext,
};
use tokio::time::timeout;
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
    pub provider_failures: Vec<ProviderFailure>,
    pub truncated: bool,
}

const MAX_PROVIDER_HITS: usize = 4_096;
const MAX_MERGED_HITS: usize = 16_384;

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
                provider_failures: Vec::new(),
                truncated: false,
            });
        }
        let started = Instant::now();
        let deadline_ms = retrieval_deadline_ms(need, &context, now_ms);
        let mut hits = Vec::new();
        let mut provider_failures = Vec::new();
        let mut provider_truncated = false;

        for selection in selections {
            if hits.len() >= MAX_MERGED_HITS {
                provider_truncated = true;
                break;
            }
            let Some(method) = self.registry.get(&selection.method_id) else {
                return Err(RetrievalError::ProviderUnavailable {
                    method_id: selection.method_id,
                });
            };

            let remaining = remaining_ms(deadline_ms, now_ms, started.elapsed());
            if remaining == Some(0) {
                provider_truncated = true;
                break;
            }
            let provider_context = context_for_provider(&context, &selection.method_id, remaining);
            let mut provider_query = need.query.clone();
            // Every provider receives the same bounded top-k request. The
            // coordinator ranks the merged result globally afterwards, so an
            // early provider cannot hide a better hit from a later provider.
            provider_query.budget.max_hits = need.query.budget.max_hits;
            provider_query.budget.max_evidence_bytes = need.query.budget.max_evidence_bytes;
            let provider_future = method.retrieve(provider_query, provider_context);
            let response = match remaining {
                Some(remaining_ms) => timeout(Duration::from_millis(remaining_ms), provider_future)
                    .await
                    .map_err(|_| ()),
                None => Ok(provider_future.await),
            };
            match response {
                Err(_) => {
                    provider_truncated = true;
                    provider_failures.push(ProviderFailure {
                        method_id: selection.method_id,
                        error: MemoryError::Timeout,
                    });
                }
                Ok(Err(error)) if is_fail_closed(&error) => {
                    return Err(RetrievalError::ProviderRejected {
                        method_id: selection.method_id,
                        error,
                    });
                }
                Ok(Err(error)) => provider_failures.push(ProviderFailure {
                    method_id: selection.method_id,
                    error,
                }),
                Ok(Ok(response)) => {
                    if response.hits.len() > MAX_PROVIDER_HITS {
                        return Err(RetrievalError::ProviderResponseTooLarge {
                            method_id: selection.method_id,
                            hits: response.hits.len(),
                        });
                    }
                    provider_truncated |= response.truncated;
                    for hit in response.hits {
                        if hit.method_id != selection.method_id {
                            return Err(RetrievalError::ProviderContractViolation {
                                method_id: selection.method_id,
                                message: format!(
                                    "hit {} was attributed to {}",
                                    hit.item_id, hit.method_id
                                ),
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

        hits.sort_by(compare_hits);
        let mut selected = Vec::with_capacity(need.query.budget.max_hits);
        let mut evidence_bytes = 0_usize;
        for hit in hits {
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
            selected.push(hit);
        }

        Ok(RetrievalReport {
            need_id: need.need_id,
            hits: selected,
            provider_failures,
            truncated: provider_truncated,
        })
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
    let evidence = hit.evidence.iter().map(evidence_size_bytes).sum::<usize>();
    evidence.saturating_add(
        hit.projection
            .as_ref()
            .map(content_size_bytes)
            .unwrap_or_default(),
    )
}

fn content_size_bytes(content: &ContentRef) -> usize {
    match content {
        ContentRef::Text { text } => text.len(),
        ContentRef::Json { value } => {
            serde_json::to_vec(value).map_or(usize::MAX, |bytes| bytes.len())
        }
        ContentRef::Blob { size_bytes, .. } => usize::try_from(*size_bytes).unwrap_or(usize::MAX),
    }
}

fn evidence_size_bytes(evidence: &EvidenceSpan) -> usize {
    match evidence {
        EvidenceSpan::Text { .. } => std::mem::size_of::<(usize, usize)>(),
        EvidenceSpan::Image { .. } => std::mem::size_of::<(u32, u32, u32, u32)>(),
        EvidenceSpan::Audio { .. } | EvidenceSpan::Video { .. } => 24,
        EvidenceSpan::Trajectory { .. } => 16,
    }
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
    use super::*;

    fn hit(item_id: u128, method_id: &str, score: Option<f32>, rank: usize) -> MemoryHit {
        MemoryHit {
            item_id: Uuid::from_u128(item_id),
            version: 1,
            method_id: method_id.to_string(),
            rank,
            score,
            evidence: Vec::new(),
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
}
