//! Deterministic memory-need prediction.
//!
//! The heuristic predictor is intentionally small. It gives the coordinator a
//! stable cold-start policy while leaving the [`NeedPredictor`] boundary open
//! for a small model or a JitRL-style scorer later.

use std::{error::Error, fmt};

use pi_agent_memory::{ContentRef, MemoryBudget, MemoryQuery};
use uuid::Uuid;

use crate::contracts::{MemoryNeed, NeedHorizon, TaskState};

/// Pluggable boundary for converting task state into memory needs.
pub trait NeedPredictor: Send + Sync {
    fn predict(&self, state: &TaskState, now_ms: u64) -> Result<Vec<MemoryNeed>, PredictionError>;
}

/// Configuration for the deterministic cold-start predictor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeuristicPredictorConfig {
    pub max_needs: usize,
    pub max_hits_per_need: usize,
}

impl Default for HeuristicPredictorConfig {
    fn default() -> Self {
        Self {
            max_needs: 2,
            max_hits_per_need: 8,
        }
    }
}

/// Predicts the current request and, when available, the next planned request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeuristicNeedPredictor {
    config: HeuristicPredictorConfig,
}

impl Default for HeuristicNeedPredictor {
    fn default() -> Self {
        Self::new(HeuristicPredictorConfig::default())
    }
}

impl HeuristicNeedPredictor {
    pub const fn new(config: HeuristicPredictorConfig) -> Self {
        Self { config }
    }

    pub const fn config(self) -> HeuristicPredictorConfig {
        self.config
    }
}

impl NeedPredictor for HeuristicNeedPredictor {
    fn predict(&self, state: &TaskState, now_ms: u64) -> Result<Vec<MemoryNeed>, PredictionError> {
        validate_state(state)?;
        if self.config.max_needs == 0 || self.config.max_hits_per_need == 0 {
            return Ok(Vec::new());
        }

        let expiry = state
            .remaining_deadline_ms
            .and_then(|remaining| now_ms.checked_add(remaining));
        let mut needs = Vec::with_capacity(self.config.max_needs.min(2));

        if let Some(query_text) = first_non_empty([
            state.latest_query.as_deref(),
            Some(state.goal_summary.as_str()),
        ]) {
            needs.push(build_need(
                state,
                query_text,
                NeedHorizon::CurrentTurn,
                1.0,
                0.8,
                self.config.max_hits_per_need,
                now_ms,
                expiry,
            ));
        }

        if needs.len() < self.config.max_needs {
            if let Some(plan) = state
                .current_plan
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            {
                if Some(plan) != state.latest_query.as_deref()
                    && plan.trim() != state.goal_summary.trim()
                {
                    needs.push(build_need(
                        state,
                        plan,
                        NeedHorizon::NextTurn,
                        0.55,
                        0.6,
                        self.config.max_hits_per_need,
                        now_ms,
                        expiry,
                    ));
                }
            }
        }

        needs.truncate(self.config.max_needs);
        Ok(needs)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionError {
    EmptyNamespace,
    EmptyTaskSummary,
}

impl fmt::Display for PredictionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyNamespace => formatter.write_str("memory prediction requires a namespace"),
            Self::EmptyTaskSummary => formatter.write_str("memory prediction requires task text"),
        }
    }
}

impl Error for PredictionError {}

fn validate_state(state: &TaskState) -> Result<(), PredictionError> {
    if state.namespace.trim().is_empty() {
        return Err(PredictionError::EmptyNamespace);
    }
    if first_non_empty([
        state.latest_query.as_deref(),
        Some(state.goal_summary.as_str()),
    ])
    .is_none()
    {
        return Err(PredictionError::EmptyTaskSummary);
    }
    Ok(())
}

fn first_non_empty<const N: usize>(values: [Option<&str>; N]) -> Option<&str> {
    values
        .into_iter()
        .flatten()
        .find(|value| !value.trim().is_empty())
        .map(str::trim)
}

#[allow(clippy::too_many_arguments)]
fn build_need(
    state: &TaskState,
    text: &str,
    horizon: NeedHorizon,
    confidence: f32,
    estimated_benefit: f32,
    max_hits: usize,
    created_at_ms: u64,
    expires_at_ms: Option<u64>,
) -> MemoryNeed {
    MemoryNeed {
        need_id: Uuid::new_v4(),
        query: MemoryQuery {
            input: Some(ContentRef::Text {
                text: text.to_string(),
            }),
            namespace: state.namespace.clone(),
            scope: None,
            subject: None,
            episode_id: None,
            kinds: Vec::new(),
            modalities: state.active_modalities.clone(),
            time_range: None,
            visibility_cutoff_ms: Some(created_at_ms),
            graph: None,
            budget: MemoryBudget {
                max_hits,
                max_context_tokens: state.context_token_budget as usize,
                ..MemoryBudget::default()
            },
        },
        provider_hints: Vec::new(),
        horizon,
        confidence,
        estimated_benefit,
        created_at_ms,
        expires_at_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{MemoryEpoch, PolicyEpoch, ToolObservation, ToolOutcome};
    use pi_agent_memory::Modality;
    use pi_agent_runtime::{Revision, RunId};

    fn state() -> TaskState {
        TaskState {
            run_id: RunId::new(),
            turn_id: Uuid::new_v4(),
            namespace: "benchmark/code".to_string(),
            runtime_revision: Revision::new(4),
            memory_epoch: MemoryEpoch::new(2),
            policy_epoch: PolicyEpoch::new(1),
            goal_summary: "fix parser regression".to_string(),
            latest_query: Some("inspect parser history".to_string()),
            current_plan: Some("run the parser regression test".to_string()),
            recent_tools: vec![ToolObservation {
                tool_name: "read_file".to_string(),
                outcome: ToolOutcome::Succeeded,
                summary: "opened parser.rs".to_string(),
            }],
            active_modalities: vec![Modality::Text],
            remaining_deadline_ms: Some(500),
            context_token_budget: 1_024,
        }
    }

    #[test]
    fn predicts_current_and_next_turn_needs_with_a_deadline() {
        let predictor = HeuristicNeedPredictor::default();
        let needs = predictor.predict(&state(), 1_000).unwrap();

        assert_eq!(needs.len(), 2);
        assert_eq!(needs[0].horizon, NeedHorizon::CurrentTurn);
        assert_eq!(needs[1].horizon, NeedHorizon::NextTurn);
        assert_eq!(needs[0].expires_at_ms, Some(1_500));
        assert_eq!(needs[0].query.visibility_cutoff_ms, Some(1_000));
    }

    #[test]
    fn rejects_an_unscoped_task() {
        let predictor = HeuristicNeedPredictor::default();
        let mut task = state();
        task.namespace.clear();

        assert_eq!(
            predictor.predict(&task, 1_000),
            Err(PredictionError::EmptyNamespace)
        );
    }
}
