//! Dataset-specific answer scoring for LongMemEval.
//!
//! Retrieval/build evaluation stays in the generic evaluator.  This module
//! consumes a reference answer only at evaluation time and records answer
//! metrics separately, so the answer cannot enter a provider's memory index.

use std::collections::HashMap;

use pi_agent_benchmark::{MetricDirection, MetricSpec, MetricValue, RunTrace};
use thiserror::Error;

const METRIC_VERSION: &str = "1";
pub const LONGMEMEVAL_JUDGE_METRIC: &str = "longmemeval.answer_judge_accuracy";

/// Returns whether a LongMemEval sample uses the official abstention task.
pub fn longmemeval_is_abstention_question(sample_id: &str) -> bool {
    sample_id.ends_with("_abs")
}

/// Returns the official LongMemEval yes/no judge prompt for one question.
pub fn longmemeval_judge_prompt(
    question_type: &str,
    question: &str,
    answer: &str,
    response: &str,
    abstention: bool,
) -> Result<String, LongMemEvalJudgeError> {
    let template = if abstention {
        "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    } else if question_type == "temporal-reasoning" {
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    } else if question_type == "knowledge-update" {
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    } else if question_type == "single-session-preference" {
        "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    } else if matches!(
        question_type,
        "single-session-user" | "single-session-assistant" | "multi-session"
    ) {
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    } else {
        return Err(LongMemEvalJudgeError::UnsupportedQuestionType(
            question_type.to_owned(),
        ));
    };
    Ok(template
        .replacen("{}", question, 1)
        .replacen("{}", answer, 1)
        .replacen("{}", response, 1))
}

/// Parses the first yes/no decision returned by the official judge prompt.
pub fn parse_longmemeval_judge_response(response: &str) -> Option<bool> {
    let first_line = response.trim().to_ascii_lowercase();
    let first_line = first_line.lines().next()?.trim();
    let tokens = first_line
        .replace(['.', '!', ':', ';'], " ")
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let head = tokens.first()?.as_str();
    if matches!(head, "yes" | "y") {
        return Some(true);
    }
    if matches!(head, "no" | "n") {
        return Some(false);
    }
    let has_yes = first_line.contains("yes");
    let has_no = first_line.contains("no");
    match (has_yes, has_no) {
        (true, false) => Some(true),
        (false, true) => Some(false),
        _ => None,
    }
}

/// Metric declaration for the optional official judge result.
pub fn longmemeval_judge_metric_specs() -> Vec<MetricSpec> {
    vec![MetricSpec {
        name: LONGMEMEVAL_JUDGE_METRIC.to_owned(),
        version: METRIC_VERSION.to_owned(),
        unit: "ratio".to_owned(),
        direction: MetricDirection::HigherIsBetter,
    }]
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LongMemEvalJudgeError {
    #[error("unsupported LongMemEval question type: {0:?}")]
    UnsupportedQuestionType(String),
}

/// The answer-only reference kept outside the generic retrieval sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongMemEvalAnswerReference {
    pub sample_id: String,
    pub answer: String,
}

/// Deterministic lexical scores for one predicted answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LongMemEvalAnswerScore {
    pub exact_match: f64,
    pub token_f1: f64,
}

/// Scores the answer produced for one LongMemEval question.
#[derive(Debug)]
pub struct LongMemEvalAnswerEvaluator {
    reference: LongMemEvalAnswerReference,
}

impl LongMemEvalAnswerEvaluator {
    pub fn new(reference: LongMemEvalAnswerReference) -> Result<Self, LongMemEvalAnswerError> {
        if reference.sample_id.trim().is_empty() {
            return Err(LongMemEvalAnswerError::InvalidReference(
                "sample_id must be non-empty".to_owned(),
            ));
        }
        if reference.answer.trim().is_empty() {
            return Err(LongMemEvalAnswerError::InvalidReference(
                "answer must be non-empty".to_owned(),
            ));
        }
        Ok(Self { reference })
    }

    pub fn metric_specs(&self) -> Vec<MetricSpec> {
        vec![
            MetricSpec {
                name: "longmemeval.answer_exact_match".to_owned(),
                version: METRIC_VERSION.to_owned(),
                unit: "ratio".to_owned(),
                direction: MetricDirection::HigherIsBetter,
            },
            MetricSpec {
                name: "longmemeval.answer_token_f1".to_owned(),
                version: METRIC_VERSION.to_owned(),
                unit: "ratio".to_owned(),
                direction: MetricDirection::HigherIsBetter,
            },
        ]
    }

    pub fn evaluate(&self, trace: &RunTrace) -> Result<Vec<MetricValue>, LongMemEvalAnswerError> {
        let mut matching_steps = trace
            .steps
            .iter()
            .filter(|step| step.sample_id == self.reference.sample_id);
        let step = matching_steps.next().ok_or_else(|| {
            LongMemEvalAnswerError::MissingSample(self.reference.sample_id.clone())
        })?;
        if matching_steps.next().is_some() {
            return Err(LongMemEvalAnswerError::DuplicateSample(
                self.reference.sample_id.clone(),
            ));
        }
        let score = score_answer(
            &self.reference.answer,
            step.agent_answer.as_deref().unwrap_or_default(),
        );
        Ok(vec![
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: "longmemeval.answer_exact_match".to_owned(),
                value: score.exact_match,
                sample_count: 1,
            },
            MetricValue {
                run_id: trace.run.run_id.clone(),
                name: "longmemeval.answer_token_f1".to_owned(),
                value: score.token_f1,
                sample_count: 1,
            },
        ])
    }
}

/// Computes normalized exact match and multiset token F1.
pub fn score_answer(reference: &str, prediction: &str) -> LongMemEvalAnswerScore {
    let reference_tokens = answer_tokens(reference);
    let prediction_tokens = answer_tokens(prediction);
    let exact_match = if reference_tokens == prediction_tokens {
        1.0
    } else {
        0.0
    };
    let token_f1 = if reference_tokens.is_empty() && prediction_tokens.is_empty() {
        1.0
    } else if reference_tokens.is_empty() || prediction_tokens.is_empty() {
        0.0
    } else {
        let mut reference_counts = HashMap::new();
        for token in &reference_tokens {
            *reference_counts.entry(token).or_insert(0_u64) += 1;
        }
        let mut overlap = 0_u64;
        let mut prediction_counts = HashMap::new();
        for token in &prediction_tokens {
            *prediction_counts.entry(token).or_insert(0_u64) += 1;
        }
        for (token, count) in prediction_counts {
            overlap += count.min(*reference_counts.get(&token).unwrap_or(&0));
        }
        let precision = overlap as f64 / prediction_tokens.len() as f64;
        let recall = overlap as f64 / reference_tokens.len() as f64;
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    };
    LongMemEvalAnswerScore {
        exact_match,
        token_f1,
    }
}

fn answer_tokens(value: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut ascii_word = String::new();
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            ascii_word.push(character.to_ascii_lowercase());
        } else {
            if !ascii_word.is_empty() {
                tokens.push(std::mem::take(&mut ascii_word));
            }
            if character.is_alphanumeric() {
                tokens.push(character.to_lowercase().collect());
            }
        }
    }
    if !ascii_word.is_empty() {
        tokens.push(ascii_word);
    }
    tokens
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LongMemEvalAnswerError {
    #[error("invalid LongMemEval answer reference: {0}")]
    InvalidReference(String),
    #[error("trace does not contain sample {0:?}")]
    MissingSample(String),
    #[error("trace contains duplicate sample {0:?}")]
    DuplicateSample(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_f1_ignores_case_and_punctuation() {
        let score = score_answer("The fox, chicken, and grain.", "the fox chicken and grain");
        assert_eq!(score.exact_match, 1.0);
        assert_eq!(score.token_f1, 1.0);
    }

    #[test]
    fn token_f1_handles_chinese_characters_and_partial_answers() {
        let score = score_answer("红色的船", "红色船");
        assert_eq!(score.exact_match, 0.0);
        assert!(score.token_f1 > 0.0 && score.token_f1 < 1.0);
    }

    #[test]
    fn empty_prediction_scores_zero_for_a_nonempty_reference() {
        let score = score_answer("answer", "");
        assert_eq!(score.exact_match, 0.0);
        assert_eq!(score.token_f1, 0.0);
    }

    #[test]
    fn reference_validation_rejects_empty_answer() {
        let error = LongMemEvalAnswerEvaluator::new(LongMemEvalAnswerReference {
            sample_id: "sample-1".to_owned(),
            answer: " ".to_owned(),
        })
        .unwrap_err();
        assert!(matches!(error, LongMemEvalAnswerError::InvalidReference(_)));
    }

    #[test]
    fn judge_parser_accepts_official_first_line_forms() {
        assert_eq!(parse_longmemeval_judge_response("Yes\nreason"), Some(true));
        assert_eq!(parse_longmemeval_judge_response("no."), Some(false));
        assert_eq!(
            parse_longmemeval_judge_response("The answer is yes."),
            Some(true)
        );
        assert_eq!(parse_longmemeval_judge_response("maybe"), None);
    }

    #[test]
    fn judge_prompt_uses_task_specific_rules() {
        let prompt =
            longmemeval_judge_prompt("temporal-reasoning", "When?", "Monday", "Monday", false)
                .unwrap();
        assert!(prompt.contains("off-by-one"));
        assert!(prompt.contains("Question: When?"));

        let abstention =
            longmemeval_judge_prompt("single-session-user", "Q", "A", "R", true).unwrap();
        assert!(abstention.contains("unanswerable question"));
    }

    #[test]
    fn judge_prompt_rejects_unknown_question_type() {
        let error = longmemeval_judge_prompt("future-task", "Q", "A", "R", false).unwrap_err();
        assert_eq!(
            error,
            LongMemEvalJudgeError::UnsupportedQuestionType("future-task".to_owned())
        );
        assert!(longmemeval_is_abstention_question("q_abs"));
        assert!(!longmemeval_is_abstention_question("q_1"));
        assert!(!longmemeval_is_abstention_question("q_abs_1"));
        assert!(!longmemeval_is_abstention_question("q_absolute_1"));
    }
}
