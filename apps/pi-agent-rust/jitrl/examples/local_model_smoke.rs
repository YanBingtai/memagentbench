//! Exercise the independent JitRL scorer with an OpenAI-compatible local model.
//!
//! The model produces an action-level prior (a ranked JSON list). JitRL then
//! scores the same actions from historical trajectories. This example keeps
//! those signals separate: the model is not treated as a token-logit source,
//! and the deterministic JitRL score remains the recommendation authority.

use std::{collections::BTreeSet, env, error::Error, time::Duration};

use pi_agent_jitrl::{ActionCandidate, JitRlConfig, JitRlEngine, Trajectory, Transition};
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::json;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:6273/v1";
const DEFAULT_MODEL: &str = "qwen3.8";

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ActionPriorEnvelope {
    state_embedding: Vec<f32>,
    candidates: Vec<ActionPrior>,
}

#[derive(Debug, Deserialize)]
struct ActionPrior {
    action: String,
    prior: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let base_url =
        env::var("JITRL_MODEL_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let model = env::var("JITRL_MODEL_NAME").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

    let model_prior = request_action_prior(&base_url, &model).await?;
    if model_prior.state_embedding.len() != 2
        || model_prior
            .state_embedding
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err("model state_embedding must contain two finite values".into());
    }
    if model_prior
        .state_embedding
        .iter()
        .all(|value| value.abs() <= f32::EPSILON)
    {
        return Err("model state_embedding must not be the zero vector".into());
    }
    println!(
        "model_state_vector=[{:.3}, {:.3}]",
        model_prior.state_embedding[0], model_prior.state_embedding[1]
    );
    println!("model_action_prior:");
    for candidate in &model_prior.candidates {
        println!("  {}: {:.3}", candidate.action, candidate.prior);
    }

    let known_actions = ["inspect_file", "apply_patch", "run_tests"];
    let mut candidates = model_prior
        .candidates
        .iter()
        .filter(|candidate| known_actions.contains(&candidate.action.as_str()))
        .map(|candidate| ActionCandidate {
            action: candidate.action.clone(),
            // The model supplies an action-level prior. JitRL adds a normalized
            // historical advantage to this value below.
            base_logit: candidate.prior,
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        eprintln!("model returned no known actions; scoring the complete fixture action set");
        candidates = known_actions
            .into_iter()
            .map(|action| ActionCandidate {
                action: action.to_string(),
                base_logit: 0.0,
            })
            .collect();
    }

    let mut engine = JitRlEngine::new(
        8,
        JitRlConfig {
            gamma: 0.95,
            top_k: 8,
            min_similarity: 0.2,
            advantage_scale: 1.0,
            min_correction_advantage: 0.1,
            min_correction_confidence: 0.5,
        },
    )?;
    for (action, reward) in [
        ("inspect_file", 0.2),
        ("apply_patch", 1.0),
        ("run_tests", 0.7),
    ] {
        let transition =
            Transition::new(model_prior.state_embedding.clone(), action, reward, true, 0)?;
        engine.record(Trajectory::new(
            vec![transition],
            Some("local-smoke".to_string()),
        )?)?;
    }

    let recommendation = engine.adjust(&model_prior.state_embedding, &candidates)?;
    println!(
        "jitrl_recommendation={} base_logit={:.3} adjusted_logit={:.3} expected_return={:.3} confidence={:.3}",
        recommendation.selected.action,
        recommendation.selected.base_logit,
        recommendation.selected.adjusted_logit,
        recommendation.selected.expected_return,
        recommendation.confidence
    );
    println!("signals=model_state_and_action_prior;jitrl=historical_advantage;token_logits=false");
    Ok(())
}

async fn request_action_prior(
    base_url: &str,
    model: &str,
) -> Result<ActionPriorEnvelope, Box<dyn Error>> {
    let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = json!({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return JSON only. The JSON shape is {\"state_embedding\":[number,number],\"candidates\":[{\"action\":string,\"prior\":number}]}. Use a finite 2D task-state vector. Rank these actions for the task: inspect_file, apply_patch, run_tests. prior must be between 0 and 1. Do not include markdown."
            },
            {
                "role": "user",
                "content": "Task: fix a code bug, inspect the relevant file, apply the smallest safe patch, and run the tests."
            }
        ],
        "temperature": 0.0,
        "stream": false,
        "chat_template_kwargs": {"enable_thinking": false}
    });

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?
        .post(endpoint)
        .json(&body)
        .send()
        .await?;
    let status = response.status();
    let raw = response.text().await?;
    if status != StatusCode::OK {
        return Err(format!("model returned HTTP {status}: {raw}").into());
    }

    let response: ChatResponse = serde_json::from_str(&raw)?;
    let content = response
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or("model response did not contain assistant content")?;
    parse_action_prior(&content)
}

fn parse_action_prior(content: &str) -> Result<ActionPriorEnvelope, Box<dyn Error>> {
    let content = content.trim();
    let content = content
        .strip_prefix("<think>")
        .and_then(|value| value.split_once("</think>").map(|(_, rest)| rest))
        .map(str::trim)
        .unwrap_or(content);
    let content = content
        .strip_prefix("```json")
        .and_then(|value| value.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(content);
    let start = content
        .find('{')
        .ok_or("model content has no JSON object")?;
    let end = content
        .rfind('}')
        .ok_or("model content has no JSON terminator")?;
    let envelope: ActionPriorEnvelope = serde_json::from_str(&content[start..=end])?;
    if envelope.state_embedding.len() != 2 {
        return Err("model state_embedding must contain two values".into());
    }
    if envelope.candidates.is_empty() {
        return Err("model returned an empty action prior".into());
    }
    let mut actions = BTreeSet::new();
    for candidate in &envelope.candidates {
        if candidate.action.trim().is_empty()
            || !candidate.prior.is_finite()
            || !(0.0..=1.0).contains(&candidate.prior)
        {
            return Err("model returned an invalid action prior".into());
        }
        if !actions.insert(candidate.action.as_str()) {
            return Err("model returned duplicate action candidates".into());
        }
    }
    Ok(envelope)
}
