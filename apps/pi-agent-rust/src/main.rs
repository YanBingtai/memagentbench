use std::{process::ExitCode, time::Duration};

use clap::Parser;
use thiserror::Error;

use pi_agent_rust::model::ChatOptions;
use pi_agent_rust::{Message, ModelClient, ModelError};

#[derive(Debug, Parser)]
#[command(
    name = "pi-agent-rust",
    version,
    about = "A small, OpenAI-compatible Pi agent client"
)]
struct Args {
    /// OpenAI-compatible API base URL.
    #[arg(long, default_value = "http://127.0.0.1:6273/v1")]
    base_url: String,

    /// Model identifier sent to the provider.
    #[arg(long, default_value = "qwen3.8")]
    model: String,

    /// Environment variable containing an optional bearer token.
    #[arg(long, default_value = "OPENAI_API_KEY")]
    api_key_env: String,

    /// Request timeout in seconds.
    #[arg(long, default_value_t = 120)]
    timeout_secs: u64,

    /// Sampling temperature forwarded to the provider.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Explicitly enable thinking; this is already the default.
    #[arg(long, conflicts_with = "no_think")]
    think: bool,

    /// Disable thinking for providers that support this option.
    #[arg(long = "no-think", conflicts_with = "think")]
    no_think: bool,

    /// A prompt supplied as one argument.
    #[arg(long, value_name = "PROMPT", conflicts_with = "prompt_words")]
    prompt: Option<String>,

    /// Prompt words. Multiple words are joined with a single space.
    #[arg(value_name = "PROMPT", num_args = 0..)]
    prompt_words: Vec<String>,
}

impl Args {
    fn thinking_enabled(&self) -> bool {
        self.think || !self.no_think
    }
}

#[derive(Debug, Error)]
enum AppError {
    #[error(transparent)]
    Model(#[from] ModelError),

    #[error("failed to serialize assistant message: {0}")]
    Serialize(#[from] serde_json::Error),

    #[error("a prompt is required; pass --prompt or provide positional prompt words")]
    MissingPrompt,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), AppError> {
    let args = Args::parse();
    let client = ModelClient::new(&args.base_url, Duration::from_secs(args.timeout_secs))?;
    let api_key = std::env::var(&args.api_key_env).ok();
    let thinking_enabled = args.thinking_enabled();
    let prompt = args
        .prompt
        .or_else(|| (!args.prompt_words.is_empty()).then(|| args.prompt_words.join(" ")))
        .ok_or(AppError::MissingPrompt)?;
    let response = client
        .complete_with_options(
            &args.model,
            api_key.as_deref(),
            &[Message::user(prompt)],
            &[],
            ChatOptions {
                temperature: args.temperature,
                enable_thinking: Some(thinking_enabled),
            },
        )
        .await?;

    if let Some(content) = response.content {
        println!("{content}");
    } else {
        println!("{}", serde_json::to_string_pretty(&response)?);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_is_enabled_by_default() {
        let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
            .expect("arguments should parse");

        assert!(args.thinking_enabled());
    }

    #[test]
    fn think_flag_can_be_enabled() {
        let args = Args::try_parse_from(["pi-agent-rust", "--think", "--prompt", "hello"])
            .expect("arguments should parse");

        assert!(args.thinking_enabled());
    }

    #[test]
    fn think_mode_can_be_disabled() {
        let args = Args::try_parse_from(["pi-agent-rust", "--no-think", "--prompt", "hello"])
            .expect("arguments should parse");

        assert!(!args.thinking_enabled());
    }
}
