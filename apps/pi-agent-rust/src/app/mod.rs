mod deltabox;
mod reconcile;
mod worker;

use clap::Parser;

use crate::{
    agent_runner::run_agent,
    app_error::AppError,
    cli::Args,
    config_loader::{load_config, load_skills},
};

pub(super) use worker::run_worker;

pub(super) async fn run() -> Result<(), AppError> {
    let args = Args::parse();
    let config = load_config(&args)?;
    if let Some(action_id) = args.reconcile_action_id {
        return reconcile::run(&args, &config, action_id);
    }
    if let Some(action) = args.deltabox_action {
        return deltabox::run_action(&args, &config, action);
    }

    let prompt = args
        .prompt
        .clone()
        .or_else(|| (!args.prompt_words.is_empty()).then(|| args.prompt_words.join(" ")))
        .ok_or(AppError::MissingPrompt)?;
    let skills = load_skills(args.skills_dir.as_deref(), config.max_file_bytes)?;

    tokio::fs::create_dir_all(&config.workspace)
        .await
        .map_err(|source| AppError::Workspace {
            path: config.workspace.clone(),
            source,
        })?;
    let sandbox = deltabox::start_if_requested(&args, &config)?;
    let result = run_agent(&args, &config, prompt, skills, sandbox.as_ref()).await;
    if let Some(sandbox) = sandbox {
        deltabox::finish(sandbox, &args, &result).await?;
    }

    let result = result?;
    if let Some(content) = result.final_message.content {
        println!("{content}");
    } else {
        println!("{}", serde_json::to_string_pretty(&result.final_message)?);
    }
    Ok(())
}
