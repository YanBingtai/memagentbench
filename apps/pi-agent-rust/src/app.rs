use std::time::Duration;

use clap::Parser;
use pi_agent_runtime::{DeltaBoxClient, DeltaBoxConfig};
use serde_json::Value;
use uuid::Uuid;

use crate::{
    agent_runner::run_agent,
    app_error::AppError,
    cli::{Args, DeltaBoxActionArg, SandboxModeArg},
    config_loader::{load_config, load_skills},
};
use pi_agent_rust::deltabox::{run_deltabox_worker, DeltaBoxDisposition, DeltaBoxSandbox};
use pi_agent_rust::{AgentLoop, Config, RuntimeReconcileOutcome, ToolRegistry};

pub(super) async fn run_worker(arguments: Vec<String>) -> Result<(), AppError> {
    let result = run_deltabox_worker(arguments)
        .await
        .map_err(|error| AppError::ShellSandbox {
            message: error.to_string(),
        })?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

pub(super) async fn run() -> Result<(), AppError> {
    let args = Args::parse();
    let config = load_config(&args)?;
    if let Some(action_id) = args.reconcile_action_id {
        return reconcile_runtime_action(&args, &config, action_id);
    }
    if let Some(action) = args.deltabox_action {
        let client = DeltaBoxClient::attach(
            deltabox_config(&args, &config)?,
            args.deltabox_session
                .as_deref()
                .expect("clap requires a session ID"),
        )
        .map_err(|error| AppError::ShellSandbox {
            message: error.to_string(),
        })?;
        let receipt = match action {
            DeltaBoxActionArg::Status => client.status(),
            DeltaBoxActionArg::Diff => client.diff(),
            DeltaBoxActionArg::Commit => client.commit(),
            DeltaBoxActionArg::Abort => client.abort(),
            DeltaBoxActionArg::Stop => client.stop(),
        }
        .map_err(|error| AppError::ShellSandbox {
            message: error.to_string(),
        })?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        return Ok(());
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
    let sandbox = if args.sandbox_mode == SandboxModeArg::Deltabox {
        let sandbox = DeltaBoxSandbox::start(
            deltabox_config(&args, &config)?,
            Duration::from_secs(config.tool_timeout_secs),
        )
        .map_err(|error| AppError::ShellSandbox {
            message: error.to_string(),
        })?;
        eprintln!(
            "[deltabox] session {}: {:?}, visibility={}, provider={}",
            sandbox.client().session_id(),
            sandbox.client().info().resolved_fs_mode,
            sandbox.client().info().host_visibility,
            sandbox.client().info().isolation_provider
        );
        if let Some(reason) = &sandbox.client().info().fallback_reason {
            eprintln!("[deltabox] filesystem fallback: {reason}");
        }
        if !sandbox.client().info().degraded_protections.is_empty() {
            eprintln!(
                "[deltabox] unavailable protections: {:?}",
                sandbox.client().info().degraded_protections
            );
        }
        Some(sandbox)
    } else {
        None
    };
    let result = run_agent(&args, &config, prompt, skills, sandbox.as_ref()).await;
    if let Some(sandbox) = sandbox {
        let disposition = if result.is_ok() {
            args.deltabox_on_success.into()
        } else {
            DeltaBoxDisposition::Abort
        };
        let session_id = sandbox.client().session_id().to_string();
        let finish = tokio::task::spawn_blocking(move || sandbox.finish(disposition))
            .await
            .map_err(|error| AppError::ShellSandbox {
                message: error.to_string(),
            })?;
        let receipt = finish.map_err(|error| AppError::ShellSandbox {
            message: format!(
                "session {session_id} needs recovery: {error}; agent result: {:?}",
                result.as_ref().err()
            ),
        })?;
        eprintln!(
            "[deltabox] {disposition:?} session {session_id}: {}",
            serde_json::to_string(&receipt)?
        );
        if receipt
            .get("folded_publications")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            > 0
        {
            eprintln!(
                "[deltabox] sibling changes were included; re-run project checks after commit"
            );
        }
    }
    let result = result?;
    if let Some(content) = result.final_message.content {
        println!("{content}");
    } else {
        println!("{}", serde_json::to_string_pretty(&result.final_message)?);
    }
    Ok(())
}

fn deltabox_config(args: &Args, config: &Config) -> Result<DeltaBoxConfig, AppError> {
    let mut sandbox = match &args.deltabox_bin {
        Some(binary) => DeltaBoxConfig::from_binary(binary, &config.workspace),
        None => {
            DeltaBoxConfig::discover(&config.workspace).map_err(|error| AppError::ShellSandbox {
                message: error.to_string(),
            })?
        }
    };
    sandbox.fs_mode = args.deltabox_fs_mode.into();
    sandbox.require_fs_mode = args.deltabox_require_fs_mode;
    Ok(sandbox)
}

fn reconcile_runtime_action(args: &Args, config: &Config, action_id: Uuid) -> Result<(), AppError> {
    let evidence = args
        .reconcile_evidence
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or(AppError::MissingReconcileEvidence)?
        .to_string();
    let outcome = if let Some(error) = &args.reconcile_error {
        RuntimeReconcileOutcome::Failed {
            error: error.clone(),
            evidence,
        }
    } else {
        let output = match &args.reconcile_output {
            Some(raw) => serde_json::from_str(raw).map_err(AppError::InvalidReconcileOutput)?,
            None => Value::Null,
        };
        RuntimeReconcileOutcome::Succeeded { output, evidence }
    };
    let agent = AgentLoop::from_config(config, ToolRegistry::new())?
        .with_runtime_ledger_file(config.session_file.with_extension("runtime.jsonl"));
    agent.reconcile_runtime_action(action_id, outcome)?;
    println!("reconciled runtime action {action_id}");
    Ok(())
}
