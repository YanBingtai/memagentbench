use uuid::Uuid;

use pi_agent_rust::{AgentLoop, Config, RuntimeReconcileOutcome, ToolRegistry};
use serde_json::Value;

use crate::{app_error::AppError, cli::Args};

pub(super) fn run(args: &Args, config: &Config, action_id: Uuid) -> Result<(), AppError> {
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

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn empty_evidence_is_rejected_before_opening_the_runtime_ledger() {
        let action_id = Uuid::new_v4();
        let args = Args::try_parse_from([
            "pi-agent-rust",
            "--reconcile-action-id",
            &action_id.to_string(),
            "--reconcile-evidence",
            "  ",
        ])
        .expect("arguments should parse");

        let error = run(&args, &Config::default(), action_id)
            .expect_err("blank evidence should be rejected");

        assert!(matches!(error, AppError::MissingReconcileEvidence));
    }

    #[test]
    fn malformed_success_output_is_rejected_before_writing() {
        let action_id = Uuid::new_v4();
        let args = Args::try_parse_from([
            "pi-agent-rust",
            "--reconcile-action-id",
            &action_id.to_string(),
            "--reconcile-evidence",
            "verified",
            "--reconcile-output",
            "not-json",
        ])
        .expect("arguments should parse");

        let error = run(&args, &Config::default(), action_id)
            .expect_err("malformed output should be rejected");

        assert!(matches!(error, AppError::InvalidReconcileOutput(_)));
    }
}
