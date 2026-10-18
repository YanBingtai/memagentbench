use std::path::Path;

use pi_agent_rust::{Config, SkillCatalog, SkillLoader};

use crate::{app_error::AppError, cli::Args};

pub(super) fn load_config(args: &Args) -> Result<Config, AppError> {
    let thinking_override = args.thinking_override();
    let has_config_file = args.config.is_some();
    let mut config = match args.config.as_deref() {
        Some(path) => Config::load(path)?,
        None => Config::default(),
    };
    if !has_config_file {
        if args.base_url.is_none() {
            config.base_url = "http://127.0.0.1:6273/v1".to_string();
        }
        if args.model.is_none() {
            config.model = "qwen3.8".to_string();
        }
        if args.temperature.is_none() {
            config.temperature = 0.7;
        }
    }
    if let Some(value) = &args.base_url {
        config.base_url = value.clone();
    }
    if let Some(value) = &args.model {
        config.model = value.clone();
    }
    if let Some(value) = &args.api_key_env {
        config.api_key_env = (!value.is_empty()).then_some(value.clone());
    }
    if let Some(value) = &args.workspace {
        config.workspace = value.clone();
    }
    if let Some(value) = &args.session_file {
        config.session_file = value.clone();
    }
    if let Some(value) = args.max_steps {
        config.max_steps = value;
    }
    if let Some(value) = args.run_timeout_secs {
        config.run_timeout_secs = Some(value);
    }
    if let Some(value) = args.temperature {
        config.temperature = value;
    }
    if let Some(value) = &args.system_prompt {
        config.system_prompt = (!value.trim().is_empty()).then_some(value.clone());
    }
    if let Some(value) = args.timeout_secs {
        config.request_timeout_secs = value;
    }
    if let Some(value) = thinking_override {
        config.enable_thinking = Some(value);
    }
    Ok(config)
}

pub(super) fn load_skills(
    root: Option<&Path>,
    max_file_bytes: usize,
) -> Result<Option<SkillCatalog>, AppError> {
    let Some(root) = root else {
        return Ok(None);
    };

    let report = SkillLoader::new(root)
        .with_max_file_bytes(max_file_bytes as u64)
        .load()?;
    for diagnostic in report.diagnostics() {
        eprintln!(
            "warning: skill {}: {}",
            diagnostic.path.display(),
            diagnostic.error
        );
    }

    Ok((!report.catalog().is_empty()).then_some(report.catalog))
}
