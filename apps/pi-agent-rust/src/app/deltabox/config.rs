use pi_agent_runtime::DeltaBoxConfig;
use pi_agent_rust::Config;

use crate::{app_error::AppError, cli::Args};

pub(super) fn deltabox_config(args: &Args, config: &Config) -> Result<DeltaBoxConfig, AppError> {
    let mut sandbox = match &args.deltabox_bin {
        Some(binary) => DeltaBoxConfig::from_binary(binary, &config.workspace),
        None => DeltaBoxConfig::discover(&config.workspace).map_err(super::shell_error)?,
    };
    sandbox.fs_mode = args.deltabox_fs_mode.into();
    sandbox.require_fs_mode = args.deltabox_require_fs_mode;
    Ok(sandbox)
}
