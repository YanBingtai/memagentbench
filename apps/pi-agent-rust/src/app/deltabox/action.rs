use pi_agent_runtime::DeltaBoxClient;
use pi_agent_rust::Config;

use crate::{
    app_error::AppError,
    cli::{Args, DeltaBoxActionArg},
};

use super::config::deltabox_config;

pub(crate) fn run_action(
    args: &Args,
    config: &Config,
    action: DeltaBoxActionArg,
) -> Result<(), AppError> {
    let client = DeltaBoxClient::attach(
        deltabox_config(args, config)?,
        args.deltabox_session
            .as_deref()
            .expect("clap requires a session ID"),
    )
    .map_err(super::shell_error)?;
    let receipt = match action {
        DeltaBoxActionArg::Status => client.status(),
        DeltaBoxActionArg::Diff => client.diff(),
        DeltaBoxActionArg::Commit => client.commit(),
        DeltaBoxActionArg::Abort => client.abort(),
        DeltaBoxActionArg::Stop => client.stop(),
    }
    .map_err(super::shell_error)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
