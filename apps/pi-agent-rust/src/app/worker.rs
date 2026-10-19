use crate::app_error::AppError;
use pi_agent_rust::deltabox::run_deltabox_worker;

pub(crate) async fn run_worker(arguments: Vec<String>) -> Result<(), AppError> {
    let result = run_deltabox_worker(arguments)
        .await
        .map_err(|error| AppError::ShellSandbox {
            message: error.to_string(),
        })?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
