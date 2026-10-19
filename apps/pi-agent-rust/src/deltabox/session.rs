use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::Duration,
};

use pi_agent_runtime::DeltaBoxClient;
use tempfile::TempDir;

#[derive(Debug)]
pub(super) struct DeltaBoxSession {
    pub(super) client: DeltaBoxClient,
    pub(super) worker: PathBuf,
    pub(super) control_dir: Mutex<Option<TempDir>>,
    pub(super) timeout: Duration,
    pub(super) finished: AtomicBool,
}

impl DeltaBoxSession {
    pub(super) fn preserve_policy(&self) {
        if let Some(directory) = self
            .control_dir
            .lock()
            .expect("control directory lock")
            .take()
        {
            let _ = directory.keep();
        }
    }
}

impl Drop for DeltaBoxSession {
    fn drop(&mut self) {
        if self.finished.load(Ordering::Acquire) {
            return;
        }
        if let Err(error) = self.client.abort().and_then(|_| self.client.stop()) {
            self.preserve_policy();
            eprintln!(
                "[deltabox] session {} needs recovery: {error}",
                self.client.session_id()
            );
        }
    }
}
