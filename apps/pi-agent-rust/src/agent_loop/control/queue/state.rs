use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};

#[derive(Debug, Default)]
pub(in crate::agent_loop) struct ControlState {
    pub(super) steering: VecDeque<String>,
    pub(super) follow_up: VecDeque<String>,
    pub(super) active: bool,
    pub(super) has_run: bool,
}

pub(in crate::agent_loop) struct RunGuard {
    control: Arc<StdMutex<ControlState>>,
}

impl RunGuard {
    pub(super) fn new(control: Arc<StdMutex<ControlState>>) -> Self {
        Self { control }
    }
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        let mut control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        control.steering.clear();
        control.follow_up.clear();
        control.active = false;
    }
}
