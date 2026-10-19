use super::super::api::ModelUsage;

use super::ModelClient;

impl ModelClient {
    pub(crate) fn clear_last_usage(&self) {
        *self
            .last_usage
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
    }

    pub(crate) fn store_last_usage(&self, usage: Option<ModelUsage>) {
        *self
            .last_usage
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = usage.filter(|value| !value.is_empty());
    }

    /// Take and clear usage from the most recent completed request.
    pub fn take_last_usage(&self) -> Option<ModelUsage> {
        self.last_usage
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
    }
}
