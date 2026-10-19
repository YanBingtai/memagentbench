//! Runtime statistics collected during one agent loop.

use crate::model::ModelUsage;

#[derive(Debug, Clone, Copy)]
pub(crate) struct RunStats {
    pub(in crate::agent_loop) steps: usize,
    pub(in crate::agent_loop) model_retries: usize,
    pub(in crate::agent_loop) has_unknown_action: bool,
    model_requests: usize,
    input_usage_complete: bool,
    output_usage_complete: bool,
    total_usage_complete: bool,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

impl Default for RunStats {
    fn default() -> Self {
        Self {
            steps: 0,
            model_retries: 0,
            has_unknown_action: false,
            model_requests: 0,
            input_usage_complete: true,
            output_usage_complete: true,
            total_usage_complete: true,
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
        }
    }
}

impl RunStats {
    pub(crate) fn record_model_usage(&mut self, usage: Option<ModelUsage>) {
        self.model_requests = self.model_requests.saturating_add(1);
        let Some(usage) = usage else {
            self.input_usage_complete = false;
            self.output_usage_complete = false;
            self.total_usage_complete = false;
            return;
        };

        record_token(
            &mut self.input_tokens,
            &mut self.input_usage_complete,
            usage.input_tokens,
        );
        record_token(
            &mut self.output_tokens,
            &mut self.output_usage_complete,
            usage.output_tokens,
        );
        record_token(
            &mut self.total_tokens,
            &mut self.total_usage_complete,
            usage.total_tokens,
        );
    }

    pub(crate) fn record_retry(&mut self) {
        self.model_retries = self.model_retries.saturating_add(1);
    }

    pub(crate) fn model_usage(&self) -> (Option<u64>, Option<u64>, Option<u64>) {
        if self.model_requests == 0 {
            return (None, None, None);
        }
        (
            self.input_tokens.filter(|_| self.input_usage_complete),
            self.output_tokens.filter(|_| self.output_usage_complete),
            self.total_tokens.filter(|_| self.total_usage_complete),
        )
    }
}

fn record_token(slot: &mut Option<u64>, complete: &mut bool, value: Option<u64>) {
    let Some(value) = value else {
        *complete = false;
        return;
    };
    *slot = Some(slot.unwrap_or_default().saturating_add(value));
}
