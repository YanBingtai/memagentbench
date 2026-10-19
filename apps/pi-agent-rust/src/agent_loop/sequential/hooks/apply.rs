use crate::model::ChatModel;

use super::super::super::tool_batch::{await_tool_hook, ToolHookWait};
use super::super::super::{
    timeout_error, timeout_millis, AgentError, AgentLoop, ToolResultContext,
};
use super::request::{AfterToolHookOutcome, AfterToolHookRequest};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(in crate::agent_loop::sequential) async fn apply_after_tool_hook(
        &self,
        request: AfterToolHookRequest<'_>,
    ) -> AfterToolHookOutcome {
        let AfterToolHookRequest {
            context,
            step,
            tool_call,
            arguments,
            tool_result,
            deadline,
            cancellation,
            skip,
        } = request;
        let original_content = tool_result.content.clone();
        if skip {
            return AfterToolHookOutcome {
                content: original_content,
                is_error: tool_result.is_error,
                stop: None,
            };
        }

        let Some(hook) = &self.after_tool_hook else {
            return AfterToolHookOutcome {
                content: original_content,
                is_error: tool_result.is_error,
                stop: None,
            };
        };

        let hook_result = await_tool_hook(
            hook(ToolResultContext {
                run: context,
                step,
                call: tool_call.clone(),
                arguments: arguments.clone(),
                content: tool_result.content.clone(),
                is_error: tool_result.is_error,
                details: tool_result.details.clone(),
                terminate: tool_result.terminate,
            }),
            deadline,
            cancellation,
        )
        .await;

        match hook_result {
            ToolHookWait::Completed(Ok(override_result)) => {
                if let Some(details) = override_result.details {
                    tool_result.details = details;
                }
                if let Some(terminate) = override_result.terminate {
                    tool_result.terminate = terminate;
                }
                AfterToolHookOutcome {
                    content: override_result.content.unwrap_or(original_content),
                    is_error: override_result.is_error.unwrap_or(tool_result.is_error),
                    stop: None,
                }
            }
            ToolHookWait::Completed(Err(error)) => AfterToolHookOutcome {
                content: format!("after_tool_call hook failed: {error}"),
                is_error: true,
                stop: None,
            },
            ToolHookWait::TimedOut(run_timeout) => {
                let reason = format!(
                    "after_tool_call hook timed out after {} milliseconds",
                    timeout_millis(run_timeout)
                );
                AfterToolHookOutcome {
                    content: reason.clone(),
                    is_error: true,
                    stop: Some((timeout_error(run_timeout), reason)),
                }
            }
            ToolHookWait::Cancelled => {
                let reason = "after_tool_call hook cancelled by the caller".to_string();
                AfterToolHookOutcome {
                    content: reason.clone(),
                    is_error: true,
                    stop: Some((AgentError::RunCancelled, reason)),
                }
            }
        }
    }
}
