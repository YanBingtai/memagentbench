use super::super::*;

#[derive(Debug)]
pub(crate) struct SlowTool;

impl Tool for SlowTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "slow_tool",
            "A test tool that completes after a delay.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok::<String, ToolError>("done".to_string())
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
pub(crate) struct ParallelSleepTool {
    pub(in crate::agent_loop::tests) active: Arc<AtomicUsize>,
    pub(in crate::agent_loop::tests) max_active: Arc<AtomicUsize>,
}

#[derive(Debug)]
pub(crate) struct PanicParallelTool;

impl Tool for PanicParallelTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "panic_parallel",
            "A test tool that panics inside a parallel worker.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("unreachable".to_string()) })
    }

    fn execute_result_with_context<'a>(
        &'a self,
        _arguments: serde_json::Value,
        _context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(async { panic!("intentional parallel worker panic") })
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Safe
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

impl Tool for ParallelSleepTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "parallel_sleep",
            "A read-only tool used to verify parallel dispatch.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("done".to_string()) })
    }

    fn execute_result_with_context<'a>(
        &'a self,
        _arguments: serde_json::Value,
        _context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        let active = Arc::clone(&self.active);
        let max_active = Arc::clone(&self.max_active);
        Box::pin(async move {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            max_active.fetch_max(current, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(100)).await;
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(ToolResult::text("done"))
        })
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    fn replay_policy(&self) -> ToolReplayPolicy {
        ToolReplayPolicy::Safe
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
pub(crate) struct CancelInsideTool;

impl Tool for CancelInsideTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "cancel_inside_tool",
            "Cancel the caller while the tool is running.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("unused".to_string()) })
    }

    fn execute_result_with_context<'a>(
        &'a self,
        _arguments: serde_json::Value,
        context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(async move {
            context.cancellation_token().cancel();
            std::future::pending::<Result<ToolResult, ToolError>>().await
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
pub(crate) struct ToolTimeoutTool;

impl Tool for ToolTimeoutTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "tool_timeout_tool",
            "A test tool that exceeds the per-tool timeout.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async {
            tokio::time::sleep(Duration::from_millis(1_100)).await;
            Ok::<String, ToolError>("too late".to_string())
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
pub(crate) struct ImmediateTool;

impl Tool for ImmediateTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "echo_tool",
            "Return a stable test result.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("runtime-ok".to_string()) })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
pub(crate) struct UnknownRiskTool;

impl Tool for UnknownRiskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "unknown_risk_tool",
            "A tool whose risk class is intentionally unspecified.",
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("must not execute".to_string()) })
    }
}

#[derive(Debug)]
pub(crate) struct ErrorResultTool;

impl Tool for ErrorResultTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "error_tool",
            "Return a structured tool error.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("legacy error".to_string()) })
    }

    fn execute_result<'a>(&'a self, _arguments: serde_json::Value) -> ToolResultFuture<'a> {
        Box::pin(async {
            Ok(ToolResult::error("known failure").with_details(json!({"kind": "test"})))
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
pub(crate) struct UncertainResultTool;

impl Tool for UncertainResultTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "uncertain_tool",
            "Return a result whose external outcome is uncertain.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("uncertain".to_string()) })
    }

    fn execute_result<'a>(&'a self, _arguments: serde_json::Value) -> ToolResultFuture<'a> {
        Box::pin(async {
            Ok(ToolResult::error("command timed out").with_details(json!({"status": "timed_out"})))
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::Reversible
    }
}

#[derive(Debug)]
pub(crate) struct CountingValidationTool {
    pub(in crate::agent_loop::tests) validations: Arc<AtomicUsize>,
}

impl Tool for CountingValidationTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "counting_validation_tool",
            "A test tool that records argument validation calls.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn validate_arguments(&self, _arguments: &serde_json::Value) -> Result<(), ToolError> {
        self.validations.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("validated".to_string()) })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
pub(crate) struct HiddenModelTool {
    pub(in crate::agent_loop::tests) executions: Arc<AtomicUsize>,
}

impl Tool for HiddenModelTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "hidden_model_tool",
            "A trusted caller-only test tool.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Hidden
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        self.executions.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok::<String, ToolError>("should not execute".to_string()) })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
pub(crate) struct ProgressTool;

impl Tool for ProgressTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "progress_tool",
            "Emit one progress update before completing.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("legacy".to_string()) })
    }

    fn execute_result_with_context<'a>(
        &'a self,
        _arguments: serde_json::Value,
        context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        Box::pin(async move {
            context.update(ToolUpdate::text("halfway").with_details(json!({"progress": 0.5})));
            Ok(ToolResult::text("done"))
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
pub(crate) struct StructuredTool;

impl Tool for StructuredTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "structured_tool",
            "Return structured output and terminate the run.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("legacy".to_string()) })
    }

    fn execute_result<'a>(&'a self, _arguments: serde_json::Value) -> ToolResultFuture<'a> {
        Box::pin(async {
            Ok(ToolResult::text("structured")
                .with_details(json!({"source": "test"}))
                .terminating())
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}
