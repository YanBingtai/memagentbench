use super::limits::{
    truncate_output, MAX_TOOL_ARGUMENT_BYTES, MAX_TOOL_ARGUMENT_DEPTH, MAX_WEB_RESPONSE_BYTES,
};
use super::*;
use crate::agent_loop::CancellationToken;
use crate::protocol::{FunctionCall, ToolCall, ToolDefinition};
use crate::skills::{Skill, SkillCatalog};
use pi_agent_runtime::{RiskClass, SandboxMode};
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tempfile::tempdir;
use wiremock::{
    matchers::{header, method, path, query_param},
    Mock, MockServer, ResponseTemplate,
};

fn read_call(path: &str) -> ToolCall {
    ToolCall {
        id: "call-1".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "read_file".to_string(),
            arguments: serde_json::json!({ "path": path }).to_string(),
        },
    }
}

fn list_call(arguments: Value) -> ToolCall {
    ToolCall {
        id: "list-call-1".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "list_files".to_string(),
            arguments: arguments.to_string(),
        },
    }
}

fn search_call(arguments: Value) -> ToolCall {
    ToolCall {
        id: "search-call-1".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "search_files".to_string(),
            arguments: arguments.to_string(),
        },
    }
}

fn shell_call(arguments: Value) -> ToolCall {
    ToolCall {
        id: "shell-call-1".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "bash".to_string(),
            arguments: arguments.to_string(),
        },
    }
}

fn write_call(path: &str, content: &str) -> ToolCall {
    ToolCall {
        id: "write-call-1".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "write_file".to_string(),
            arguments: serde_json::json!({
                "path": path,
                "content": content,
            })
            .to_string(),
        },
    }
}

fn edit_call(path: &str, old_text: &str, new_text: &str) -> ToolCall {
    ToolCall {
        id: "edit-call-1".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "edit_file".to_string(),
            arguments: serde_json::json!({
                "path": path,
                "old_text": old_text,
                "new_text": new_text,
            })
            .to_string(),
        },
    }
}

fn skill_call(name: &str) -> ToolCall {
    ToolCall {
        id: "skill-call-1".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "read_skill".to_string(),
            arguments: serde_json::json!({ "name": name }).to_string(),
        },
    }
}

#[derive(Debug)]
struct SchemaTool;

impl Tool for SchemaTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "schema_tool",
            "A custom tool whose schema is enforced by the registry.",
            json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "minLength": 1
                    }
                },
                "required": ["message"],
                "additionalProperties": false
            }),
        )
    }

    fn execute<'a>(&'a self, arguments: Value) -> ToolFuture<'a> {
        Box::pin(async move {
            Ok(arguments["message"]
                .as_str()
                .expect("schema validation should ensure message is a string")
                .to_string())
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
struct InvalidSchemaTool;

impl Tool for InvalidSchemaTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "invalid_schema_tool",
            "A tool with an unsupported parameter schema.",
            json!({"type": "string"}),
        )
    }

    fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
        Box::pin(async { Ok("unused".to_string()) })
    }
}

#[derive(Debug)]
struct InvalidMetadataTool;

impl Tool for InvalidMetadataTool {
    fn definition(&self) -> ToolDefinition {
        let mut definition = ToolDefinition::function(
            "bad.name",
            "Invalid metadata fixture.",
            json!({"type": "object"}),
        );
        definition.definition_type = "custom".to_string();
        definition
    }

    fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
        Box::pin(async { Ok("unused".to_string()) })
    }
}

#[derive(Debug)]
struct ChangingDefinitionTool {
    definition_calls: Arc<AtomicUsize>,
}

impl Tool for ChangingDefinitionTool {
    fn definition(&self) -> ToolDefinition {
        let call = self.definition_calls.fetch_add(1, Ordering::SeqCst);
        if call == 0 {
            ToolDefinition::function(
                "changing_tool",
                "A stable registration snapshot fixture.",
                json!({
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                    "additionalProperties": false
                }),
            )
        } else {
            ToolDefinition::function(
                "changing_tool",
                "A changed definition that must not escape the registry.",
                json!({"type": "object", "properties": {}, "additionalProperties": false}),
            )
        }
    }

    fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
        Box::pin(async { Ok("changed".to_string()) })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
struct LargeResultTool;

impl Tool for LargeResultTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "large_result",
            "Return an intentionally large structured result.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("unused".to_string()) })
    }

    fn execute_result<'a>(&'a self, _arguments: Value) -> ToolResultFuture<'a> {
        Box::pin(async {
            Ok(ToolResult {
                content: "你好".repeat(10_000),
                details: json!({"payload": "x".repeat(10_000)}),
                is_error: true,
                terminate: true,
            })
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

#[derive(Debug)]
struct LargeUnknownResultTool;

impl Tool for LargeUnknownResultTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "large_unknown_result",
            "Return an uncertain result with oversized details.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("unused".to_string()) })
    }

    fn execute_result<'a>(&'a self, _arguments: Value) -> ToolResultFuture<'a> {
        Box::pin(async {
            Ok(
                ToolResult::error("external outcome is uncertain").with_details(json!({
                    "status": "unknown",
                    "payload": "x".repeat(10_000),
                })),
            )
        })
    }

    fn risk_class(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
}

struct RetainedContextTool {
    retained: Arc<Mutex<Option<ToolExecutionContext>>>,
}

impl Tool for RetainedContextTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "retained_context",
            "Retain a tool context to verify terminal update behavior.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
        Box::pin(async { Ok::<String, ToolError>("done".to_string()) })
    }

    fn execute_result_with_context<'a>(
        &'a self,
        _arguments: Value,
        context: ToolExecutionContext,
    ) -> ToolResultFuture<'a> {
        let retained = Arc::clone(&self.retained);
        Box::pin(async move {
            context.update(ToolUpdate::text("during"));
            retained
                .lock()
                .expect("retained context mutex should not be poisoned")
                .replace(context.clone());
            Ok(ToolResult::text("done"))
        })
    }
}

#[derive(Debug)]
struct HiddenTool;

impl Tool for HiddenTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "hidden_tool",
            "A trusted caller-only tool.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        )
    }

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Hidden
    }

    fn execute<'a>(&'a self, _arguments: Value) -> ToolFuture<'a> {
        Box::pin(async { Ok("hidden".to_string()) })
    }
}

mod contracts;
mod filesystem;
mod limits;
mod registry;
mod shell;
mod skills;
mod web_search;
