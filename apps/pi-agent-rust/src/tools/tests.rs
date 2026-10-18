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

#[test]
fn execution_context_reports_cancellation_and_updates() {
    let cancellation = CancellationToken::new();
    let updates = Arc::new(Mutex::new(Vec::new()));
    let context = ToolExecutionContext::new(cancellation.clone()).with_update_sink({
        let updates = Arc::clone(&updates);
        move |update| {
            updates
                .lock()
                .expect("update mutex should not poison")
                .push(update)
        }
    });

    context.update(ToolUpdate::text("started"));
    assert!(!context.is_cancelled());
    let child_token = context.cancellation_token();
    cancellation.cancel();
    assert!(context.is_cancelled());
    assert!(child_token.is_cancelled());
    assert_eq!(
        updates
            .lock()
            .expect("update mutex should not poison")
            .len(),
        1
    );
}

#[test]
fn execution_context_bounds_progress_updates() {
    let updates = Arc::new(Mutex::new(Vec::new()));
    let context = ToolExecutionContext::new(CancellationToken::new())
        .with_output_limit(32)
        .with_update_sink({
            let updates = Arc::clone(&updates);
            move |update| {
                updates
                    .lock()
                    .expect("update mutex should not poison")
                    .push(update)
            }
        });

    context.update(ToolUpdate::text("你好".repeat(32)).with_details(json!({
        "payload": "x".repeat(128)
    })));

    let updates = updates.lock().expect("update mutex should not poison");
    assert_eq!(updates.len(), 1);
    assert!(updates[0].content.len() <= 32);
    assert!(updates[0].details.is_null());
}

#[tokio::test]
async fn registry_closes_progress_updates_after_tool_completion() {
    let retained = Arc::new(Mutex::new(None));
    let updates = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry
        .register(RetainedContextTool {
            retained: Arc::clone(&retained),
        })
        .expect("retained context tool should register");
    let call = ToolCall {
        id: "retained-call".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "retained_context".to_string(),
            arguments: "{}".to_string(),
        },
    };
    let context = ToolExecutionContext::new(CancellationToken::new()).with_update_sink({
        let updates = Arc::clone(&updates);
        move |update| {
            updates
                .lock()
                .expect("updates mutex should not be poisoned")
                .push(update.content);
        }
    });

    registry
        .execute_result_with_context(&call, context)
        .await
        .expect("tool should complete");
    retained
        .lock()
        .expect("retained context mutex should not be poisoned")
        .as_ref()
        .expect("tool should retain its context")
        .update(ToolUpdate::text("late"));

    assert_eq!(
        *updates
            .lock()
            .expect("updates mutex should not be poisoned"),
        vec!["during"]
    );
}

#[tokio::test]
async fn registry_bounds_custom_tool_results() {
    let mut registry = ToolRegistry::new().with_output_limit(4_096);
    registry
        .register(LargeResultTool)
        .expect("large result tool should register");
    let call = ToolCall {
        id: "large-call".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "large_result".to_string(),
            arguments: "{}".to_string(),
        },
    };

    let result = registry
        .execute_result(&call)
        .await
        .expect("large result tool should execute");

    assert!(result.content.len() <= 4_096);
    assert_eq!(result.details["status"], "details_truncated");
    assert!(result.is_error);
    assert!(result.terminate);
}

#[tokio::test]
async fn registry_preserves_uncertain_status_when_bounding_details() {
    let mut registry = ToolRegistry::new().with_output_limit(256);
    registry
        .register(LargeUnknownResultTool)
        .expect("uncertain result tool should register");
    let call = ToolCall {
        id: "unknown-call".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "large_unknown_result".to_string(),
            arguments: "{}".to_string(),
        },
    };

    let result = registry
        .execute_result(&call)
        .await
        .expect("uncertain result tool should execute");

    assert_eq!(result.details["status"], "unknown");
    assert_eq!(result.details["details_truncated"], true);
}

#[test]
fn tool_result_bounding_preserves_reconciled_status() {
    let result = ToolResult::text("confirmed")
        .with_details(json!({
            "status": "reconciled",
            "evidence": "x".repeat(1_000),
        }))
        .bounded(64);

    assert_eq!(result.details["status"], "reconciled");
}

#[tokio::test]
async fn custom_tool_schema_is_enforced_before_execution() {
    let mut registry = ToolRegistry::new();
    registry
        .register(SchemaTool)
        .expect("valid custom schema should register");

    let mut call = ToolCall {
        id: "schema-call".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "schema_tool".to_string(),
            arguments: "{}".to_string(),
        },
    };
    let missing = registry
        .execute(&call)
        .await
        .expect_err("required field should be enforced");
    assert!(matches!(
        missing,
        ToolError::SchemaViolation { path, .. } if path == "arguments.message"
    ));

    call.function.arguments = json!({"message": 7}).to_string();
    let wrong_type = registry
        .execute(&call)
        .await
        .expect_err("property type should be enforced");
    assert!(matches!(
        wrong_type,
        ToolError::SchemaViolation { path, .. } if path == "arguments.message"
    ));

    call.function.arguments = json!({"message": "ok", "extra": true}).to_string();
    let unknown = registry
        .execute(&call)
        .await
        .expect_err("additional properties should be rejected");
    assert!(matches!(
        unknown,
        ToolError::SchemaViolation { path, .. } if path == "arguments.extra"
    ));

    call.function.arguments = json!({"message": "ok"}).to_string();
    assert_eq!(
        registry
            .execute(&call)
            .await
            .expect("valid schema arguments should execute"),
        "ok"
    );
}

#[tokio::test]
async fn tool_arguments_have_size_and_depth_limits() {
    let mut registry = ToolRegistry::new();
    registry
        .register(SchemaTool)
        .expect("valid custom schema should register");
    let mut call = ToolCall {
        id: "bounded-call".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "schema_tool".to_string(),
            arguments: String::new(),
        },
    };

    call.function.arguments = json!({
        "message": "x".repeat(MAX_TOOL_ARGUMENT_BYTES)
    })
    .to_string();
    let oversized = registry
        .execute(&call)
        .await
        .expect_err("oversized arguments should be rejected");
    assert!(matches!(
        oversized,
        ToolError::ArgumentsTooLarge { limit, .. } if limit == MAX_TOOL_ARGUMENT_BYTES
    ));

    let mut nested = json!("leaf");
    for _ in 0..=MAX_TOOL_ARGUMENT_DEPTH {
        nested = json!([nested]);
    }
    call.function.arguments = json!({"message": nested}).to_string();
    let too_deep = registry
        .execute(&call)
        .await
        .expect_err("deep arguments should be rejected");
    assert!(matches!(
        too_deep,
        ToolError::ArgumentsTooDeep { limit, .. } if limit == MAX_TOOL_ARGUMENT_DEPTH
    ));
}

#[test]
fn registry_rejects_an_unsupported_tool_schema() {
    let error = ToolRegistry::new()
        .register(InvalidSchemaTool)
        .expect_err("tool parameters must be an object schema");

    assert!(matches!(
        error,
        RegistryError::InvalidDefinition { name, reason }
            if name == "invalid_schema_tool" && reason.contains("type object")
    ));
}

#[test]
fn registry_rejects_invalid_tool_definition_metadata() {
    let error = ToolRegistry::new()
        .register(InvalidMetadataTool)
        .expect_err("non-function tool definitions should be rejected");

    assert!(matches!(
        error,
        RegistryError::InvalidDefinition { name, reason }
            if name == "bad.name" && reason == "definition type must be function"
    ));
}

#[tokio::test]
async fn registry_uses_the_definition_snapshot_for_advertising_and_execution() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new();
    registry
        .register(ChangingDefinitionTool {
            definition_calls: Arc::clone(&calls),
        })
        .expect("initial definition should register");

    let definitions = registry.definitions();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        definitions[0].function.parameters["required"],
        json!(["message"])
    );

    let call = ToolCall {
        id: "changing-call".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "changing_tool".to_string(),
            arguments: "{}".to_string(),
        },
    };
    let error = registry
        .execute(&call)
        .await
        .expect_err("the registration snapshot should still require message");
    assert!(matches!(
        error,
        ToolError::SchemaViolation { path, .. } if path == "arguments.message"
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn schema_subset_checks_unicode_lengths_numeric_bounds_and_array_items() {
    let definition = ToolDefinition::function(
        "bounded_schema",
        "A schema exercising the supported constraints.",
        json!({
            "type": "object",
            "properties": {
                "label": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 3},
                "names": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["label", "count", "names"],
            "additionalProperties": false
        }),
    );
    validate_schema_definition(&definition).expect("supported schema should register");

    validate_schema_arguments(
        &definition,
        &json!({
            "label": "你好",
            "count": 2,
            "names": ["alpha", "beta"]
        }),
    )
    .expect("valid constraints should pass");

    let short_label =
        validate_schema_arguments(&definition, &json!({"label": "a", "count": 2, "names": []}))
            .expect_err("Unicode string length should be checked");
    assert!(matches!(
        short_label,
        ToolError::SchemaViolation { path, .. } if path == "arguments.label"
    ));

    let bad_item = validate_schema_arguments(
        &definition,
        &json!({"label": "ok", "count": 2, "names": [1]}),
    )
    .expect_err("array item types should be checked");
    assert!(matches!(
        bad_item,
        ToolError::SchemaViolation { path, .. } if path == "arguments.names[0]"
    ));

    let large_integer_definition = ToolDefinition::function(
        "large_integer_schema",
        "An integer precision regression fixture.",
        json!({
            "type": "object",
            "properties": {
                "value": {"type": "integer", "maximum": 9007199254740992u64}
            },
            "required": ["value"],
            "additionalProperties": false
        }),
    );
    validate_schema_definition(&large_integer_definition)
        .expect("large integer schema should be valid");
    let too_large = validate_schema_arguments(
        &large_integer_definition,
        &json!({"value": 9007199254740993u64}),
    )
    .expect_err("integer bounds must not lose precision through f64");
    assert!(matches!(
        too_large,
        ToolError::SchemaViolation { path, .. } if path == "arguments.value"
    ));

    let enum_definition = ToolDefinition::function(
        "enum_schema",
        "A schema exercising enum and maximum string length.",
        json!({
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["fast", "safe"],
                    "maxLength": 4
                },
                "label": {
                    "type": "string",
                    "maxLength": 4
                }
            },
            "required": ["mode", "label"],
            "additionalProperties": false
        }),
    );
    validate_schema_definition(&enum_definition).expect("enum and maxLength should be supported");
    validate_schema_arguments(&enum_definition, &json!({"mode": "safe", "label": "ok"}))
        .expect("an allowed enum value within the length bound should pass");
    let invalid_enum =
        validate_schema_arguments(&enum_definition, &json!({"mode": "slow", "label": "ok"}))
            .expect_err("an unknown enum value should be rejected");
    assert!(matches!(
        invalid_enum,
        ToolError::SchemaViolation { path, .. } if path == "arguments.mode"
    ));
    let too_long =
        validate_schema_arguments(&enum_definition, &json!({"mode": "safe", "label": "hello"}))
            .expect_err("a string over maxLength should be rejected");
    assert!(matches!(
        too_long,
        ToolError::SchemaViolation { path, .. } if path == "arguments.label"
    ));

    let invalid_enum_definition = ToolDefinition::function(
        "invalid_enum_schema",
        "An enum with a value of the wrong type.",
        json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": [1]}
            },
            "required": ["mode"],
            "additionalProperties": false
        }),
    );
    let invalid_enum_schema = validate_schema_definition(&invalid_enum_definition)
        .expect_err("enum members must match their declared type");
    assert!(invalid_enum_schema.contains("enum[0]"));

    let contradictory_schema = ToolDefinition::function(
        "contradictory_enum_schema",
        "An enum member outside the string length bounds.",
        json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["long"], "maxLength": 3}
            },
            "required": ["mode"],
            "additionalProperties": false
        }),
    );
    let contradictory = validate_schema_definition(&contradictory_schema)
        .expect_err("an enum outside maxLength should be rejected at registration");
    assert!(contradictory.contains("enum[0]"));
}

#[test]
fn bounded_details_falls_back_to_minimal_uncertain_status() {
    let result = ToolResult::error("uncertain")
        .with_details(json!({
            "status": "unknown",
            "payload": "x".repeat(128),
        }))
        .bounded(20);

    assert_eq!(result.details, json!({"status": "unknown"}));
}

#[tokio::test]
async fn read_file_returns_workspace_content() {
    let directory = tempdir().expect("temporary directory should exist");
    fs::write(directory.path().join("hello.txt"), "hello from workspace")
        .expect("fixture should be written");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let registry = ToolRegistry::with_read_file(context).expect("tool should register");

    let result = registry
        .execute(&read_call("hello.txt"))
        .await
        .expect("read_file should succeed");

    assert_eq!(result, "hello from workspace");
}

#[tokio::test]
async fn read_file_supports_one_indexed_offset_and_limit() {
    let directory = tempdir().expect("temporary directory should exist");
    fs::write(directory.path().join("lines.txt"), "one\ntwo\nthree\nfour")
        .expect("fixture should be written");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let registry = ToolRegistry::with_read_file(context).expect("tool should register");
    let mut call = read_call("lines.txt");
    call.function.arguments = serde_json::json!({
        "path": "lines.txt",
        "offset": 2,
        "limit": 2,
    })
    .to_string();

    let result = registry
        .execute(&call)
        .await
        .expect("ranged read should succeed");

    assert_eq!(result, "two\nthree");
}

#[tokio::test]
async fn read_file_rejects_an_offset_outside_the_file() {
    let directory = tempdir().expect("temporary directory should exist");
    fs::write(directory.path().join("lines.txt"), "one\ntwo").expect("fixture should be written");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let registry = ToolRegistry::with_read_file(context).expect("tool should register");
    let mut call = read_call("lines.txt");
    call.function.arguments = serde_json::json!({
        "path": "lines.txt",
        "offset": 3,
    })
    .to_string();

    let error = registry
        .execute(&call)
        .await
        .expect_err("out-of-range offset should fail");

    assert!(matches!(error, ToolError::InvalidRange { tool, .. } if tool == "read_file"));
}

#[tokio::test]
async fn read_file_rejects_parent_directory() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let registry = ToolRegistry::with_read_file(context).expect("tool should register");

    let error = registry
        .execute(&read_call("../secret.txt"))
        .await
        .expect_err("parent path should be rejected");

    assert!(matches!(error, ToolError::InvalidPath { .. }));
}

#[tokio::test]
async fn list_files_returns_sorted_entries_with_bounded_depth() {
    let directory = tempdir().expect("temporary directory should exist");
    fs::create_dir_all(directory.path().join("nested/deeper"))
        .expect("nested directories should be created");
    fs::write(directory.path().join("z.txt"), "z").expect("fixture should be written");
    fs::write(directory.path().join("a.txt"), "a").expect("fixture should be written");
    fs::write(directory.path().join("nested/child.txt"), "child")
        .expect("fixture should be written");
    fs::write(directory.path().join("nested/deeper/deep.txt"), "deep")
        .expect("fixture should be written");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_list_files(context)
        .expect("list_files should register");

    let result = registry
        .execute(&list_call(json!({
            "max_depth": 1,
            "max_entries": 10
        })))
        .await
        .expect("list_files should succeed");

    assert_eq!(
        result.lines().collect::<Vec<_>>(),
        vec![
            "a.txt",
            "nested/",
            "z.txt",
            "nested/child.txt",
            "nested/deeper/"
        ]
    );
    assert!(!result.contains("deep.txt"));
}

#[tokio::test]
async fn list_files_rejects_parent_directory_and_bounds_results() {
    let directory = tempdir().expect("temporary directory should exist");
    fs::write(directory.path().join("a.txt"), "a").expect("fixture should be written");
    fs::write(directory.path().join("b.txt"), "b").expect("fixture should be written");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_list_files(context)
        .expect("list_files should register");

    let error = registry
        .execute(&list_call(json!({"path": "../"})))
        .await
        .expect_err("parent path should be rejected");
    assert!(matches!(error, ToolError::InvalidPath { tool, .. } if tool == "list_files"));

    let result = registry
        .execute(&list_call(json!({"max_entries": 1})))
        .await
        .expect("bounded listing should succeed");
    assert!(result.contains("[truncated after 1 entries]"));
}

#[tokio::test]
async fn search_files_returns_literal_matches_with_line_numbers() {
    let directory = tempdir().expect("temporary directory should exist");
    fs::create_dir_all(directory.path().join("nested"))
        .expect("nested directory should be created");
    fs::create_dir_all(directory.path().join("target"))
        .expect("generated directory should be created");
    fs::write(
        directory.path().join("a.txt"),
        "needle at the beginning\nno match",
    )
    .expect("fixture should be written");
    fs::write(
        directory.path().join("nested/b.txt"),
        "first\nneedle in nested file",
    )
    .expect("fixture should be written");
    fs::write(
        directory.path().join("target/generated.txt"),
        "needle hidden",
    )
    .expect("generated fixture should be written");

    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_search_files(context)
        .expect("search_files should register");

    let result = registry
        .execute(&search_call(json!({"query": "needle"})))
        .await
        .expect("search_files should succeed");

    assert!(result.contains("a.txt:1:needle at the beginning"));
    assert!(result.contains("nested/b.txt:2:needle in nested file"));
    assert!(!result.contains("generated.txt"));
}

#[tokio::test]
async fn search_files_rejects_empty_query_and_bounds_matches() {
    let directory = tempdir().expect("temporary directory should exist");
    fs::write(directory.path().join("a.txt"), "needle one\nneedle two")
        .expect("fixture should be written");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_search_files(context)
        .expect("search_files should register");

    let empty = registry
        .execute(&search_call(json!({"query": ""})))
        .await
        .expect_err("empty query should fail validation");
    assert!(matches!(
        empty,
        ToolError::InvalidArgumentValue { name, .. } if name == "search_files"
    ));

    let result = registry
        .execute(&search_call(json!({
            "query": "needle",
            "max_results": 1
        })))
        .await
        .expect("bounded search should succeed");
    assert!(result.contains("[search truncated after 1 matching lines]"));
}

#[tokio::test]
async fn web_search_sends_bounded_query_and_formats_results() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/"))
        .and(header("X-Subscription-Token", "test-key"))
        .and(query_param("q", "rust"))
        .and(query_param("count", "1"))
        .and(query_param("country", "CN"))
        .and(query_param("search_lang", "zh-cn"))
        .and(query_param("freshness", "pm"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "web": {
                "results": [{
                    "title": "Rust",
                    "url": "https://www.rust-lang.org/",
                    "description": "A language empowering everyone."
                }]
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let tool =
        WebSearchTool::new("test-key", Duration::from_secs(5), 1024).with_endpoint(server.uri());
    let result = tool
        .execute_result(json!({
            "query": "rust",
            "count": 1,
            "country": "cn",
            "search_lang": "zh-cn",
            "freshness": "pm"
        }))
        .await
        .expect("web search should succeed");

    assert!(result.content.contains("Rust"));
    assert!(result.content.contains("https://www.rust-lang.org/"));
    assert_eq!(result.details["provider"], "brave");
    assert_eq!(result.details["result_count"], 1);
}

#[tokio::test]
async fn web_search_reports_provider_errors_and_empty_results() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/error"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/empty"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"web": {}})))
        .mount(&server)
        .await;

    let tool = WebSearchTool::new("test-key", Duration::from_secs(5), 1024);
    let error = tool
        .clone()
        .with_endpoint(format!("{}/error", server.uri()))
        .execute_result(json!({"query": "rust"}))
        .await
        .expect_err("provider error should be returned");
    assert!(
        error
            .to_string()
            .contains("HTTP 429 Too Many Requests: rate limited"),
        "unexpected provider error: {error}"
    );

    let result = tool
        .with_endpoint(format!("{}/empty", server.uri()))
        .execute_result(json!({"query": "rust"}))
        .await
        .expect("empty provider response should be valid");
    assert_eq!(result.content, "(no results)");
    assert_eq!(result.details["result_count"], 0);
}

#[tokio::test]
async fn web_search_rejects_oversized_provider_response() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200).set_body_bytes(vec![b'x'; MAX_WEB_RESPONSE_BYTES + 1]),
        )
        .mount(&server)
        .await;

    let error = WebSearchTool::new("test-key", Duration::from_secs(5), 1024)
        .with_endpoint(server.uri())
        .execute_result(json!({"query": "rust"}))
        .await
        .expect_err("oversized response should be rejected");
    assert!(error
        .to_string()
        .contains("search response exceeds the 2097152-byte limit"));
}

#[tokio::test]
async fn web_search_honors_cancellation_while_waiting_for_provider() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(5))
                .set_body_json(json!({"web": {"results": []}})),
        )
        .mount(&server)
        .await;

    let tool =
        WebSearchTool::new("test-key", Duration::from_secs(10), 1024).with_endpoint(server.uri());
    let cancellation = CancellationToken::new();
    let context = ToolExecutionContext::new(cancellation.clone());
    let task = tokio::spawn(async move {
        tool.execute_result_with_context(json!({"query": "rust"}), context)
            .await
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    cancellation.cancel();

    let error = task
        .await
        .expect("search task should finish")
        .expect_err("cancelled search should fail");
    assert_eq!(
        error.to_string(),
        "web search failed: search cancelled by the caller"
    );
}

#[test]
fn web_search_rejects_invalid_arguments() {
    let tool = WebSearchTool::new("test-key", Duration::from_secs(5), 1024);
    let error = tool
        .validate_arguments(&json!({"query": "", "count": 1}))
        .expect_err("empty query should fail validation");
    assert!(matches!(
        error,
        ToolError::InvalidArgumentValue { name, .. } if name == "web_search"
    ));
}

#[tokio::test]
async fn bash_executes_in_workspace_and_reports_command_metadata() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_shell(context)
        .expect("bash should register");

    let result = registry
        .execute_result(&shell_call(json!({
            "command": "printf 'hello\\n'; pwd"
        })))
        .await
        .expect("bash should execute");

    assert!(!result.is_error);
    assert!(result.content.contains("hello"));
    assert!(result
        .content
        .contains(&directory.path().to_string_lossy().to_string()));
    assert_eq!(result.details["status"], "completed");
    assert_eq!(result.details["sandboxed"], false);
    assert_eq!(
        registry.concurrency("bash"),
        Some(ToolConcurrency::Exclusive)
    );
    assert_eq!(
        registry.replay_policy("bash"),
        Some(ToolReplayPolicy::Never)
    );
    assert_eq!(registry.risk_class("bash"), Some(RiskClass::Irreversible));
}

#[tokio::test]
#[ignore = "requires bubblewrap user and network namespaces"]
async fn bash_executes_inside_the_managed_sandbox() {
    let directory = tempdir().expect("temporary directory should exist");
    let tool = ShellTool::with_sandbox_mode(
        ToolContext {
            workspace: directory.path().to_path_buf(),
            max_file_bytes: 1024,
            max_tool_output_bytes: 1024,
        },
        SandboxMode::Managed,
    )
    .expect("managed sandbox should be available");

    let result = tool
        .execute_result(json!({
            "command": "test -e /workspace && test ! -e /etc/shadow && printf 'sandbox-ok'"
        }))
        .await
        .expect("managed bash should execute");

    assert!(!result.is_error);
    assert_eq!(result.content, "sandbox-ok");
    assert_eq!(result.details["sandboxed"], true);
}

#[test]
fn bash_rejects_empty_and_out_of_range_arguments() {
    let tool = ShellTool::new(ToolContext {
        workspace: PathBuf::from("/tmp"),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    });

    for arguments in [
        json!({"command": ""}),
        json!({"command": "echo ok", "timeout": 0}),
    ] {
        let error = tool
            .validate_arguments(&arguments)
            .expect_err("invalid bash arguments should fail");
        assert!(matches!(
            error,
            ToolError::InvalidArgumentValue { name, .. } if name == "bash"
        ));
    }
}

#[tokio::test]
async fn bash_reports_nonzero_exit_and_bounded_output() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 128,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_shell(context)
        .expect("bash should register");

    let failed = registry
        .execute_result(&shell_call(json!({
            "command": "printf 'failure\\n' >&2; exit 7"
        })))
        .await
        .expect("nonzero exit should be a structured tool result");
    assert!(failed.is_error);
    assert!(failed.content.contains("failure"));
    assert_eq!(failed.details["status"], "exited_nonzero");
    assert_eq!(failed.details["status_code"], 7);

    let bounded = registry
        .execute_result(&shell_call(json!({
            "command": "printf '%*s' 256 x"
        })))
        .await
        .expect("bounded output should be a structured tool result");
    assert!(bounded.is_error);
    assert_eq!(bounded.details["status"], "truncated");
}

#[tokio::test]
async fn bash_honors_the_per_call_timeout() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_shell(context)
        .expect("bash should register");

    let result = registry
        .execute_result(&shell_call(json!({
            "command": "sleep 2",
            "timeout": 1
        })))
        .await
        .expect("timeout should be a structured tool result");

    assert!(result.is_error);
    assert_eq!(result.details["status"], "timed_out");
    assert_eq!(result.details["timeout_ms"], 1_000);
}

#[tokio::test]
async fn write_file_replaces_content_atomically_within_workspace() {
    let directory = tempdir().expect("temporary directory should exist");
    fs::write(directory.path().join("hello.txt"), "old").expect("fixture should be written");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry =
        ToolRegistry::with_read_file(context.clone()).expect("read tool should register");
    registry
        .register_write_file(context)
        .expect("write tool should register");

    let result = registry
        .execute(&write_call("hello.txt", "new contents"))
        .await
        .expect("write_file should succeed");

    assert!(result.contains("Wrote 12 bytes"));
    assert_eq!(
        fs::read_to_string(directory.path().join("hello.txt"))
            .expect("written file should be readable"),
        "new contents"
    );
    assert_eq!(
        registry.concurrency("write_file"),
        Some(ToolConcurrency::Exclusive)
    );
    assert_eq!(
        registry.risk_class("write_file"),
        Some(RiskClass::Reversible)
    );
}

#[tokio::test]
async fn write_file_creates_missing_nested_parent_directories() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_write_file(context)
        .expect("write tool should register");

    registry
        .execute(&write_call("nested/path/file.txt", "nested contents"))
        .await
        .expect("write_file should create nested parents");

    assert_eq!(
        fs::read_to_string(directory.path().join("nested/path/file.txt"))
            .expect("nested file should be readable"),
        "nested contents"
    );
}

#[tokio::test]
async fn write_file_rejects_parent_directory() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_write_file(context)
        .expect("write tool should register");

    let error = registry
        .execute(&write_call("../secret.txt", "blocked"))
        .await
        .expect_err("parent path should be rejected");

    assert!(matches!(error, ToolError::InvalidPath { tool, .. } if tool == "write_file"));
}

#[tokio::test]
async fn write_file_enforces_the_file_size_limit_before_creating_a_temp_file() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 3,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_write_file(context)
        .expect("write tool should register");

    let error = registry
        .execute(&write_call("hello.txt", "too large"))
        .await
        .expect_err("oversized content should be rejected");

    assert!(matches!(error, ToolError::FileTooLarge { limit: 3, .. }));
    assert!(!directory.path().join("hello.txt").exists());
}

#[tokio::test]
async fn edit_file_replaces_one_exact_match_atomically() {
    let directory = tempdir().expect("temporary directory should exist");
    fs::write(directory.path().join("hello.txt"), "hello world")
        .expect("fixture should be written");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_edit_file(context)
        .expect("edit tool should register");

    registry
        .execute(&edit_call("hello.txt", "world", "rust"))
        .await
        .expect("edit_file should succeed");

    assert_eq!(
        fs::read_to_string(directory.path().join("hello.txt"))
            .expect("edited file should be readable"),
        "hello rust"
    );
}

#[tokio::test]
async fn edit_file_requires_one_unique_match() {
    let directory = tempdir().expect("temporary directory should exist");
    fs::write(directory.path().join("hello.txt"), "same and same")
        .expect("fixture should be written");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::new();
    registry
        .register_edit_file(context)
        .expect("edit tool should register");

    let missing = registry
        .execute(&edit_call("hello.txt", "absent", "new"))
        .await
        .expect_err("missing target should fail");
    assert!(matches!(missing, ToolError::EditTargetNotFound { .. }));

    let duplicate = registry
        .execute(&edit_call("hello.txt", "same", "new"))
        .await
        .expect_err("duplicate target should fail");
    assert!(matches!(
        duplicate,
        ToolError::EditTargetNotUnique { matches: 2, .. }
    ));
}

#[tokio::test]
async fn read_file_rejects_unknown_arguments() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let registry = ToolRegistry::with_read_file(context).expect("tool should register");
    let mut call = read_call("hello.txt");
    call.function.arguments = serde_json::json!({
        "path": "hello.txt",
        "unexpected": true
    })
    .to_string();

    let error = registry
        .execute(&call)
        .await
        .expect_err("unknown arguments should fail validation");

    assert!(matches!(error, ToolError::InvalidArguments { name, .. } if name == "read_file"));
}

#[test]
fn validates_builtin_arguments_before_execution() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let registry = ToolRegistry::with_read_file(context).expect("tool should register");
    let mut call = read_call("hello.txt");
    call.function.arguments = serde_json::json!({
        "path": "hello.txt",
        "offset": 0
    })
    .to_string();
    let arguments: Value = serde_json::from_str(&call.function.arguments)
        .expect("test arguments should be valid JSON");

    let error = registry
        .validate_arguments(&call, &arguments)
        .expect_err("zero offset should fail preflight validation");

    assert!(matches!(
        error,
        ToolError::InvalidRange { tool, reason }
            if tool == "read_file" && reason == "offset must be greater than zero"
    ));
}

#[tokio::test]
async fn rejects_invalid_tool_call_invariants_before_lookup() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let registry = ToolRegistry::with_read_file(context).expect("tool should register");
    let mut call = read_call("hello.txt");
    call.id.clear();

    let error = registry
        .execute(&call)
        .await
        .expect_err("empty call ids should fail validation");

    assert!(
        matches!(error, ToolError::InvalidCall { reason, .. } if reason == "call id must not be empty")
    );
}

#[test]
fn built_in_read_tools_declare_parallel_execution() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry = ToolRegistry::with_read_file(context).expect("tool should register");
    assert_eq!(
        registry.concurrency("read_file"),
        Some(ToolConcurrency::Parallel)
    );
    assert_eq!(
        registry.replay_policy("read_file"),
        Some(ToolReplayPolicy::Safe)
    );
    assert_eq!(registry.risk_class("read_file"), Some(RiskClass::ReadOnly));
    assert_eq!(registry.concurrency("missing"), None);
    assert_eq!(registry.replay_policy("missing"), None);
    assert_eq!(registry.risk_class("missing"), None);

    registry
        .register_skill_reader(SkillCatalog::new(), 1024)
        .expect("read_skill should register");
    assert_eq!(
        registry.concurrency("read_skill"),
        Some(ToolConcurrency::Parallel)
    );
    assert_eq!(
        registry.replay_policy("read_skill"),
        Some(ToolReplayPolicy::Safe)
    );
    assert_eq!(registry.risk_class("read_skill"), Some(RiskClass::ReadOnly));
}

#[test]
fn mutating_tools_default_to_never_replay() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut registry =
        ToolRegistry::with_read_file(context.clone()).expect("read tool should register");
    registry
        .register_write_file(context.clone())
        .expect("write tool should register");
    registry
        .register_edit_file(context)
        .expect("edit tool should register");

    assert_eq!(
        registry.replay_policy("write_file"),
        Some(ToolReplayPolicy::Never)
    );
    assert_eq!(
        registry.replay_policy("edit_file"),
        Some(ToolReplayPolicy::Never)
    );
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

#[test]
fn hidden_tools_remain_executable_but_are_not_model_visible() {
    let mut registry = ToolRegistry::new();
    registry
        .register(HiddenTool)
        .expect("hidden tool should register");

    assert_eq!(registry.exposure("hidden_tool"), Some(ToolExposure::Hidden));
    assert_eq!(registry.definitions().len(), 1);
    assert!(registry.model_definitions().is_empty());
}

#[tokio::test]
async fn gateway_separates_model_discovery_from_trusted_execution() {
    let mut registry = ToolRegistry::new();
    registry
        .register(HiddenTool)
        .expect("hidden tool should register");
    let gateway = registry.gateway();
    let call = ToolCall {
        id: "hidden-call".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "hidden_tool".to_string(),
            arguments: "{}".to_string(),
        },
    };

    assert!(gateway.model_definitions().is_empty());
    assert_eq!(
        gateway.capabilities("hidden_tool"),
        Some(ToolCapabilities {
            concurrency: ToolConcurrency::Exclusive,
            replay_policy: ToolReplayPolicy::Never,
            exposure: ToolExposure::Hidden,
            risk_class: RiskClass::Unknown,
        })
    );
    assert!(matches!(
        gateway.prepare_model_call(&call, json!({})),
        Err(ToolError::ToolNotModelVisible { .. })
    ));

    let prepared = gateway
        .prepare_trusted_call(&call, json!({}))
        .expect("trusted callers may execute hidden tools");
    let result = gateway
        .execute_prepared(prepared)
        .await
        .expect("trusted execution should succeed");
    assert_eq!(result.content, "hidden");
}

#[test]
fn registry_can_remove_a_child_capability() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let registry = ToolRegistry::with_read_file(context).expect("tool should register");
    let restricted = registry.without("read_file");

    assert!(registry.concurrency("read_file").is_some());
    assert!(restricted.concurrency("read_file").is_none());
}

#[tokio::test]
async fn read_skill_returns_full_instructions_for_a_registered_name() {
    let mut catalog = SkillCatalog::new();
    catalog
        .insert(
            Skill::new(
                "research",
                "Research tasks",
                "Use primary sources.",
                "/skills/research/SKILL.md",
            )
            .expect("skill fixture should be valid")
            .with_model_invocation_disabled(true),
        )
        .expect("skill should be registered");
    let mut registry = ToolRegistry::new();
    registry
        .register_skill_reader(catalog, 1024)
        .expect("read_skill should register");

    let result = registry
        .execute(&skill_call("research"))
        .await
        .expect("registered skill should be readable");

    assert!(result.contains("Use primary sources."));
    assert!(result.contains("References are relative to /skills/research."));
}

#[tokio::test]
async fn read_skill_rejects_unknown_arguments() {
    let mut catalog = SkillCatalog::new();
    catalog
        .insert(
            Skill::new(
                "research",
                "Research tasks",
                "Use primary sources.",
                "/skills/research/SKILL.md",
            )
            .expect("skill fixture should be valid"),
        )
        .expect("skill should be registered");
    let mut registry = ToolRegistry::new();
    registry
        .register_skill_reader(catalog, 1024)
        .expect("read_skill should register");
    let call = ToolCall {
        id: "skill-call-2".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "read_skill".to_string(),
            arguments: serde_json::json!({
                "name": "research",
                "path": "outside.txt"
            })
            .to_string(),
        },
    };

    let error = registry
        .execute(&call)
        .await
        .expect_err("unknown arguments should fail validation");

    assert!(matches!(error, ToolError::InvalidArguments { name, .. } if name == "read_skill"));
}

#[tokio::test]
async fn read_skill_rejects_unknown_names() {
    let mut registry = ToolRegistry::new();
    registry
        .register_skill_reader(SkillCatalog::new(), 1024)
        .expect("read_skill should register");

    let error = registry
        .execute(&skill_call("missing"))
        .await
        .expect_err("unknown skill should fail");

    assert!(matches!(error, ToolError::UnknownSkill { name } if name == "missing"));
}

#[tokio::test]
async fn read_skill_enforces_output_limit_without_truncating_instructions() {
    let mut catalog = SkillCatalog::new();
    catalog
        .insert(
            Skill::new(
                "large",
                "Large skill",
                "This instruction must remain complete.",
                "/skills/large/SKILL.md",
            )
            .expect("skill fixture should be valid"),
        )
        .expect("skill should be registered");
    let mut registry = ToolRegistry::new();
    registry
        .register_skill_reader(catalog, 8)
        .expect("read_skill should register");

    let error = registry
        .execute(&skill_call("large"))
        .await
        .expect_err("oversized skill should fail");

    assert!(matches!(error, ToolError::SkillTooLarge { name, .. } if name == "large"));
}

#[test]
fn truncate_output_keeps_utf8_and_limit() {
    let result = truncate_output("你好，世界".to_string(), 7);

    assert!(result.is_char_boundary(result.len()));
    assert!(result.len() <= 7);
    assert!(result.ends_with('…'));
}
