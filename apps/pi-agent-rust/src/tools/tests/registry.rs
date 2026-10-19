use super::*;

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
