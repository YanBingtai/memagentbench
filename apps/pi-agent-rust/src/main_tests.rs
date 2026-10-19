use super::*;
use crate::cli::DeltaBoxFsModeArg;
use pi_agent_rust::ConfigError;
use std::{fs, path::PathBuf};
use tempfile::tempdir;

#[test]
fn thinking_is_enabled_by_default() {
    let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
        .expect("arguments should parse");

    assert_eq!(args.thinking_override(), None);
}

#[test]
fn think_flag_can_be_enabled() {
    let args = Args::try_parse_from(["pi-agent-rust", "--think", "--prompt", "hello"])
        .expect("arguments should parse");

    assert_eq!(args.thinking_override(), Some(true));
}

#[test]
fn think_mode_can_be_disabled() {
    let args = Args::try_parse_from(["pi-agent-rust", "--no-think", "--prompt", "hello"])
        .expect("arguments should parse");

    assert_eq!(args.thinking_override(), Some(false));
}

#[test]
fn tool_choice_defaults_to_auto() {
    let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
        .expect("arguments should parse");

    assert!(matches!(&args.tool_choice, ToolChoiceArg::Auto));
}

#[test]
fn tool_choice_none_is_parsed_and_mapped() {
    let args = Args::try_parse_from([
        "pi-agent-rust",
        "--tool-choice",
        "none",
        "--prompt",
        "hello",
    ])
    .expect("arguments should parse");

    assert!(matches!(&args.tool_choice, ToolChoiceArg::None));
    assert_eq!(ToolChoice::from(args.tool_choice), ToolChoice::None);
}

#[test]
fn tool_choice_required_is_parsed_and_mapped() {
    let args = Args::try_parse_from([
        "pi-agent-rust",
        "--tool-choice",
        "required",
        "--prompt",
        "hello",
    ])
    .expect("required tool choice should parse");

    assert!(matches!(&args.tool_choice, ToolChoiceArg::Required));
    assert_eq!(ToolChoice::from(args.tool_choice), ToolChoice::Required);
}

#[test]
fn tool_choice_function_is_parsed_and_mapped() {
    let args = Args::try_parse_from([
        "pi-agent-rust",
        "--tool-choice",
        "function:read_file",
        "--prompt",
        "hello",
    ])
    .expect("named function tool choice should parse");

    assert_eq!(
        ToolChoice::from(args.tool_choice),
        ToolChoice::Function {
            name: "read_file".to_string()
        }
    );
}

#[test]
fn tool_choice_function_rejects_an_empty_name() {
    let error = Args::try_parse_from([
        "pi-agent-rust",
        "--tool-choice",
        "function:",
        "--prompt",
        "hello",
    ])
    .expect_err("an empty function name should be rejected");

    assert!(error.to_string().contains("1-64 character name"));
}

#[test]
fn tool_choice_function_rejects_invalid_name_characters() {
    let error = Args::try_parse_from([
        "pi-agent-rust",
        "--tool-choice",
        "function:read.file",
        "--prompt",
        "hello",
    ])
    .expect_err("invalid function name characters should be rejected");

    assert!(error.to_string().contains("1-64 character name"));
}

#[test]
fn tool_choice_function_must_be_registered_and_model_visible() {
    let directory = tempdir().expect("temporary directory should exist");
    let context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let tools = ToolRegistry::with_read_file(context).expect("read tool should register");

    let known = resolve_tool_choice(
        &ToolChoiceArg::Function {
            name: "read_file".to_string(),
        },
        &tools,
    )
    .expect("registered tool should be selectable");
    assert_eq!(
        known,
        ToolChoice::Function {
            name: "read_file".to_string()
        }
    );

    let error = resolve_tool_choice(
        &ToolChoiceArg::Function {
            name: "missing_tool".to_string(),
        },
        &tools,
    )
    .expect_err("unknown tool should fail before the model request");
    assert!(matches!(
        error,
        AppError::UnknownToolChoiceFunction { name } if name == "missing_tool"
    ));
}

#[test]
fn run_timeout_is_unset_by_default() {
    let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
        .expect("arguments should parse");

    assert_eq!(args.run_timeout_secs, None);
}

#[test]
fn run_timeout_can_be_configured() {
    let args = Args::try_parse_from([
        "pi-agent-rust",
        "--run-timeout-secs",
        "300",
        "--prompt",
        "hello",
    ])
    .expect("arguments should parse");

    assert_eq!(args.run_timeout_secs, Some(300));
}

#[test]
fn system_prompt_can_be_configured() {
    let args = Args::try_parse_from([
        "pi-agent-rust",
        "--system-prompt",
        "Act as a careful coding agent.",
        "--prompt",
        "hello",
    ])
    .expect("system prompt should parse");

    assert_eq!(
        args.system_prompt.as_deref(),
        Some("Act as a careful coding agent.")
    );
}

#[test]
fn verbose_is_disabled_by_default() {
    let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
        .expect("arguments should parse");

    assert!(!args.verbose);
}

#[test]
fn unsafe_shell_requires_an_explicit_flag() {
    let default_args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
        .expect("arguments should parse");
    assert!(!default_args.allow_unsafe_shell);

    let enabled_args =
        Args::try_parse_from(["pi-agent-rust", "--allow-unsafe-shell", "--prompt", "hello"])
            .expect("unsafe shell flag should parse");
    assert!(enabled_args.allow_unsafe_shell);
}

#[test]
fn sandbox_mode_defaults_to_managed_and_accepts_explicit_disabled_mode() {
    let default_args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
        .expect("arguments should parse");
    assert_eq!(default_args.sandbox_mode, SandboxModeArg::Managed);

    let disabled_args = Args::try_parse_from([
        "pi-agent-rust",
        "--sandbox-mode",
        "disabled",
        "--prompt",
        "hello",
    ])
    .expect("disabled sandbox mode should parse");
    assert_eq!(disabled_args.sandbox_mode, SandboxModeArg::Disabled);

    let deltabox_args = Args::try_parse_from([
        "pi-agent-rust",
        "--sandbox-mode",
        "deltabox",
        "--deltabox-fs-mode",
        "copy-private",
        "--prompt",
        "hello",
    ])
    .expect("deltabox sandbox mode should parse");
    assert_eq!(deltabox_args.sandbox_mode, SandboxModeArg::Deltabox);
    assert_eq!(
        deltabox_args.deltabox_fs_mode,
        DeltaBoxFsModeArg::CopyPrivate
    );
}

#[test]
fn verbose_flag_can_be_enabled() {
    let args = Args::try_parse_from(["pi-agent-rust", "--verbose", "hello"])
        .expect("arguments should parse");

    assert!(args.verbose);
}

#[test]
fn runtime_reconciliation_arguments_parse_without_a_prompt() {
    let action_id = Uuid::new_v4();
    let args = Args::try_parse_from([
        "pi-agent-rust",
        "--reconcile-action-id",
        &action_id.to_string(),
        "--reconcile-evidence",
        "provider-check-1",
        "--reconcile-error",
        "provider confirmed no execution",
    ])
    .expect("reconciliation arguments should parse");

    assert_eq!(args.reconcile_action_id, Some(action_id));
    assert_eq!(args.reconcile_evidence.as_deref(), Some("provider-check-1"));
    assert_eq!(
        args.reconcile_error.as_deref(),
        Some("provider confirmed no execution")
    );
}

#[test]
fn runtime_reconciliation_rejects_a_prompt() {
    let error = Args::try_parse_from([
        "pi-agent-rust",
        "--reconcile-action-id",
        &Uuid::new_v4().to_string(),
        "--prompt",
        "hello",
    ])
    .expect_err("reconciliation must be a standalone operation");

    assert!(error.to_string().contains("cannot be used with"));
}

#[test]
fn skills_directory_can_be_selected() {
    let args = Args::try_parse_from([
        "pi-agent-rust",
        "--skills-dir",
        "/tmp/skills",
        "--prompt",
        "hello",
    ])
    .expect("arguments should parse");

    assert_eq!(args.skills_dir, Some(PathBuf::from("/tmp/skills")));
}

#[test]
fn skills_directory_loads_a_catalog_for_the_agent() {
    let directory = tempdir().expect("temporary directory should exist");
    let skill_directory = directory.path().join("research");
    fs::create_dir_all(&skill_directory).expect("skill directory should be created");
    fs::write(
        skill_directory.join("SKILL.md"),
        "---\nname: research\ndescription: Research tasks\n---\nUse primary sources.",
    )
    .expect("skill fixture should be written");

    let catalog = load_skills(Some(directory.path()), 1024)
        .expect("skill directory should load")
        .expect("valid skill should produce a catalog");

    assert_eq!(catalog.len(), 1);
    assert_eq!(
        catalog.get("research").unwrap().content,
        "Use primary sources."
    );
}

#[test]
fn skills_directory_registers_the_default_filesystem_tools() {
    let directory = tempdir().expect("temporary directory should exist");
    let skill_directory = directory.path().join("research");
    fs::create_dir_all(&skill_directory).expect("skill directory should be created");
    fs::write(
        skill_directory.join("SKILL.md"),
        "---\nname: research\ndescription: Research tasks\n---\nUse primary sources.",
    )
    .expect("skill fixture should be written");
    let skills = load_skills(Some(directory.path()), 1024)
        .expect("skill directory should load")
        .expect("valid skill should produce a catalog");
    let tool_context = ToolContext {
        workspace: directory.path().to_path_buf(),
        max_file_bytes: 1024,
        max_tool_output_bytes: 1024,
    };
    let mut tools =
        ToolRegistry::with_read_file(tool_context.clone()).expect("read_file should register");
    tools
        .register_write_file(tool_context.clone())
        .expect("write_file should register");
    tools
        .register_list_files(tool_context.clone())
        .expect("list_files should register");
    tools
        .register_search_files(tool_context.clone())
        .expect("search_files should register");
    tools
        .register_edit_file(tool_context)
        .expect("edit_file should register");

    tools
        .register_skill_reader(skills, 1024)
        .expect("read_skill should register");

    assert!(tools
        .definitions()
        .iter()
        .any(|definition| definition.function.name == "read_skill"));
    assert!(tools
        .definitions()
        .iter()
        .any(|definition| definition.function.name == "write_file"));
    assert!(tools
        .definitions()
        .iter()
        .any(|definition| definition.function.name == "edit_file"));
    assert!(tools
        .definitions()
        .iter()
        .any(|definition| definition.function.name == "list_files"));
    assert!(tools
        .definitions()
        .iter()
        .any(|definition| definition.function.name == "search_files"));
}

#[test]
fn skills_directory_uses_the_configured_file_limit() {
    let directory = tempdir().expect("temporary directory should exist");
    let skill_directory = directory.path().join("research");
    fs::create_dir_all(&skill_directory).expect("skill directory should be created");
    fs::write(
            skill_directory.join("SKILL.md"),
            "---\nname: research\ndescription: Research tasks\n---\nThis body is larger than the test limit.",
        )
        .expect("skill fixture should be written");

    let catalog = load_skills(Some(directory.path()), 16).expect("skill directory should load");

    assert!(catalog.is_none());
}

#[test]
fn config_file_values_are_used_when_cli_does_not_override_them() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("agent.toml");
    fs::write(
        &path,
        r#"
                base_url = "http://config.example/v1"
                model = "config-model"
                workspace = "config-workspace"
                run_timeout_secs = 45
            "#,
    )
    .expect("config fixture should be written");
    let args = Args::try_parse_from([
        "pi-agent-rust",
        "--config",
        path.to_str().expect("config path should be valid UTF-8"),
        "--prompt",
        "hello",
    ])
    .expect("arguments should parse");

    let config = load_config(&args).expect("config should load");

    assert_eq!(config.base_url, "http://config.example/v1");
    assert_eq!(config.model, "config-model");
    assert_eq!(config.workspace, directory.path().join("config-workspace"));
    assert_eq!(config.run_timeout_secs, Some(45));
}

#[test]
fn no_config_file_preserves_cli_defaults() {
    let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
        .expect("arguments should parse");

    let config = load_config(&args).expect("default config should load");

    assert_eq!(config.base_url, "http://127.0.0.1:6273/v1");
    assert_eq!(config.model, "qwen3.8");
    assert_eq!(config.temperature, 0.7);
}

#[test]
fn explicit_cli_values_override_config_file_values() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("agent.toml");
    fs::write(&path, "model = \"config-model\"\n").expect("config fixture should be written");
    let args = Args::try_parse_from([
        "pi-agent-rust",
        "--config",
        path.to_str().expect("config path should be valid UTF-8"),
        "--model",
        "cli-model",
        "--prompt",
        "hello",
    ])
    .expect("arguments should parse");

    let config = load_config(&args).expect("config should load");

    assert_eq!(config.model, "cli-model");
}

#[test]
fn missing_config_file_preserves_typed_error() {
    let directory = tempdir().expect("temporary directory should exist");
    let path = directory.path().join("missing.toml");
    let args = Args::try_parse_from([
        "pi-agent-rust",
        "--config",
        path.to_str().expect("config path should be valid UTF-8"),
        "--prompt",
        "hello",
    ])
    .expect("arguments should parse");

    let error = load_config(&args).expect_err("missing config should fail");

    assert!(matches!(error, AppError::Config(ConfigError::Read { .. })));
}

#[test]
fn missing_session_id_creates_a_new_id() {
    let id = resolve_session_id(None).expect("a missing id should create one");

    assert_ne!(id, Uuid::nil());
}

#[test]
fn valid_session_id_is_preserved() {
    let expected = Uuid::new_v4();

    assert_eq!(
        resolve_session_id(Some(&expected.to_string())).expect("id should parse"),
        expected
    );
}

#[test]
fn invalid_session_id_is_rejected() {
    assert!(matches!(
        resolve_session_id(Some("not-a-uuid")),
        Err(AppError::InvalidSessionId { .. })
    ));
}
