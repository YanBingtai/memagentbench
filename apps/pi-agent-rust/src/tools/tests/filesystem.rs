use super::*;

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
