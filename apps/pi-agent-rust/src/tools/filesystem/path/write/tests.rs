use std::fs;

use tempfile::tempdir;

use super::*;

#[tokio::test]
async fn resolves_nested_target_and_unique_temporary_path() {
    let directory = tempdir().expect("temporary directory should exist");

    let (target, temporary) =
        resolve_workspace_write_path(directory.path(), Path::new("nested/hello.txt"))
            .await
            .expect("write path should resolve");

    assert_eq!(target, directory.path().join("nested/hello.txt"));
    assert_eq!(temporary.parent(), target.parent());
    assert!(temporary
        .file_name()
        .expect("temporary file should have a name")
        .to_string_lossy()
        .starts_with(".hello.txt."));
    assert!(temporary
        .file_name()
        .expect("temporary file should have a name")
        .to_string_lossy()
        .ends_with(".tmp"));
}

#[tokio::test]
async fn rejects_empty_absolute_and_parent_paths() {
    let directory = tempdir().expect("temporary directory should exist");

    for requested in [Path::new(""), Path::new("../secret.txt")] {
        let error = resolve_workspace_write_path(directory.path(), requested)
            .await
            .expect_err("unsafe path should fail");
        assert!(matches!(error, ToolError::InvalidPath { tool, .. } if tool == "write_file"));
    }

    let absolute = directory.path().join("absolute.txt");
    let error = resolve_workspace_write_path(directory.path(), &absolute)
        .await
        .expect_err("absolute path should fail");
    assert!(matches!(error, ToolError::InvalidPath { tool, .. } if tool == "write_file"));
}

#[tokio::test]
async fn rejects_directory_target() {
    let directory = tempdir().expect("temporary directory should exist");
    fs::create_dir(directory.path().join("target")).expect("target directory should be created");

    let error = resolve_workspace_write_path(directory.path(), Path::new("target"))
        .await
        .expect_err("directory target should fail");
    assert!(matches!(error, ToolError::InvalidPath { tool, .. } if tool == "write_file"));
}

#[cfg(unix)]
#[tokio::test]
async fn rejects_symlink_parent_and_target() {
    use std::os::unix::fs::symlink;

    let directory = tempdir().expect("temporary directory should exist");
    let outside = tempdir().expect("outside directory should exist");
    symlink(outside.path(), directory.path().join("link"))
        .expect("parent symlink should be created");

    let parent_error = resolve_workspace_write_path(directory.path(), Path::new("link/file"))
        .await
        .expect_err("symlink parent should fail");
    assert!(matches!(parent_error, ToolError::InvalidPath { tool, .. } if tool == "write_file"));

    fs::write(outside.path().join("target"), "outside").expect("outside target should be written");
    symlink(
        outside.path().join("target"),
        directory.path().join("target"),
    )
    .expect("target symlink should be created");
    let target_error = resolve_workspace_write_path(directory.path(), Path::new("target"))
        .await
        .expect_err("symlink target should fail");
    assert!(matches!(
        target_error,
        ToolError::PathOutsideWorkspace { tool, .. } if tool == "write_file"
    ));
}
