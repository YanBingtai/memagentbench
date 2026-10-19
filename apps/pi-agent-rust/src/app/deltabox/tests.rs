use clap::Parser;
use pi_agent_runtime::DeltaBoxFsMode;
use pi_agent_rust::Config;
use std::path::PathBuf;
use tempfile::tempdir;

use super::{config::deltabox_config, lifecycle::start_if_requested};
use crate::cli::Args;

#[test]
fn explicit_options_are_applied_without_discovering_a_binary() {
    let workspace = tempdir().expect("workspace should exist");
    let args = Args::try_parse_from([
        "pi-agent-rust",
        "--deltabox-bin",
        "/bin/true",
        "--deltabox-fs-mode",
        "copy-private",
        "--deltabox-require-fs-mode",
        "--prompt",
        "hello",
    ])
    .expect("arguments should parse");
    let config = Config {
        workspace: workspace.path().to_path_buf(),
        ..Config::default()
    };

    let config = deltabox_config(&args, &config).expect("configuration should build");

    assert_eq!(config.binary, PathBuf::from("/bin/true"));
    assert_eq!(config.project, workspace.path());
    assert_eq!(config.fs_mode, DeltaBoxFsMode::CopyPrivate);
    assert!(config.require_fs_mode);
}

#[test]
fn non_deltabox_modes_skip_startup() {
    let workspace = tempdir().expect("workspace should exist");
    let args = Args::try_parse_from(["pi-agent-rust", "--prompt", "hello"])
        .expect("arguments should parse");
    let config = Config {
        workspace: workspace.path().to_path_buf(),
        ..Config::default()
    };

    assert!(start_if_requested(&args, &config)
        .expect("managed mode should not fail")
        .is_none());
}
