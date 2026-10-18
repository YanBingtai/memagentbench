//! End-to-end CLI tests with a simulated supervisor and deterministic model.
//! These test our adapter and native tools, not DeltaBox's OS confinement.
#![cfg(unix)]

use serde_json::{json, Value};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    process::{Command, Output},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tempfile::{tempdir, TempDir};
use wiremock::{matchers::method, Mock, MockServer, Request, Respond, ResponseTemplate};

struct ScriptedModel {
    steps: Vec<(&'static str, Value)>,
    next: Arc<AtomicUsize>,
}

impl Respond for ScriptedModel {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        let index = self.next.fetch_add(1, Ordering::SeqCst);
        let (delta, finish) = if let Some((name, arguments)) = self.steps.get(index) {
            (
                json!({"role":"assistant","tool_calls":[{
                    "index":0, "id":format!("call-{index}"), "type":"function",
                    "function":{"name":name,"arguments":arguments.to_string()},
                }]}),
                "tool_calls",
            )
        } else {
            (
                json!({"role":"assistant","content":"TOOLS_VERIFIED"}),
                "stop",
            )
        };
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({"choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
            ))
    }
}

struct Fixture {
    root: TempDir,
    project: PathBuf,
    binary: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let root = tempdir().unwrap();
        let project = root.path().join("project");
        fs::create_dir(&project).unwrap();
        let binary = root.path().join("deltabox-fixture");
        fs::write(&binary, include_str!("fixtures/deltabox_rpc.py")).unwrap();
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();
        Self {
            root,
            project,
            binary,
        }
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_pi-agent-rust"));
        command
            .args(["--workspace"])
            .arg(&self.project)
            .args(["--deltabox-bin"])
            .arg(&self.binary)
            .env("DELTABOX_STATE_DIR", self.root.path().join("state"))
            .env("NO_PROXY", "127.0.0.1,localhost")
            .env("no_proxy", "127.0.0.1,localhost");
        command
    }

    async fn run(&self, server: &MockServer, extra: &[&str]) -> Output {
        let mut command = self.command();
        command
            .args(["--sandbox-mode", "deltabox", "--base-url"])
            .arg(server.uri())
            .args([
                "--model",
                "test-model",
                "--no-think",
                "--max-steps",
                "12",
                "--allow-unsafe-shell",
                "--session-file",
            ])
            .arg(self.root.path().join("transcript.jsonl"))
            .args(extra)
            .args(["--prompt", "exercise the workspace tools"]);
        tokio::task::spawn_blocking(move || command.output().unwrap())
            .await
            .unwrap()
    }

    fn transcript(&self) -> Vec<Value> {
        fs::read_to_string(self.root.path().join("transcript.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap()["message"].clone())
            .collect()
    }

    fn session_id(&self) -> String {
        if self.root.path().join("transcript.jsonl").exists() {
            if let Some(id) = self.transcript().iter().find_map(|message| {
                message["tool_result"]["details"]["deltabox_session_id"]
                    .as_str()
                    .map(str::to_owned)
            }) {
                return id;
            }
        }
        fs::read_to_string(self.root.path().join("state/rpc.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .find_map(|request| request["params"]["session_id"].as_str().map(str::to_owned))
            .unwrap()
    }

    fn action(&self, action: &str) -> Output {
        self.command()
            .args([
                "--deltabox-action",
                action,
                "--deltabox-session",
                &self.session_id(),
            ])
            .output()
            .unwrap()
    }
}

async fn model(steps: Vec<(&'static str, Value)>) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ScriptedModel {
            steps,
            next: Arc::new(AtomicUsize::new(0)),
        })
        .mount(&server)
        .await;
    server
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_deltabox_all_six_tools_share_private_view_and_review_commits_later() {
    let fixture = Fixture::new();
    fs::write(fixture.project.join("notes.txt"), "host-original").unwrap();
    let server = model(vec![
        (
            "write_file",
            json!({"path":"notes.txt","content":"original\n"}),
        ),
        ("read_file", json!({"path":"notes.txt"})),
        (
            "edit_file",
            json!({"path":"notes.txt","old_text":"original","new_text":"updated"}),
        ),
        (
            "bash",
            json!({"command":"printf 'shell\\n' >> notes.txt; printf generated > generated.txt"}),
        ),
        ("list_files", json!({})),
        ("search_files", json!({"query":"updated"})),
        ("read_file", json!({"path":"notes.txt"})),
    ])
    .await;
    let output = fixture.run(&server, &[]).await;
    assert_success(&output);
    assert!(String::from_utf8_lossy(&output.stdout).contains("TOOLS_VERIFIED"));
    assert_eq!(
        fs::read_to_string(fixture.project.join("notes.txt")).unwrap(),
        "host-original"
    );
    assert!(!fixture.project.join("generated.txt").exists());

    let tools: Vec<Value> = fixture
        .transcript()
        .into_iter()
        .filter(|message| message["role"] == "tool")
        .collect();
    assert_eq!(tools.len(), 7);
    for result in &tools {
        assert_eq!(result["tool_result"]["is_error"], false, "{result}");
        assert_eq!(result["tool_result"]["details"]["supervisor"], "deltabox");
        assert!(result["tool_result"]["details"]["deltabox_session_id"].is_string());
    }
    assert_eq!(tools[1]["content"], "original\n");
    assert_eq!(tools[6]["content"], "updated\nshell\n");
    assert!(tools[4]["content"]
        .as_str()
        .unwrap()
        .contains("generated.txt"));
    assert!(tools[5]["content"]
        .as_str()
        .unwrap()
        .contains("notes.txt:1:updated"));
    assert_success(&fixture.action("diff"));
    assert_success(&fixture.action("commit"));
    assert_eq!(
        fs::read_to_string(fixture.project.join("notes.txt")).unwrap(),
        "updated\nshell\n"
    );
    assert_eq!(
        fs::read_to_string(fixture.project.join("generated.txt")).unwrap(),
        "generated"
    );
    assert_success(&fixture.action("stop"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_deltabox_run_timeout_cancels_worker_and_aborts_without_late_writes() {
    let fixture = Fixture::new();
    let server = model(vec![(
        "bash",
        json!({"command":"sleep 3; printf late > late.txt"}),
    )])
    .await;
    let started = Instant::now();
    let output = fixture.run(&server, &["--run-timeout-secs", "1"]).await;
    assert!(!output.status.success());
    assert!(started.elapsed() < Duration::from_secs(5), "{output:?}");
    assert!(!fixture.project.join("late.txt").exists());
    assert_eq!(
        serde_json::from_slice::<Value>(&fixture.action("status").stdout).unwrap()["status"],
        "stopped"
    );
    let sid = fixture.session_id();
    assert!(!fixture
        .root
        .path()
        .join("state")
        .join(sid)
        .join("view/late.txt")
        .exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_deltabox_tool_errors_and_output_limits_survive_round_trip() {
    let fixture = Fixture::new();
    let server = model(vec![
        ("bash", json!({"command":"printf failure >&2; exit 7"})),
        ("bash", json!({"command":"yes x | head -c 200000"})),
        ("read_file", json!({"path":"../outside.txt"})),
        (
            "write_file",
            json!({"path":"abort.txt","content":"private"}),
        ),
    ])
    .await;
    let output = fixture
        .run(&server, &["--deltabox-on-success", "abort"])
        .await;
    assert_success(&output);
    assert!(!fixture.project.join("abort.txt").exists());
    let tools: Vec<Value> = fixture
        .transcript()
        .into_iter()
        .filter(|message| message["role"] == "tool")
        .collect();
    assert_eq!(tools[0]["tool_result"]["details"]["status_code"], 7);
    assert_eq!(tools[0]["tool_result"]["is_error"], true);
    assert_eq!(tools[1]["tool_result"]["details"]["status"], "truncated");
    assert_eq!(tools[2]["tool_result"]["is_error"], true);
    assert!(tools[2]["content"]
        .as_str()
        .unwrap()
        .contains("invalid path"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unavailable_deltabox_confinement_fails_before_contacting_model() {
    let fixture = Fixture::new();
    let server = MockServer::start().await;
    let mut command = fixture.command();
    command
        .env("FAIL_DELTABOX_PROBE", "1")
        .args(["--sandbox-mode", "deltabox", "--base-url"])
        .arg(server.uri())
        .args(["--prompt", "never send this prompt"]);
    let output = tokio::task::spawn_blocking(move || command.output().unwrap())
        .await
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("capability probe failed"));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires PI_AGENT_DELTABOX_BIN, Landlock, and writable DELTABOX_STATE_DIR"]
async fn real_deltabox_cli_executes_without_publishing_before_commit() {
    let mut fixture = Fixture::new();
    fixture.binary = std::env::var_os("PI_AGENT_DELTABOX_BIN").unwrap().into();
    let server = model(vec![
        ("bash", json!({"command":"printf actual > actual.txt"})),
        ("read_file", json!({"path":"actual.txt"})),
    ])
    .await;
    let output = fixture
        .run(
            &server,
            &[
                "--deltabox-fs-mode",
                "copy-private",
                "--deltabox-require-fs-mode",
            ],
        )
        .await;
    assert_success(&output);
    assert!(!fixture.project.join("actual.txt").exists());
    assert_success(&fixture.action("commit"));
    assert_eq!(
        fs::read_to_string(fixture.project.join("actual.txt")).unwrap(),
        "actual"
    );
    assert_success(&fixture.action("stop"));
}
