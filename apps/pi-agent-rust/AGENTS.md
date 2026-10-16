# Repository Guidelines

## Project Structure & Module Organization

This directory is a Rust 2021 Cargo **workspace** (`resolver = "2"`) with twelve
members; the root package `pi-agent-rust` is one of them. Members are split into
a core agent, two independent control planes, and a benchmark stack:

- **Core agent** — `src/` (`src/main.rs` CLI, `src/lib.rs` library surface).
  `agent_loop.rs` owns the model/tool orchestration loop, `model.rs` the
  OpenAI-compatible transport, `tools.rs` the tool registry, `config.rs`
  configuration loading, `message.rs` message serialization, `session.rs` the
  JSONL transcript, `skills.rs` `SKILL.md` loading, `events.rs` the ephemeral
  event protocol, and `subagent.rs` the read-only delegated child agent.
- **Memory control plane** — `memory/` (domain types and provider/storage
  contracts), `memory-agent/` (need prediction, provider routing, retrieval
  merging, projection), `memory-remote/` (JSON HTTP adapter for out-of-process
  methods), `artifacts/` (content-addressed blob store). `memory/` depends on
  nothing internal; `artifacts/` depends on `memory/`.
- **Runtime control plane** — `runtime/` holds the side-effect contracts
  (action proposal, admission, commit grants, preparations, JSONL event ledger)
  and `runtime-adapter/` composes that crate with the memory crates and the
  agent loop. `runtime/` depends only on `memory/`.
- **Benchmark stack** — `benchmark/` (experiment contracts and append-only
  JSONL traces), `benchmark-runner/` (ingest/retrieve/agent orchestration,
  `AgentStepExecutor` boundary), `benchmark-evaluator/` (offline scoring and
  aggregation), `agent-benchmark-adapter/` (the only crate that binds
  `AgentLoop` to the runner), and `jitrl/` (standalone retrieval-based scorer,
  currently not depended on by any other member).

Conventions worth preserving:

- Only the root package produces a binary; every other member is a library plus
  `examples/`. `runtime-adapter/` additionally ships `src/bin/`.
- Keep the memory and runtime contracts provider-neutral. `memory/` must not
  learn about a concrete database, embedding, or algorithm; `runtime/` must not
  depend on the agent loop or on a concrete memory backend.
- Tool calls currently execute **sequentially**; `RunPolicy::max_parallel_tool_calls`
  is validated but not yet consumed, and `RuntimeCoordinator::admit` admits one
  action at a time. Concurrency lives in preparation work instead
  (`runtime-adapter/src/preparation_scheduler.rs`,
  `src/speculative_memory.rs`, semaphore-bounded, coordinator behind a mutex).
  `admit_batch` exists in `runtime/src/coordinator.rs` but has no production
  caller yet. Do not assume parallel tool dispatch works.
- `shencha/` holds evaluation artifacts (`result1.json`,
  `diagnosis-*.md`/`.json`); do not edit them unless a task specifically
  concerns those results.
- Some members still carry a committed nested `Cargo.lock` from when they were
  built as standalone packages. Cargo ignores them inside a workspace; edit only
  the root `Cargo.lock`.

## Build, Test, and Development Commands

- `cargo fmt --check` verifies rustfmt compliance; run `cargo fmt` to apply it.
- `cargo test --workspace` builds every member and runs inline unit tests plus
  the integration suites under `tests/`.
- `cargo check --workspace --all-targets` type-checks without a full build
  (fastest way to validate a wide change).
- `cargo run -- --help` lists CLI options. A local model can be queried with
  `cargo run -- --base-url http://127.0.0.1:8000/v1 --model your-model --prompt "..."`.
- Member examples need an explicit package, for example
  `cargo run -p pi-agent-benchmark-runner --example trace_evaluator_smoke`.
  `jitrl`'s `local_model_smoke` example is gated behind its `local-model`
  feature.

## Coding Style & Naming Conventions

Use standard rustfmt formatting with four-space indentation. Prefer small,
typed functions and `Result`-based error handling; use `thiserror` for public
error enums. Follow Rust naming conventions: `snake_case` for functions,
variables, and modules; `PascalCase` for types; `SCREAMING_SNAKE_CASE` for
constants. Add focused doc comments to public APIs and keep provider-specific
logic isolated in the model client. Public APIs in the memory and runtime
crates are a stable contract; document guarantee boundaries (what is advisory,
what is enforced) rather than only the happy path.

## Testing Guidelines

Name tests for observable behavior, such as
`returns_final_assistant_message_without_tools`. Use `#[test]` for synchronous
logic and `#[tokio::test]` for async behavior. HTTP interactions should use the
existing `wiremock` fixtures, and filesystem tests should use `tempfile` so
they remain isolated. Inline `#[cfg(test)]` modules are the default; use a
crate's `tests/` directory for cross-module contracts (for example
`memory/tests/projection_store_contract.rs`,
`artifacts/tests/store_contract.rs`). Run `cargo test --workspace` before
submitting changes; no separate coverage threshold is configured.

## Configuration & Security

CLI defaults differ depending on how configuration is supplied. With no
`--config`, defaults are `http://127.0.0.1:6273/v1`, model `qwen3.8`, and
temperature `0.7`; when `--config` is given and a CLI override is absent, the
`Config` defaults apply instead (`http://127.0.0.1:8000/v1`, `your-model`,
temperature `0.0`). The bearer token is read from the environment variable
named by `api_key_env` (default `OPENAI_API_KEY`).

Never commit real keys or private workspace files. `environment.env` is
git-ignored and is not read by the Rust code; prefer environment injection over
adding new key files. Optional integrations read their own variables:
`BRAVE_SEARCH_API_KEY` (and `BRAVE_SEARCH_TIMEOUT_SECS`) enables the
`web_search` tool, which is silently skipped when unset. Keep tool paths within
the configured workspace and preserve the file size, tool-output, and
argument-depth limits when adding tools; new tools must declare an explicit
`risk_class` and `replay_policy`, because the runtime denies unknown-risk tools
by default.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects; the history includes concise subjects
such as `tool use` and scoped `chore:` changes. Pull requests should explain
the behavior change, identify configuration or API impact, link the relevant
issue when one exists, and report validation commands (at minimum
`cargo fmt --check` and `cargo test`). Keep unrelated formatting or generated
artifacts out of the diff.
## 实现准则

---
name: karpathy-guidelines
description: Behavioral guidelines to reduce common LLM coding mistakes. Use when writing, reviewing, or refactoring code to avoid overcomplication, make surgical changes, surface assumptions, and define verifiable success criteria.
license: MIT
---

### Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes, derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

#### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

#### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

#### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

#### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.