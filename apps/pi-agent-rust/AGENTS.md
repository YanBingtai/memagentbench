# Repository Guidelines

## Project Structure & Module Organization

This repository is a Rust 2021 Cargo crate. The command-line entry point is
`src/main.rs`; `src/lib.rs` exposes the reusable library modules. Keep model
transport in `src/model.rs`, configuration loading in `src/config.rs`, message
serialization in `src/message.rs`, the model/tool orchestration loop in
`src/agent_loop.rs`, and tool implementations and workspace checks in
`src/tools.rs`. Tests currently live beside their implementations in inline
`#[cfg(test)]` modules. `shencha/result.json` is an evaluation artifact; do not
edit it unless a task specifically concerns that result.

## Build, Test, and Development Commands

- `cargo fmt --check` verifies rustfmt compliance; run `cargo fmt` to apply it.
- `cargo test` builds the crate and runs all unit and async integration-style
  tests.
- `cargo build` checks a normal debug build without running tests.
- `cargo run -- --help` lists CLI options. A local model can be queried with
  `cargo run -- --base-url http://127.0.0.1:8000/v1 --model your-model --prompt "..."`.

## Coding Style & Naming Conventions

Use standard rustfmt formatting with four-space indentation. Prefer small,
typed functions and `Result`-based error handling; use `thiserror` for public
error enums. Follow Rust naming conventions: `snake_case` for functions,
variables, and modules; `PascalCase` for types; `SCREAMING_SNAKE_CASE` for
constants. Add focused doc comments to public APIs and keep provider-specific
logic isolated in the model client.

## Testing Guidelines

Name tests for observable behavior, such as
`returns_final_assistant_message_without_tools`. Use `#[test]` for synchronous
logic and `#[tokio::test]` for async behavior. HTTP interactions should use the
existing `wiremock` fixtures, and filesystem tests should use `tempfile` so
they remain isolated. Run `cargo test` before submitting changes; no separate
coverage threshold is configured.

## Configuration & Security

The CLI defaults to an OpenAI-compatible local endpoint and reads an optional
bearer token from `OPENAI_API_KEY`. Never commit real keys or private workspace
files. Keep tool paths within the configured workspace and preserve the file
size and tool-output limits when adding tools.

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