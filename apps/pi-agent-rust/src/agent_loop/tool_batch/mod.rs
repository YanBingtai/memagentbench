//! Small execution primitives shared by sequential and parallel tool paths.

mod hook_wait;

pub(crate) use hook_wait::{await_tool_hook, ToolHookWait};
