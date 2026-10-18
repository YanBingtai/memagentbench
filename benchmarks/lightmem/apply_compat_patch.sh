#!/usr/bin/env bash
# Apply the project-owned LightMem compatibility patch to an external checkout.
# The official LightMem repository stays outside the root repository; this
# script is the reproducible overlay used by every canonical LightMem runner.

set -Eeuo pipefail

PROJECT_ROOT="${MEMORYARENA_PROJECT_ROOT:-${PI_AGENT_PROJECT_ROOT:-/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench}}"
LIGHTMEM_ROOT="${LIGHTMEM_ROOT:-$PROJECT_ROOT/methods/lightmem/LightMem}"
PATCH_FILE="$PROJECT_ROOT/benchmarks/lightmem/lightmem_compat.patch"
LIGHTMEM_EXPECTED_COMMIT="${LIGHTMEM_EXPECTED_COMMIT:-8449d574df6bae1bdf3314a1564da65e2f37e046}"

fail() {
  echo "[lightmem-compat] error: $*" >&2
  exit 1
}

git -C "$LIGHTMEM_ROOT" rev-parse --git-dir >/dev/null 2>&1 || {
  fail "LightMem checkout is not a git repository: $LIGHTMEM_ROOT"
}
[[ -f "$PATCH_FILE" ]] || fail "compatibility patch not found: $PATCH_FILE"

actual_commit="$(git -C "$LIGHTMEM_ROOT" rev-parse HEAD)"
[[ "$actual_commit" == "$LIGHTMEM_EXPECTED_COMMIT" ]] || {
  fail "unsupported LightMem commit: $actual_commit (expected $LIGHTMEM_EXPECTED_COMMIT)"
}

if git -C "$LIGHTMEM_ROOT" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "[lightmem-compat] already applied: $PATCH_FILE"
  exit 0
fi

git -C "$LIGHTMEM_ROOT" apply --check "$PATCH_FILE" >/dev/null 2>&1 || {
  fail "compatibility patch does not apply cleanly to $LIGHTMEM_ROOT; check the pinned LightMem checkout"
}

git -C "$LIGHTMEM_ROOT" apply "$PATCH_FILE"
echo "[lightmem-compat] applied: $PATCH_FILE"
