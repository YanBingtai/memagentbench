#!/usr/bin/env python3
"""Test-only DeltaBox protocol fixture. It provides NO OS confinement."""
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

project = Path.cwd()
state = Path(os.environ["DELTABOX_STATE_DIR"])
state.mkdir(parents=True, exist_ok=True)


def snapshot(root):
    return {str(path.relative_to(root)): path.read_text()
            for path in root.rglob("*") if path.is_file()}


def replace_tree(source, target):
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def handle(method, params):
    if method == "session.start":
        sid = str(uuid.uuid4())
        home = state / sid
        home.mkdir()
        shutil.copytree(project, home / "view")
        shutil.copytree(project, home / "base")
        info = dict(session_id=sid, project=str(project), status="running",
                    resolved_fs_mode="copy-private", host_visibility="on-commit",
                    isolation="deltabox", isolation_provider="sandlock",
                    fallback_reason=None)
        (home / "info.json").write_text(json.dumps(info))
        return info
    home = state / params["session_id"]
    view = home / "view"
    info = json.loads((home / "info.json").read_text())
    if method == "session.exec":
        if params.get("label") == "capability-probe" and os.getenv("FAIL_DELTABOX_PROBE"):
            return dict(exit_code=1, stdout="", stderr="fixture capability denied",
                        duration_ms=1, truncated=False)
        child_env = {key: os.environ[key] for key in ("PATH", "LANG") if key in os.environ}
        result = subprocess.run(params["argv"], cwd=view, env=child_env,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True)
        return dict(exit_code=result.returncode, stdout=result.stdout,
                    stderr=result.stderr, duration_ms=1, truncated=False)
    if method == "session.status":
        return info
    if method == "session.stop":
        info["status"] = "stopped"
        (home / "info.json").write_text(json.dumps(info))
        return info
    if method == "diff.list":
        before, after = snapshot(home / "base"), snapshot(view)
        changed = sorted(path for path in before.keys() | after.keys()
                         if before.get(path) != after.get(path))
        return dict(pending_changes=len(changed), changes=[
            dict(path=path, text_diff="".join(difflib.unified_diff(
                before.get(path, "").splitlines(True),
                after.get(path, "").splitlines(True)))) for path in changed])
    if method == "changes.commit":
        for path in project.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        shutil.copytree(view, project, dirs_exist_ok=True)
        replace_tree(view, home / "base")
        return {"committed": True}
    if method == "changes.abort":
        replace_tree(home / "base", view)
        return {"restored": True}
    raise ValueError("unknown method " + method)


for line in sys.stdin:
    request = json.loads(line)
    with (state / "rpc.jsonl").open("a") as log:
        log.write(json.dumps(request) + "\n")
    try:
        response = dict(id=request["id"], result=handle(request["method"], request["params"]))
    except Exception as error:
        response = dict(id=request["id"], error=dict(code=-32000, message=str(error)))
    print(json.dumps(response), flush=True)
