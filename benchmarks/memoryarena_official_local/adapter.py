"""MemoryArena 语义的本地、隔离 HTTP 适配器。

目的：给 LightMem/Pi 流式 runner 提供可审计的 loopback 工具边界。官方
checkout 的真实 WebShop/BrowseComp 服务由 ``20261016_160000...`` 脚本和
官方 smoke 单独验证；本适配器不假装把这些服务接入，资产存在但未配置
upstream/index bridge 时仍返回 ``not_implemented``，避免把 toy 结果写成
官方准确率。适配器只绑定 127.0.0.1，不修改数据集目录。
"""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SLOTS = ["current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


def _similarity(left: Any, right: Any) -> float:
    if not left or not right:
        return 0.0
    left_text = str(left).strip().lower()
    right_text = str(right).strip().lower()
    if left_text == "-" and right_text == "-":
        return 1.0
    # 与官方 travel_env.py 一致，只比较较短字符串长度的前缀。
    m = min(len(left_text), len(right_text))
    return difflib.SequenceMatcher(None, left_text[:m], right_text[:m]).ratio() if m else 0.0


def _day(plan: Any, day_index: Any) -> dict[str, Any] | None:
    if not isinstance(plan, list):
        return None
    for item in plan:
        if isinstance(item, dict) and (item.get("day") == day_index or item.get("days") == day_index):
            return item
    return None


def _parse_plan(text: str, name: str) -> list[dict[str, Any]]:
    pattern = rf"===\s*{re.escape(name)}'s Plan\s*===(.*?)(?====|$)"
    match = re.search(pattern, text or "", re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    days: list[dict[str, Any]] = []
    for day_match in re.finditer(r"Day\s*(\d+):(.*?)(?=Day\s*\d+:|$)", match.group(1), re.DOTALL | re.IGNORECASE):
        item: dict[str, Any] = {"day": int(day_match.group(1))}
        for line in day_match.group(2).strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            item[key] = value.strip()
        days.append(item)
    return days


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _token_f1(prediction: Any, reference: Any) -> float:
    pred = _normalise_text(prediction).split()
    ref = _normalise_text(reference).split()
    if not pred or not ref:
        return 1.0 if pred == ref else 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = sum(min(count, ref_counts.get(token, 0)) for token, count in pred_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


class OfficialLocalAdapter:
    """一个任务一个状态的本地 env server。

    `/prepare` 相当于官方 `reset` 后选择当前子任务，`/tool` 是 agent 工具调用，
    `/evaluate` 相当于官方 `env.step(action, ground_truth, need_judge=True)`。
    """

    def __init__(self, data_root: str | Path, config: str):
        self.data_root = Path(data_root)
        self.config = config
        self.lock = threading.RLock()
        self.state: dict[str, Any] = {}
        self.tool_events: list[dict[str, Any]] = []
        self._tmp_dirs: list[tempfile.TemporaryDirectory[str]] = []

    @property
    def env_name(self) -> str:
        return {
            "formal_reasoning_math": "math",
            "formal_reasoning_phys": "math",
            "group_travel_planner": "travel_planner",
            "bundled_shopping": "webshop",
            "progressive_search": "browsecomp-plus",
        }.get(self.config, self.config)

    def prepare(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.state = dict(payload)
            self.tool_events = []
            raw = payload.get("task_raw") or {}
            observation = {
                "env_name": self.env_name,
                "config": self.config,
                "task_id": payload.get("task_id"),
                "subtask_index": payload.get("subtask_index"),
                "tool_names": self.tool_names(),
                "base_person": raw.get("base_person") if self.config == "group_travel_planner" else None,
                "official_assets": self.asset_status(),
                "state": "ready",
            }
            self.state["observation"] = observation
            return {"status": "ready", "observation": observation}

    def tool_names(self) -> list[str]:
        if self.config in {"formal_reasoning_math", "formal_reasoning_phys"}:
            return ["memoryarena_reasoning", "memoryarena_coding"]
        if self.config == "bundled_shopping":
            return ["memoryarena_webshop_search", "memoryarena_webshop_click", "memoryarena_webshop_add_to_cart", "memoryarena_webshop_buy"]
        if self.config == "progressive_search":
            return ["memoryarena_web_search", "memoryarena_web_open"]
        return []

    def asset_status(self) -> dict[str, Any]:
        if self.config == "bundled_shopping":
            root = Path(os.environ.get("MEMORYARENA_SHOPPING_DATA_ROOT", "") or self.data_root / "shopping")
            product_catalog = Path(os.environ.get("MEMORYARENA_PRODUCT_CATALOG_DIR", "") or root / "product_catalog")
            search_index = root / "search_engine" / "indexes-full"
            required = [root / "items_shuffle.json", root / "items_ins_v2.json", root / "domain_data.json", product_catalog, search_index]
            found = [str(path) for path in required if path.exists()]
            return {
                "required": [str(path) for path in required],
                "found": found,
                "ready": all(path.exists() for path in required)
                and any(product_catalog.glob("*.json"))
                and any(search_index.iterdir()) if search_index.exists() else False,
                "upstream_service": os.environ.get("MEMORYARENA_WEBSHOP_URL", "http://127.0.0.1:36004"),
            }
        if self.config == "progressive_search":
            root = Path(os.environ.get("MEMORYARENA_SEARCH_DATA_ROOT", "") or self.data_root / "progressive_search_assets")
            corpus = Path(os.environ.get("MEMORYARENA_SEARCH_CORPUS", "") or root / "corpus.jsonl")
            query_data = root / "browsecomp_all_jsons.jsonl"
            index = Path(os.environ.get("MEMORYARENA_SEARCH_INDEX", "") or root / "embeddings")
            required = [corpus, query_data, index]
            found = [str(path) for path in required if path.exists()]
            return {
                "required": [str(path) for path in required],
                "found": found,
                "ready": all(path.exists() for path in required) and any(index.glob("*.index")),
            }
        return {"required": [], "found": [], "ready": True}

    def _record_tool(self, name: str, args: dict[str, Any], result: dict[str, Any], started: float) -> dict[str, Any]:
        event = {
            "tool": name,
            "arguments": args,
            "result": result,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        self.tool_events.append(event)
        return event

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            if name == "memoryarena_reasoning":
                result = self._reasoning(str(args.get("task", "")))
            elif name == "memoryarena_coding":
                result = self._coding(str(args.get("code", "")))
            elif name in {"memoryarena_webshop_search", "memoryarena_webshop_click", "memoryarena_webshop_add_to_cart", "memoryarena_webshop_buy", "memoryarena_web_search", "memoryarena_web_open"}:
                status = self.asset_status()
                if not status.get("ready"):
                    result = {
                        "status": "blocked",
                        "reason": "official_assets_missing",
                        "message": f"{self.env_name} requires official assets/service; found={status.get('found', [])}",
                        "asset_status": status,
                    }
                else:
                    result = {"status": "not_implemented", "message": "资产存在但需要启动官方服务适配器", "asset_status": status}
            else:
                result = {"status": "error", "reason": "unknown_tool", "tool": name}
        except Exception as exc:  # 工具失败必须返回给 agent，不能让 runner 丢记忆。
            result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        self._record_tool(name, args, result, started)
        return result

    def _reasoning(self, task: str) -> dict[str, Any]:
        if not task:
            return {"status": "error", "reason": "task is empty"}
        base_url = os.environ.get("MEMORYARENA_MODEL_BASE_URL", "http://127.0.0.1:6273/v1").rstrip("/")
        model = os.environ.get("MEMORYARENA_MODEL", "qwen3.8")
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": f"Solve this problem and return only the final answer:\n{task}"}], "temperature": 0, "max_tokens": 2048, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}, ensure_ascii=False).encode("utf-8")
        import urllib.request

        request = urllib.request.Request(f"{base_url}/chat/completions", data=body, headers={"Content-Type": "application/json"}, method="POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"status": "ok", "tool": "reasoning", "output": content, "usage": payload.get("usage", {})}

    def _coding(self, code: str) -> dict[str, Any]:
        if len(code) > 12000:
            return {"status": "error", "reason": "code_too_long"}
        temp = tempfile.TemporaryDirectory(prefix="memoryarena_code_")
        self._tmp_dirs.append(temp)
        script = Path(temp.name) / "main.py"
        script.write_text(code, encoding="utf-8")
        proc = subprocess.run([os.environ.get("PYTHON", "python3"), str(script)], cwd=temp.name, capture_output=True, text=True, timeout=15)
        return {"status": "ok" if proc.returncode == 0 else "error", "returncode": proc.returncode, "stdout": proc.stdout[-12000:], "stderr": proc.stderr[-12000:]}

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            action = str(payload.get("action", ""))
            reference = payload.get("reference")
            if self.config in {"formal_reasoning_math", "formal_reasoning_phys"}:
                f1 = _token_f1(action, reference)
                exact = _normalise_text(action) == _normalise_text(reference)
                reward = bool(exact)
                judge_mode = "adapted_local_token_exact"
            elif self.config == "group_travel_planner":
                raw = self.state.get("task_raw") or {}
                questions = raw.get("questions") or []
                question = str(self.state.get("question", ""))
                match_index = int(self.state.get("subtask_index", 0) or 0)
                name_match = re.search(r"I am ([A-Za-z][A-Za-z -]*)[.]", question)
                name = name_match.group(1).strip() if name_match else ""
                expected = reference if isinstance(reference, list) else []
                parsed = _parse_plan(action, name) if name else []
                slot_ok = bool(expected) and all(_day(parsed, item.get("days") or item.get("day")) and all(_similarity((item or {}).get(slot, "-"), (_day(parsed, item.get("days") or item.get("day")) or {}).get(slot, "-")) >= 0.7 for slot in SLOTS) for item in expected if isinstance(item, dict))
                reward = bool(slot_ok)
                f1 = 1.0 if reward else _token_f1(action, expected)
                exact = reward
                judge_mode = "official_travel_slot_judge_compatible"
                parsed_info = {"name": name, "parsed_days": parsed, "question_index": match_index, "question_count": len(questions)}
            elif self.config == "bundled_shopping":
                expected = reference.get("target_asin") if isinstance(reference, dict) else None
                found = re.findall(r"\b[A-Z0-9]{10}\b", action.upper())
                exact = bool(expected and found and found[-1] == str(expected).upper())
                reward = exact
                f1 = 1.0 if exact else 0.0
                judge_mode = "adapted_asin_match_assets_missing" if not self.asset_status().get("ready") else "official_webshop_pending"
                parsed_info = {"target_asin": expected, "asins_in_action": found, "asset_status": self.asset_status()}
            else:
                f1 = _token_f1(action, reference)
                exact = _normalise_text(action) == _normalise_text(reference)
                reward = bool(exact)
                judge_mode = "adapted_local_token_exact_search_assets_missing"
                parsed_info = {"asset_status": self.asset_status()}
            result = {
                "observation": {"final": action, "tool_trace": list(self.tool_events), "judge_mode": judge_mode},
                "reward": reward,
                "done": True,
                "info": {"judge_mode": judge_mode, "token_f1": round(float(f1), 6), "exact_match": exact, **(parsed_info if "parsed_info" in locals() else {})},
                "tool_events": list(self.tool_events),
            }
            return result

    def close(self) -> None:
        for temp in self._tmp_dirs:
            temp.cleanup()
        self._tmp_dirs.clear()


class _Handler(BaseHTTPRequestHandler):
    adapter: OfficialLocalAdapter

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, status: int, body: Any) -> None:
        data = _json_bytes(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/health", "/"}:
            self._send(200, {"status": "ok", "config": self.adapter.config, "env_name": self.adapter.env_name})
            return
        if path == "/env/get_observation":
            self._send(200, self.adapter.state.get("observation", {"state": "not_initialized"}))
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            if path in {"/prepare", "/env/initialize", "/env/reset"}:
                body = self.adapter.prepare(payload)
            elif path == "/tool":
                body = self.adapter.call_tool(str(payload.get("tool", "")), payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {})
            elif path in {"/evaluate", "/env/step"}:
                body = self.adapter.evaluate(payload)
            elif path in {"/close", "/env/close"}:
                self.adapter.close()
                body = {"status": "closed"}
            else:
                self._send(404, {"error": "not found"})
                return
            self._send(200, body)
        except Exception as exc:
            self._send(500, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def start_adapter_server(adapter: OfficialLocalAdapter) -> tuple[ThreadingHTTPServer, str]:
    """绑定随机 loopback 端口，返回可由 Pi extension 使用的 URL。"""

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    _Handler.adapter = adapter
    thread = threading.Thread(target=server.serve_forever, name="memoryarena-official-adapter", daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"
