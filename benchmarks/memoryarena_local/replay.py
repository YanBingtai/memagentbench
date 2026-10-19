"""使用 OpenAI 兼容接口对本地 MemoryArena JSONL 做顺序 replay。"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .loader import MemoryArenaTask, TaskStep


def _chat_request_payload(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if model.startswith("mimo-"):
        payload["max_completion_tokens"] = max_tokens
        payload["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}
        if not enable_thinking:
            payload["temperature"] = temperature
    else:
        payload.update(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
        )
    return payload


@dataclass
class ChatClient:
    base_url: str = "http://127.0.0.1:6273/v1"
    model: str = "qwen3.8"
    timeout: float = 180.0
    temperature: float = 0.0
    max_tokens: int = 1024
    enable_thinking: bool = False

    def complete(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        payload = json.dumps(
            _chat_request_payload(
                self.model,
                messages,
                self.temperature,
                self.max_tokens,
                self.enable_thinking,
            ),
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # 本地服务不应该经过终端里残留的 HTTP 代理。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"模型请求失败: {exc}") from exc
        try:
            choice = body["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"模型返回格式错误: {body!r}") from exc
        usage = dict(body.get("usage") or {})
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            usage["_finish_reason"] = finish_reason
        if usage.get("completion_tokens", 0) >= self.max_tokens:
            usage["_possibly_truncated"] = True
        return content, usage


@dataclass
class RollingMemory:
    max_chars: int = 12000
    entries: list[str] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        self.entries.append(f"Q: {question}\nA: {answer}")
        while len(self.render()) > self.max_chars and self.entries:
            self.entries.pop(0)

    def render(self) -> str:
        return "\n\n".join(self.entries)


def _background_text(step: TaskStep) -> str:
    if step.background is None:
        return ""
    return "\n任务背景：\n" + json.dumps(step.background, ensure_ascii=False)


def run_task(
    task: MemoryArenaTask,
    client: ChatClient,
    *,
    memory_mode: str = "rolling",
    memory_chars: int = 12000,
) -> list[dict[str, Any]]:
    """按顺序运行一个任务；参考答案只写入结果，不进入模型消息。"""

    if memory_mode not in {"none", "rolling"}:
        raise ValueError("memory_mode 只能是 none 或 rolling")
    memory = RollingMemory(max_chars=memory_chars)
    records: list[dict[str, Any]] = []
    for step in task.steps:
        history = memory.render() if memory_mode == "rolling" else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "你正在执行一个多阶段任务。只输出当前子问题的最终答案和必要的极短解释；"
                    "不要输出思考过程、分析草稿、任务复述、‘We need’ 等元话语，也不要输出工具调用格式。"
                    "只使用题目和允许的历史记忆，不要猜测未提供的事实。"
                ),
            },
            {
                "role": "user",
                "content": (
                    ("历史记忆：\n" + history + "\n\n" if history else "")
                    + _background_text(step)
                    + "\n当前子问题：\n"
                    + step.question
                ),
            },
        ]
        call_started = time.monotonic()
        answer, usage = client.complete(messages)
        call_elapsed = time.monotonic() - call_started
        records.append(
            {
                "task_id": task.task_id,
                "subtask_index": step.index,
                "question": step.question,
                "answer": answer,
                "usage": usage,
                "answer_chars": len(answer),
                "elapsed_sec": round(call_elapsed, 3),
                "possibly_truncated": bool(usage.get("_possibly_truncated", False)),
                "memory_mode": memory_mode,
                "reference": step.reference,
            }
        )
        if memory_mode == "rolling":
            memory.add(step.question, answer)
    return records


def extract_asin(text: str) -> str | None:
    """Shopping smoke 的轻量 ASIN 提取，不替代官方 evaluator。"""

    match = re.search(r"\b[A-Z0-9]{10}\b", text.upper())
    return match.group(0) if match else None
