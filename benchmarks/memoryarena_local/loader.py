"""MemoryArena 本地 JSONL loader。

这个模块只负责读取和校验数据，不启动模型、环境服务或 NPU。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


DEFAULT_DATA_ROOT = Path("/mnt/nvme1p1/ybt/data/memoryarena")
SUPPORTED_CONFIGS = (
    "bundled_shopping",
    "progressive_search",
    "group_travel_planner",
    "formal_reasoning_math",
    "formal_reasoning_phys",
)


class DatasetSchemaError(ValueError):
    """本地数据结构不符合 MemoryArena 任务契约。"""


@dataclass(frozen=True)
class TaskStep:
    """一个按顺序执行的子任务；reference 只用于 evaluator。"""

    index: int
    question: str
    background: Any
    reference: Any


@dataclass(frozen=True)
class MemoryArenaTask:
    """一个完整任务，任务内的 steps 必须共享同一个 memory scope。"""

    config: str
    task_id: int
    steps: tuple[TaskStep, ...]
    metadata: Mapping[str, Any]

    @property
    def size(self) -> int:
        return len(self.steps)


def _context_for(config: str, raw: Mapping[str, Any], index: int) -> Any:
    if config in {"formal_reasoning_math", "formal_reasoning_phys"}:
        return raw["backgrounds"][index]
    if config == "group_travel_planner":
        # 基础旅行者的计划是后续旅行者共享的背景。
        return raw["base_person"]
    return None


def _validate_task(config: str, raw: Mapping[str, Any], line_no: int) -> MemoryArenaTask:
    prefix = f"{config}/line {line_no}"
    if not isinstance(raw.get("id"), int) or isinstance(raw.get("id"), bool):
        raise DatasetSchemaError(f"{prefix}: id 必须是整数")
    questions = raw.get("questions")
    answers = raw.get("answers")
    if not isinstance(questions, list) or not isinstance(answers, list):
        raise DatasetSchemaError(f"{prefix}: questions 和 answers 必须是 list")
    if not questions or len(questions) != len(answers):
        raise DatasetSchemaError(f"{prefix}: questions/answers 必须非空且长度相同")
    if any(not isinstance(q, str) or not q.strip() for q in questions):
        raise DatasetSchemaError(f"{prefix}: question 必须是非空字符串")

    if config == "bundled_shopping":
        for i, answer in enumerate(answers):
            if (
                not isinstance(answer, dict)
                or not isinstance(answer.get("target_asin"), str)
                or not isinstance(answer.get("attributes"), list)
            ):
                raise DatasetSchemaError(f"{prefix}/{i}: shopping answer 结构错误")
    elif config == "progressive_search":
        if any(not isinstance(answer, str) or not answer.strip() for answer in answers):
            raise DatasetSchemaError(f"{prefix}: progressive_search answer 必须是非空字符串")
    elif config == "group_travel_planner":
        person = raw.get("base_person")
        if not isinstance(person, dict) or not isinstance(person.get("daily_plans"), list):
            raise DatasetSchemaError(f"{prefix}: base_person.daily_plans 结构错误")
        if any(not isinstance(answer, list) for answer in answers):
            raise DatasetSchemaError(f"{prefix}: travel answer 必须是每日计划 list")
    elif config in {"formal_reasoning_math", "formal_reasoning_phys"}:
        backgrounds = raw.get("backgrounds")
        if not isinstance(raw.get("paper_name"), str):
            raise DatasetSchemaError(f"{prefix}: paper_name 必须是字符串")
        if not isinstance(backgrounds, list) or len(backgrounds) != len(questions):
            raise DatasetSchemaError(f"{prefix}: backgrounds 必须和 questions 对齐")
        if any(not isinstance(answer, str) for answer in answers):
            raise DatasetSchemaError(f"{prefix}: formal answer 必须是字符串")
    else:
        raise DatasetSchemaError(f"不支持的配置: {config}")

    steps = tuple(
        TaskStep(
            index=index,
            question=question,
            background=_context_for(config, raw, index),
            reference=answers[index],
        )
        for index, question in enumerate(questions)
    )
    metadata = {key: value for key, value in raw.items() if key not in {"questions", "answers", "backgrounds"}}
    return MemoryArenaTask(config=config, task_id=raw["id"], steps=steps, metadata=metadata)


def iter_tasks(
    config: str,
    root: Path | str = DEFAULT_DATA_ROOT,
    *,
    limit: int | None = None,
    task_ids: Sequence[int] | None = None,
) -> Iterator[MemoryArenaTask]:
    """顺序读取本地配置；不复制、不修改数据文件。"""

    if config not in SUPPORTED_CONFIGS:
        raise ValueError(f"config 必须是 {SUPPORTED_CONFIGS} 之一，当前为 {config!r}")
    data_path = Path(root) / config / "data.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"找不到本地数据文件: {data_path}")
    wanted = set(task_ids) if task_ids is not None else None
    seen: set[int] = set()
    yielded = 0
    with data_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetSchemaError(f"{config}/line {line_no}: JSON 无法解析") from exc
            if not isinstance(raw, dict):
                raise DatasetSchemaError(f"{config}/line {line_no}: 顶层必须是 object")
            task = _validate_task(config, raw, line_no)
            if task.task_id in seen:
                raise DatasetSchemaError(f"{config}: 重复 task id {task.task_id}")
            seen.add(task.task_id)
            if wanted is not None and task.task_id not in wanted:
                continue
            yield task
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def load_tasks(
    config: str,
    root: Path | str = DEFAULT_DATA_ROOT,
    *,
    limit: int | None = None,
    task_ids: Sequence[int] | None = None,
) -> list[MemoryArenaTask]:
    """将选定配置读入内存；全量运行建议使用 iter_tasks。"""

    return list(iter_tasks(config, root, limit=limit, task_ids=task_ids))
