"""给官方 MemoryArena runner 使用的本地数据源适配。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .loader import DEFAULT_DATA_ROOT, SUPPORTED_CONFIGS


def load_records(
    dataset_config: Mapping[str, Any],
    *,
    split: str = "test",
    default_root: Path | str = DEFAULT_DATA_ROOT,
) -> list[dict[str, Any]]:
    """优先读取 local_file，否则回退到 Hugging Face datasets。

    返回的 list 保持官方 runner 所需的字典结构；本地数据只有 test split。
    """

    local_file = dataset_config.get("local_file")
    if local_file:
        if split != "test":
            raise ValueError("本地 MemoryArena 快照只有 test split")
        path = Path(str(local_file))
        if not path.is_absolute():
            path = Path(default_root) / path
        if not path.is_file():
            raise FileNotFoundError(f"local_file 不存在: {path}")
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no} 顶层必须是 object")
                records.append(value)
        return records

    hf_config = dataset_config.get("hf_config")
    hf_dataset = dataset_config.get("hf_dataset", "ZexueHe/memoryarena")
    if hf_config not in SUPPORTED_CONFIGS:
        raise ValueError(f"hf_config 必须是 {SUPPORTED_CONFIGS} 之一，当前为 {hf_config!r}")
    from datasets import load_dataset

    return list(load_dataset(hf_dataset, hf_config, split=split))
