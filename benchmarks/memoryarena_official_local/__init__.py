"""MemoryArena 官方环境协议的本地隔离适配层。

这个包不修改 `/mnt/nvme1p1/ybt/data/memoryarena`，只读取快照，并把
官方环境的 reset/tool/step/judge 边界暴露给本地 Pi runner。
"""

from .adapter import OfficialLocalAdapter, start_adapter_server

__all__ = ["OfficialLocalAdapter", "start_adapter_server"]
