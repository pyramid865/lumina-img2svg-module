# -*- coding: utf-8 -*-
"""自动内存预算：按系统可用物理内存限制多进程并行度。

转换一张 2000x3586 的图会同时存在超分图 / 标签图 / 量化图 / SVG 文本等多个
几十 MB 的对象，再叠加每个 spawn worker 一份 Python 解释器 + numpy/cv2
（实测 ~200-300 MB/个）。旧实现直接按 CPU 核数（本机 16）开满 worker，
峰值可把 32 GB 机器吃到只剩 2 GB，用户完全没法同时做别的事。

这里只限制「同时运行的进程数」，**不改变任务切分**（网格 / chunk 划分保持
原样），因此 pool.map 的输出顺序与内容逐字节不变，画质零影响，只是慢一点。
"""
from __future__ import annotations

import os

# 单个 worker 的保守常驻估算（含解释器、numpy、任务数据）
_PER_WORKER_MB = 320.0
# 无论如何给系统和用户其它程序留出的空闲内存
_KEEP_FREE_MB = 4096.0
# 并行度硬上限（再高收益递减、内存却线性涨）
_HARD_CAP = 8


def available_mb() -> float | None:
    """可用物理内存 MB；拿不到就返回 None（调用方走保守上限）。"""
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return float(st.ullAvailPhys) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil
        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return None


def workers(hint: int) -> int:
    """给定任务数，返回本次允许同时运行的进程数。

    - ``IMG2SVG_MAX_WORKERS=<n>`` 可手动硬压（低内存机器 / 需要同时干别的事）。
    - 自动：只用「(可用内存 - 保留量) 的一半」做预算，至少 1 个进程。
    - 任何情况下不超过 ``min(hint, 8)``。
    """
    hint = max(1, int(hint))
    env = os.environ.get("IMG2SVG_MAX_WORKERS", "").strip()
    if env.isdigit() and int(env) > 0:
        return max(1, min(hint, int(env)))
    avail = available_mb()
    if avail is None:
        return max(1, min(hint, _HARD_CAP))
    budget = max(0.0, (avail - _KEEP_FREE_MB) * 0.5)
    n = int(budget // _PER_WORKER_MB)
    return max(1, min(hint, _HARD_CAP, n))


def describe() -> str:
    avail = available_mb()
    if avail is None:
        return "内存预算：未知可用内存（保守上限 %d 进程）" % _HARD_CAP
    return "内存预算：可用 %.1f GB → 最多 %d 个并行进程" % (avail / 1024.0, workers(_HARD_CAP))
