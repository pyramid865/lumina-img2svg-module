"""共享的 ONNX Runtime 设备选择。

历史 bug：`aihd._pick_providers()` 里直接用了 ``ort.get_available_providers()``，
但 ``ort`` 只在 ``_get_session()`` 内部 import（局部变量），因此该函数永远抛
NameError，被 ``except Exception`` 静默吞掉后回退到 CPU —— 明明装了
``onnxruntime-directml``（DmlExecutionProvider 可用）却一直跑 CPU。

现在所有 ONNX 会话都经本模块创建，并把实际生效的后端暴露给进度日志。
可用 ``IMG2SVG_ORT_PROVIDERS`` 环境变量强制指定，例如：
  IMG2SVG_ORT_PROVIDERS=CPUExecutionProvider            # 强制 CPU
  IMG2SVG_ORT_PROVIDERS=DmlExecutionProvider,CPUExecutionProvider
GPU EP 初始化失败时会自动回退 CPU，绝不中断转换。
"""
from __future__ import annotations

import os

_GPU_NAMES = ("DmlExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider")
_last: list[str] | None = None


def pick_providers() -> list[str]:
    """按可用性挑 EP：强制环境变量 > GPU(DML/CUDA/CoreML) > CPU。"""
    global _last
    want = os.environ.get("IMG2SVG_ORT_PROVIDERS", "").strip()
    if want:
        ps = [p.strip() for p in want.split(",") if p.strip()]
        _last = ps
        return ps
    try:
        import onnxruntime as ort
        avail = list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        ps = ["CPUExecutionProvider"]
        _last = ps
        return ps
    for name in _GPU_NAMES:
        if name in avail:
            ps = [name, "CPUExecutionProvider"]
            _last = ps
            return ps
    ps = ["CPUExecutionProvider"]
    _last = ps
    return ps


def create_session(path: str, so=None):
    """创建会话；GPU EP 建不起来时优雅回退 CPU。"""
    import onnxruntime as ort

    if so is None:
        so = ort.SessionOptions()
    ps = pick_providers()
    try:
        return ort.InferenceSession(path, so, providers=ps)
    except Exception:  # noqa: BLE001
        # 某些机器上 DML 设备被占用 / 驱动过旧；回退后仍可用。
        global _last
        _last = ["CPUExecutionProvider"]
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def active_provider() -> str:
    """返回本次会话实际请求的第一个 EP（供进度日志显示）。"""
    if _last is None:
        pick_providers()
    return (_last or ["CPUExecutionProvider"])[0]


def describe() -> str:
    p = active_provider()
    if p == "CPUExecutionProvider":
        return "CPU"
    return p.replace("ExecutionProvider", "") + " (GPU)"
