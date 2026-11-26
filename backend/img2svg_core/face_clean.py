# -*- coding: utf-8 -*-
"""CNN 平滑区识别 (v7.17.4 新增) — 离线 ONNX 推理，输出概率图。

模型：feature_cnn-v0.1.0.onnx（94k 参数，371KB，仅 CPU 推理，
required provider = CPUExecutionProvider，不依赖 GPU / torch）。
输入原图 RGB，输出 H×W 的「该合并平滑区」概率，供 region_recolor 使用。

分块推理 + 羽化权重：避免大图一次吃满内存，也消除 tile 拼接缝。
"""
from __future__ import annotations

import os
import sys

import numpy as np

MODEL_NAME = "feature_cnn-v0.1.0.onnx"
_TILE, _STRIDE, _OV = 256, 192, 64
_SESSION = None


def model_path() -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "models", MODEL_NAME)


def _get_session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        _SESSION = ort.InferenceSession(model_path(),
                                        providers=["CPUExecutionProvider"])
    return _SESSION


def _feather() -> np.ndarray:
    ramp = np.ones(_TILE, np.float32)
    e = _OV // 2
    ramp[:e] = np.linspace(0.25, 1.0, e, dtype=np.float32)
    ramp[-e:] = np.linspace(1.0, 0.25, e, dtype=np.float32)
    return np.minimum(ramp[:, None], ramp[None, :])


def _axis_pos(n: int) -> list[int]:
    ps = list(range(0, max(1, n - _TILE + 1), _STRIDE))
    if not ps or ps[-1] != n - _TILE:
        ps.append(max(0, n - _TILE))
    return ps


def smooth_prob(arr: np.ndarray, progress=None) -> np.ndarray:
    """arr: H×W×3 uint8 → H×W float32 平滑区概率。"""
    h, w = arr.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    x = np.pad(arr.astype(np.float32) / 255.0,
               ((0, ph), (0, pw), (0, 0)), mode="reflect")
    H, W = x.shape[:2]
    sess = _get_session()
    acc = np.zeros((H, W), np.float32)
    wgt = np.zeros((H, W), np.float32)
    fw = _feather()
    ys, xs = _axis_pos(H), _axis_pos(W)
    for iy, y in enumerate(ys):
        for x0 in xs:
            p = np.ascontiguousarray(
                x[y:y + _TILE, x0:x0 + _TILE].transpose(2, 0, 1)[None])
            logit = sess.run(None, {"rgb": p})[0][0, 0]
            prob = 1.0 / (1.0 + np.exp(-logit))
            acc[y:y + _TILE, x0:x0 + _TILE] += prob * fw
            wgt[y:y + _TILE, x0:x0 + _TILE] += fw
        if progress and iy % 3 == 0:
            progress("CNN 识别平滑区 %d/%d 块…" %
                     ((iy + 1) * len(xs), len(ys) * len(xs)))
    return (acc / np.maximum(wgt, 1e-3))[:h, :w]
