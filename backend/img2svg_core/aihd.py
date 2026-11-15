"""AI upscaling ("高清化") with Real-ESRGAN's anime video model via ONNX Runtime.

The pipeline follows Lumina-Vec: kill grain first (shrink by half), then let
the network regenerate sharp linework at 4x, then keep whichever size the
caller asked for.  Everything runs offline/local; the ONNX Runtime backend is
chosen by ``ort_backend`` (DirectML/CUDA GPU when available, CPU fallback).

    model: models/anime_sr.onnx  (SRVGGNetCompact, realesr-animevideov3, x4)
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
from PIL import Image

from ort_backend import create_session, pick_providers as _pick_providers

_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "anime_sr.onnx")
_session = None


def available() -> bool:
    if os.path.isfile(_MODEL) or os.path.isfile(_MODEL + ".data"):
        try:
            import onnxruntime  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False
    return False


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = create_session(_MODEL, so)
    return _session


def _run(tile: np.ndarray) -> np.ndarray:
    sess = _get_session()
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: tile[None].transpose(0, 3, 1, 2).astype(np.float32)})[0]
    return np.clip(out[0].transpose(1, 2, 0), 0.0, 1.0)


def upscale(im: Image.Image, tile: int = 192, overlap: int = 12) -> Image.Image:
    """4x anime upscale, tiled so memory stays flat."""
    a = np.asarray(im.convert("RGB"), np.float32) / 255.0
    h, w = a.shape[:2]
    scale = 4
    out = np.zeros((h * scale, w * scale, 3), np.float32)
    step = max(1, tile - overlap)
    for y in range(0, h, step):
        y1 = min(y + tile, h)
        for x in range(0, w, step):
            x1 = min(x + tile, w)
            patch = a[y:y1, x:x1]
            ph, pw = patch.shape[:2]
            pad_h = (8 - ph % 8) % 8
            pad_w = (8 - pw % 8) % 8
            if pad_h or pad_w:
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            res = _run(patch)[: ph * scale, : pw * scale]
            out[y * scale:y1 * scale, x * scale:x1 * scale] = res
    upscaled = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), "RGB")
    # Real-ESRGAN lifts pale regions (measured +1.66 luma on a real illustration).
    # Details are what we want, so the global tone is matched back to the source.
    try:
        from tone_lock import lock_tone

        upscaled, _tm = lock_tone(im.convert("RGB"), upscaled)
    except Exception:  # noqa: BLE001
        pass
    return upscaled


def preprocess(im: Image.Image, target_scale: float = 2.0, max_pixels: int = 16_000_000,
               grain: bool = True) -> tuple[Image.Image, dict[str, Any]]:
    """Shrink to kill grain, AI upscale 4x, then resize to *target_scale*."""
    w, h = im.size
    src = im
    if grain and min(w, h) >= 64:
        # half-size average removes JPEG grain; the network re-draws the lines
        small = im.resize((max(1, w // 2), max(1, h // 2)), Image.BOX)
        src = Image.blend(im, small.resize(im.size, Image.BICUBIC), 0.35)
    big = upscale(src)
    want_w = int(round(w * target_scale))
    want_h = int(round(h * target_scale))
    if want_w * want_h > max_pixels:
        k = (max_pixels / float(want_w * want_h)) ** 0.5
        want_w, want_h = max(1, int(want_w * k)), max(1, int(want_h * k))
    if (big.width, big.height) != (want_w, want_h):
        big = big.resize((want_w, want_h), Image.LANCZOS)
    return big, dict(width=big.width, height=big.height, model="animevideov3 x4")
