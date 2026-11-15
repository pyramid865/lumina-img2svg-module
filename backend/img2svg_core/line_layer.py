# -*- coding: utf-8 -*-
"""Line layer, self-developed (no AI model).

Lumina-Vec finds strokes with two neural line detectors. That needs ~390 MB of
weights and is what makes its image large. Here the same job is done with
classic morphology, which costs nothing at runtime:

1. multi-scale black-hat: a stroke is *darker than what a closing of size k
   produces*, so the residual peaks exactly on stroke cores, independent of
   overall brightness;
2. ridge test along the stroke: the residual must be locally maximal across
   the stroke, not along it, which removes flat dark areas;
3. hysteresis: strong seeds grow into connected weak pixels, so a stroke is
   kept whole instead of being chopped into dashes.

The output is a 0..1 confidence field plus a binary stroke mask.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from ptr import _potrace_path, find_potrace


def _luma(a: np.ndarray) -> np.ndarray:
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def stroke_field(rgb: np.ndarray, scales=(3, 5, 9), ridge_gain: float = 0.6) -> np.ndarray:
    """0..1 stroke likelihood, independent of global brightness."""
    if cv2 is None:
        return np.zeros(rgb.shape[:2], np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # normalise local contrast so a dark shirt does not read as one big stroke
    gray = gray / max(1.0, float(gray.mean()))

    acc = np.zeros_like(gray)
    for k in scales:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker)
        acc = np.maximum(acc, bh)

    # ridges only: keep pixels that are a local max across the stroke direction
    gx = cv2.Sobel(acc, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(acc, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ridge = acc - ridge_gain * mag
    ridge = np.clip(ridge, 0.0, None)

    lo = float(np.percentile(ridge, 82))
    hi = float(np.percentile(ridge, 99.2))
    if hi - lo < 1e-6:
        return np.zeros_like(acc)
    f = np.clip((ridge - lo) / (hi - lo), 0.0, 1.0)
    return cv2.GaussianBlur(f, (0, 0), 0.6)


def stroke_mask(rgb: np.ndarray, keep: float = 0.55, seed: float = 0.85,
                min_area: int = 12) -> np.ndarray:
    """Binary stroke mask with hysteresis (weak pixels only if connected to strong)."""
    if cv2 is None:
        return np.zeros(rgb.shape[:2], np.uint8)
    f = stroke_field(rgb)
    weak = (f >= keep).astype(np.uint8)
    strong = (f >= seed).astype(np.uint8)
    if not strong.any():
        return np.zeros_like(weak)

    n, lab = cv2.connectedComponents(weak, 8)
    keep_ids = np.unique(lab[strong > 0])
    keep_ids = keep_ids[keep_ids > 0]
    out = np.isin(lab, keep_ids).astype(np.uint8)

    n2, lab2, stats, _ = cv2.connectedComponentsWithStats(out, 8)
    clean = np.zeros_like(out)
    for i in range(1, n2):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            clean[lab2 == i] = 1
    return clean


def apply(rgb: np.ndarray, labels: np.ndarray, palette: np.ndarray,
          keep: float = 0.55, seed: float = 0.85,
          max_new_ratio: float = 0.22) -> tuple:
    """Add one top stroke layer to a label map.

    Returns (labels, palette, created_ids). The new layer takes the mean colour
    of the strokes it covers, so strokes keep their real ink colour.
    """
    m = stroke_mask(rgb, keep=keep, seed=seed)
    if not m.any():
        return labels, palette, []
    if float(m.mean()) > max_new_ratio:
        return labels, palette, []          # too much "stroke" -> likely texture

    new_id = int(labels.max()) + 1
    out = labels.copy()
    out[m > 0] = new_id
    col = rgb[m > 0].mean(0).round().astype(np.uint8)
    pal = np.concatenate([palette, col[None, :]], axis=0)
    return out, pal, [new_id]


def build_from_mask(src_img, mask, turd: int = 1, workdir: str | None = None) -> str:
    """Render a pre-computed stroke *mask* (uint8, 1 = stroke) as a top line layer.

    Reuses the same potrace pipeline as :func:`build` but skips the morphology step,
    so an externally detected stroke map (e.g. the AI line models) is rendered verbatim.
    Returns the ``<g>…</g>`` snippet, or "" when nothing drawable was found.
    """
    exe = find_potrace()
    if not exe or not mask.any():
        return ""
    arr = np.asarray(src_img.convert("RGB"), np.float32) / 255.0
    h, w = mask.shape[:2]

    fills = []
    tmp = tempfile.mkdtemp(prefix="ai_line_", dir=workdir or os.getcwd())
    try:
        y = _luma(arr)
        bands = (y < 0.18, (y >= 0.18) & (y < 0.38), y >= 0.38)
        for tag, band in enumerate(bands):
            m = mask.astype(bool) & band
            if m.sum() < 60:
                continue
            d = _potrace_path(exe, m.astype(np.uint8), tmp, tag, turd, 1.0, 0.2)
            if not d:
                continue
            col = np.median(arr[m], axis=0)
            rgb = "#%02x%02x%02x" % tuple(int(v) for v in (col * 255).round())
            fills.append('<path d="%s" fill="%s"/>' % (d, rgb))
    finally:
        for f in os.listdir(tmp):
            try:
                os.remove(os.path.join(tmp, f))
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass

    if not fills:
        return ""
    return ('<g transform="translate(0,%d) scale(0.1,-0.1)">\n%s\n</g>\n'
            % (h, "\n".join(fills)))
