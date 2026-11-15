# -*- coding: utf-8 -*-
"""Print mode: make a posterised image printable.

The slicer lays down one bead of plastic per pass, so any colour band narrower
than a bead cannot be printed - it either disappears (leaving a hole) or gets
over-extruded into a blob. This pass re-assigns those bands before tracing.

Rules (self-developed, no model):
  * a band whose inscribed half-width is below ``min_px/2`` is "narrow";
  * if both sides are clearly different from it, the band IS a line - keep it
    and thicken it to one bead, because deleting it would break the drawing;
  * otherwise it is anti-aliasing residue - merge it into the nearest thick
    neighbour, which keeps the image fully covered with no holes.

All thresholds derive from ``print_mm`` / ``line_mm``; nothing is hard-coded to
a particular artwork.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def bead_px(image_long_px: int, print_mm: float, line_mm: float = 0.42) -> float:
    """How many image pixels one extruded bead covers."""
    if print_mm <= 0 or image_long_px <= 0:
        return 0.0
    mm_per_px = float(print_mm) / float(image_long_px)
    return max(0.0, float(line_mm) / mm_per_px) if mm_per_px > 0 else 0.0


def _layer_widths(labels: np.ndarray, n_colors: int) -> np.ndarray:
    """Inscribed half-width per colour (max distance to a different label)."""
    widths = np.zeros(n_colors, np.float32)
    for c in range(n_colors):
        m = (labels == c)
        if not m.any():
            continue
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        widths[c] = float(dt.max())
    return widths


def apply_print_mode(labels: np.ndarray, palette: np.ndarray, min_px: float,
                     contrast_thr: float = 0.16, thicken: bool = True) -> tuple:
    """Return (labels, palette, stats).

    min_px: bead width in pixels. Labels equal to -1 are treated as empty.
    """
    if cv2 is None or min_px <= 1e-3:
        return labels, palette, {"merged": 0, "thickened": 0, "min_px": min_px}

    h, w = labels.shape
    n = int(labels.max()) + 1
    out = labels.copy()
    widths = _layer_widths(labels, n)
    narrow = widths < (min_px * 0.5)

    # Lab distance between colours, for the "is this a real line" test
    lab = cv2.cvtColor(palette.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab = lab.reshape(-1, 3).astype(np.float32)

    merged = 0
    thickened = 0
    k3 = np.ones((3, 3), np.uint8)

    for c in range(n):
        if not narrow[c]:
            continue
        m = (out == c)
        if not m.any():
            continue
        # thick neighbours touching this band
        nb = np.zeros(n, np.int32)
        dil = cv2.dilate(m.astype(np.uint8), k3)
        ring = (dil > 0) & (~m)
        ids, cnt = np.unique(out[ring], return_counts=True)
        for i, k in enumerate(ids):
            if 0 <= k < n:
                nb[k] += int(cnt[i])

        # side colours that matter (ignore slivers of other narrow bands)
        sides = [k for k in np.argsort(-nb) if nb[k] > 0 and not narrow[k]][:2]
        if not sides:
            continue

        def d(a, b):
            return float(np.sqrt(((lab[a] - lab[b]) ** 2).sum()))

        if len(sides) >= 2:
            s1, s2 = sides[0], sides[1]
            is_line = d(c, s1) > contrast_thr * 100 and d(c, s2) > contrast_thr * 100
        else:
            is_line = d(c, sides[0]) > contrast_thr * 100

        if is_line:
            if thicken and widths[c] < min_px * 0.5:
                # grow the band to roughly one bead so the slicer can lay it down
                need = int(round(min_px * 0.5 - widths[c]))
                if need > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (need * 2 + 1,) * 2)
                    grown = cv2.dilate(m.astype(np.uint8), k) > 0
                    grow_from = grown & (~m) & np.isin(out, sides)
                    out[grow_from] = c
                    thickened += 1
            continue

        # anti-aliasing residue -> nearest thick neighbour
        out[m] = sides[0]
        merged += 1

    return out, palette, {"merged": merged, "thickened": thickened,
                          "min_px": round(min_px, 2),
                          "narrow_bands": int(narrow.sum())}
