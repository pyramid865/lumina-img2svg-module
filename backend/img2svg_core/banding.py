"""Tone banding that respects linework ("避线稿可变模糊").

The problem this solves: a smooth gradient is a continuum, so any quantiser
turns it into a field of small patches with ragged edges.  Blurring the image
first gives clean iso-lines, but a global blur also smears the linework and
leaves halos.

Approach (same idea as Lumina-Vec's `bands.py`):

1. find line pixels: darker than their local median by a contrast threshold;
2. distance transform from those lines;
3. blur radius grows with the distance, so flat gradient areas get a wide,
   smooth blur while everything near a line stays untouched;
4. line pixels themselves keep their original colour.

The result is a flat-banded image with crisp lines, which the vectoriser can
then segment into clean regions.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _luma(a: np.ndarray) -> np.ndarray:
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def line_mask(arr: np.ndarray, contrast: float = 0.10, win: int = 9) -> np.ndarray:
    """Pixels noticeably darker than their local background."""
    y = _luma(arr)
    med = cv2.medianBlur((y * 255).astype(np.uint8), win).astype(np.float32) / 255.0
    m = (med - y) > contrast
    # drop isolated specks: a real line is at least 2px wide and 4px long
    m8 = m.astype(np.uint8)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m8, 8)
    keep = np.zeros_like(m8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 6:
            keep[lab == i] = 1
    return keep.astype(bool)


def flatten(im: Image.Image, contrast: float = 0.10, max_sigma: float = 6.0,
            near: float = 1.5, protect_lines: bool = True) -> Image.Image:
    """Blur gradients progressively, keeping linework sharp."""
    arr = np.asarray(im.convert("RGB"), np.float32) / 255.0
    h, w = arr.shape[:2]
    lines = line_mask(arr, contrast) if protect_lines else np.zeros((h, w), bool)

    dist = cv2.distanceTransform((~lines).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)

    out = arr.copy()
    # sigma ramps from `near` up to `max_sigma` with distance from the lines
    sigmas = [near, near * 2, near * 4, max_sigma]
    bounds = [0.0, 2.0, 6.0, 14.0, 1e9]
    for s, lo, hi in zip(sigmas, bounds, bounds[1:]):
        band = (dist >= lo) & (dist < hi)
        if not band.any():
            continue
        s = float(min(s, max_sigma))
        blurred = cv2.GaussianBlur(arr, (0, 0), s)
        out[band] = blurred[band]

    if protect_lines:
        out[lines] = arr[lines]
    return Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")


def speck_stats(im: Image.Image, colors: int = 96, min_area: int = 40) -> dict:
    """Count small flat patches: a proxy for "mottled" output."""
    a = np.asarray(im.convert("RGB")).reshape(-1, 3).astype(np.float32)
    q = (a // 8).astype(np.int32)
    key = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    uniq, inv = np.unique(key, return_inverse=True)
    lab = inv.reshape(im.height, im.width).astype(np.int32)
    small = 0
    total = 0
    for k in range(len(uniq)):
        m = (lab == k).astype(np.uint8)
        if m.sum() < 4:
            continue
        n, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            total += 1
            if area < min_area:
                small += 1
    return dict(patches=total, small=small,
                ratio=round(small / max(1, total), 3))
