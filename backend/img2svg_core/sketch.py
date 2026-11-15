"""Sketch / line-art vectoriser.

Painterly region tracers (lumina, potrace colour mode) treat anti-aliased grey
pixels around a 1-2 px ink line as dozens of separate colours, which shreds the
linework and litters the page with grey specks.

Line art needs the opposite approach: quantise *luminance* into a handful of
bands, trace each band as one mask, and stack them light-to-dark.  Bands are
strictly nested, so lines stay continuous and there is no speckle at all.  The
colour of each band is taken from the real pixels inside it, so a green
background or a green iris survives.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
from PIL import Image

from ptr import _potrace_path, find_potrace


def _luma(a: np.ndarray) -> np.ndarray:
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def build(im: Image.Image, bands: int = 5, workdir: str | None = None,
          min_area: int = 12, turd: int = 1, smooth: int = 0) -> tuple[str, dict]:
    """Return (svg_text, info) for a line-art style vectorisation."""
    exe = find_potrace()
    if not exe:
        return "", dict(bands=0)
    ref = im.convert("RGB")
    w, h = ref.size
    arr = np.asarray(ref, np.float32) / 255.0
    y = _luma(arr)

    # denoise a little, then split into (a) achromatic luminance bands and
    # (b) a few colour clusters - otherwise a green background leaks into the
    # white paper because both have similar luminance.
    y8 = (y * 255).astype(np.uint8)
    # No blur here. Measured on 50 one-pixel rules: bilateral -> grey mush, 3x3
    # median -> 25000 dark px collapse to 500 (the stroke IS the minority in a
    # 3x3 window). Line art is already binary-ish; noise removal costs the lines.
    sat = arr.max(2) - arr.min(2)
    chroma = sat > 0.14

    masks: list[tuple[np.ndarray, np.ndarray]] = []      # (mask, representative colour)

    # --- achromatic: nested luminance bands (light -> dark)
    grey = ~chroma
    if grey.any():
        gy = y8[grey]
        for lo, hi in ((0, 60), (60, 110), (110, 160), (160, 210), (210, 256)):
            sel = grey & (y8 >= lo) & (y8 < hi)
            if sel.sum() > 200:
                masks.append((sel, arr[sel].mean(0)))

    # --- chromatic: k-means on the colour pixels
    if chroma.sum() > 500:
        px = arr[chroma].reshape(-1, 3).astype(np.float32)
        k = min(8, max(2, int(chroma.sum() // 4000) + 2))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _, labels, centers = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lab_img = np.full(y8.shape, -1, np.int32)
        lab_img[chroma] = labels.ravel()
        for ci in range(k):
            sel = lab_img == ci
            if sel.sum() > 300:
                masks.append((sel, centers[ci].astype(np.float32)))

    fills: list[str] = []
    tmp = tempfile.mkdtemp(prefix="sk_", dir=workdir or os.getcwd())
    used = 0
    try:
        for i, (sel, col) in enumerate(masks):
            m = sel.astype(np.uint8)
            if int(m.sum()) < max(64, min_area * 4):
                continue
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            clean = np.zeros_like(m)
            for k2 in range(1, n):
                if int(stats[k2, cv2.CC_STAT_AREA]) >= min_area:
                    clean[lab == k2] = 1
            if not clean.any():
                continue
            d = _potrace_path(exe, clean, tmp, i, turd, 1.0, 0.2)
            if not d:
                continue
            rgb = "#%02x%02x%02x" % tuple(int(v) for v in (np.clip(col, 0, 1) * 255).round())
            fills.append('<path d="%s" fill="%s"/>' % (d, rgb))
            used += 1
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
        return "", dict(bands=0)

    # backdrop = brightest band colour (paper), so any hairline gap is invisible
    bg = "#%02x%02x%02x" % (255, 255, 255)
    body = '<rect x="0" y="0" width="%d" height="%d" fill="%s"/>\n' % (w, h, bg)
    body += '<g transform="translate(0,%d) scale(0.1,-0.1)">\n%s\n</g>\n' % (h, "\n".join(fills))
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="%d" height="%d" '
           'viewBox="0 0 %d %d">\n%s</svg>\n' % (w, h, w, h, body))
    return svg, dict(bands=used, width=w, height=h)


def looks_like_lineart(im: Image.Image) -> bool:
    """Cheap heuristic: few colours, lots of near-white and near-black pixels."""
    a = np.asarray(im.convert("RGB").resize((160, 160)), np.float32) / 255.0
    y = _luma(a)
    sat = a.max(2) - a.min(2)
    flat = (sat < 0.10)
    light = (y > 0.92) & flat
    dark = (y < 0.15) & flat
    return bool((light.mean() + dark.mean()) > 0.45 and flat.mean() > 0.55)
