"""OKLab colour helpers shared by the quantiser and the cel-shading pipeline."""

from __future__ import annotations

import numpy as np
from PIL import Image


def srgb_to_oklab(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    m = a > 0.04045
    a = np.where(m, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
    l = 0.4122214708 * a[..., 0] + 0.5363325363 * a[..., 1] + 0.0514459929 * a[..., 2]
    m2 = 0.2119034982 * a[..., 0] + 0.6806995451 * a[..., 1] + 0.1073969566 * a[..., 2]
    s = 0.0883024619 * a[..., 0] + 0.2817188376 * a[..., 1] + 0.6299787005 * a[..., 2]
    l_, m_, s_ = np.cbrt(l), np.cbrt(m2), np.cbrt(s)
    return np.stack([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ], axis=-1)


def oklab_to_rgb01(lab: np.ndarray) -> np.ndarray:
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * A + 0.2158037573 * B
    m_ = L - 0.1055613458 * A - 0.0638541728 * B
    s_ = L - 0.0894841775 * A - 1.2914855480 * B
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    rgb = np.clip(np.stack([r, g, b], -1), 0, 1)
    msk = rgb > 0.0031308
    out = np.where(msk, 1.055 * np.power(rgb, 1 / 2.4) - 0.055, 12.92 * rgb)
    return np.clip(out, 0, 1)


def kmeans_palette(pix: np.ndarray, k: int, iters: int = 16,
                   seed: int = 7, sample: int = 80000) -> np.ndarray:
    """Weighted k-means in OKLab; returns (k, 3) centroids."""
    rng = np.random.default_rng(seed)
    n = len(pix)
    k = max(2, int(k))
    samp = pix[rng.choice(n, min(sample, n), replace=False)]
    cent = [samp[rng.integers(len(samp))]]
    d2 = ((samp - cent[0]) ** 2).sum(1)
    for _ in range(1, k):
        total = float(d2.sum())
        nxt = (samp[rng.integers(len(samp))] if total <= 0
               else samp[rng.choice(len(samp), p=d2 / total)])
        cent.append(nxt)
        d2 = np.minimum(d2, ((samp - nxt) ** 2).sum(1))
    cent = np.array(cent, dtype=np.float32)
    for _ in range(iters):
        d = (samp ** 2).sum(1)[:, None] - 2 * samp @ cent.T + (cent ** 2).sum(1)[None, :]
        idx = d.argmin(1)
        for j in range(k):
            sel = idx == j
            if sel.any():
                cent[j] = samp[sel].mean(0)
    return cent


_PAL_CACHE: dict = {}


def palette_rgb(cent: np.ndarray) -> np.ndarray:
    return (_srgb_from_oklab_cached(cent) * 255.0 + 0.5).astype(np.uint8)


def _srgb_from_oklab_cached(cent: np.ndarray) -> np.ndarray:
    return oklab_to_rgb01(cent)


def assign(im: Image.Image, cent: np.ndarray, grid: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Map every pixel to its nearest OKLab centroid through a 3D LUT.

    Returns (palette_uint8[k,3], label_index[H,W]).
    """
    key = (id(cent), cent.shape, float(cent.sum()), grid)
    lut = _PAL_CACHE.get(key)
    if lut is None:
        xs = np.linspace(0, 255, grid).astype(np.float32)
        gridpts = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), -1).reshape(-1, 3) / 255.0
        glab = srgb_to_oklab(gridpts)
        lab_lut = np.empty(len(glab), np.int32)
        step = 8000
        for i in range(0, len(glab), step):
            seg = glab[i:i + step]
            d = (seg ** 2).sum(1)[:, None] - 2 * seg @ cent.T + (cent ** 2).sum(1)[None, :]
            lab_lut[i:i + step] = d.argmin(1)
        lut = lab_lut.reshape(grid, grid, grid)
        if len(_PAL_CACHE) > 8:
            _PAL_CACHE.clear()
        _PAL_CACHE[key] = lut

    arr = np.asarray(im.convert("RGB"), np.float32) / 255.0
    q = (arr * (grid - 1)).round().astype(np.int32)
    idx = lut[q[..., 0], q[..., 1], q[..., 2]]
    return palette_rgb(cent), idx
