"""Detail-preserving upscaling ("高清化") used before vector tracing.

The goal is *not* to invent pixels but to make the trace cleaner:

1. Lanczos resize to the target size (keeps all existing detail).
2. Guided-filter base/detail split on luminance - flat areas get smoothed
   (JPEG blocking / noise disappears), texture and lines stay untouched.
3. Edge-modulated detail gain: strong edges are boosted more than flat noise,
   so silhouettes and line art come out crisper.
4. Halo control: the sharpened luminance is clamped to the local min/max of the
   original upscaled luminance, which stops white/black ringing.
5. Light chroma smoothing removes colour fringes along high-contrast borders
   (they are what makes vectorised edges look "dirty").
"""

from __future__ import annotations

import numpy as np
from PIL import Image

MAX_PIXELS = 14_000_000
_MAX_WIDTH = 4096


# --------------------------------------------------------------------------- #
# fast box filter (integral image, edge replicated)
# --------------------------------------------------------------------------- #
def _box(img: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return img
    pad = np.pad(img, ((r, r), (r, r)), mode="edge")
    c = np.cumsum(pad, axis=0, dtype=np.float32)
    c = np.vstack([np.zeros((1, c.shape[1]), np.float32), c])
    out = c[2 * r + 1:, :] - c[:-2 * r - 1, :]
    c = np.cumsum(out, axis=1, dtype=np.float32)
    c = np.hstack([np.zeros((c.shape[0], 1), np.float32), c])
    out = c[:, 2 * r + 1:] - c[:, :-2 * r - 1]
    return out / float((2 * r + 1) ** 2)


def _guided(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    mi = _box(guide, r)
    mp = _box(src, r)
    corr_i = _box(guide * guide, r)
    corr_ip = _box(guide * src, r)
    var_i = corr_i - mi * mi
    cov_ip = corr_ip - mi * mp
    a = cov_ip / (var_i + eps)
    b = mp - a * mi
    return _box(a, r) * guide + _box(b, r)


def _grad_mag(y: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:-1] = (y[:, 2:] - y[:, :-2]) * 0.5
    gy[1:-1, :] = (y[2:, :] - y[:-2, :]) * 0.5
    return np.sqrt(gx * gx + gy * gy)


def _local_range(y: np.ndarray, r: int = 1):
    pad = np.pad(y, ((r, r), (r, r)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, (2 * r + 1, 2 * r + 1))
    return win.min(axis=(-1, -2)), win.max(axis=(-1, -2))


# --------------------------------------------------------------------------- #
# colour helpers (BT.601 luma; chroma kept in 0..1)
# --------------------------------------------------------------------------- #
def _split(arr: np.ndarray):
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    return y, cb, cr


def _merge(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb = cb - 0.5
    cr = cr - 0.5
    r = y + cr * 1.402
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + cb * 1.772
    return np.clip(np.stack([r, g, b], -1), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def target_size(width: int, height: int, scale: float) -> tuple[int, int]:
    tw = int(round(width * scale))
    if tw > _MAX_WIDTH:
        tw = _MAX_WIDTH
    th = max(1, int(round(height * tw / float(width))))
    if tw * th > MAX_PIXELS:  # keep memory / trace time sane
        f = (MAX_PIXELS / float(tw * th)) ** 0.5
        tw, th = max(width, int(tw * f)), max(height, int(th * f))
    return tw, th


def anime_hd(im: Image.Image, scale: float = 2.0, grain: float = 0.5,
             line_gain: float = 1.0) -> Image.Image:
    """Anime-oriented upscale: kill grain, then re-draw crisp linework.

    Follows the recipe used by the Lumina-Vec pipeline (Real-ESRGAN anime
    replaced by a deterministic filter chain, so no model download and no
    invented colours):

    1. shrink to 1/2 to average out JPEG grain / compression noise,
    2. upscale back with cubic interpolation (edges stay smooth, no invented
       colours - every output pixel is a blend of original pixels),
    3. re-darken line pixels: pixels darker than their local background are
       pushed back towards their *original* colour, which restores the sharp
       step that blurring rounded off,
    4. keep chroma from the upscaled image so hue never shifts.
    """
    w, h = im.size
    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale)))
    if tw * th > MAX_PIXELS:
        k = (MAX_PIXELS / float(tw * th)) ** 0.5
        tw, th = max(1, int(tw * k)), max(1, int(th * k))

    # 1. grain removal by 1/2 shrink (area average beats median for noise)
    half = im.resize((max(1, w // 2), max(1, h // 2)), Image.BOX) if grain > 0 else im
    if grain > 0:
        half = Image.blend(im, half.resize(im.size, Image.BICUBIC), 0.55)

    # 2. cubic upscale
    up = half.resize((tw, th), Image.BICUBIC)
    arr = np.asarray(up, np.float32) / 255.0
    y, cb, cr = _split(arr)

    # 3. local background + line re-darkening
    local = _box(y, max(1, int(round(2 * scale))))
    diff = y - local                      # negative = darker than surroundings
    dark = np.clip(-diff / 0.18, 0.0, 1.0)        # 0..1 line strength
    lift = np.clip(diff / 0.18, 0.0, 1.0)         # highlights
    y2 = y + line_gain * (dark * (local - y) * 0.55 - lift * 0.0)

    # 4. gentle unsharp limited to the local range (no white/black ringing)
    blur = _box(y2, 1)
    y3 = y2 + 0.45 * (y2 - blur)
    lo, hi = _local_range(y)
    y3 = np.clip(y3, lo - 0.006, hi + 0.006)

    out = (_merge(np.clip(y3, 0.0, 1.0), cb, cr) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def enhance(im: Image.Image, scale: float = 2.0, strength: float = 0.65,
            denoise: float = 0.5) -> Image.Image:
    """Upscale *im* preserving detail and sharpening edges for cleaner tracing."""
    tw, th = target_size(im.width, im.height, scale)
    up = im.resize((tw, th), Image.LANCZOS)
    if tw <= im.width and th <= im.height:
        return up

    arr = np.asarray(up, dtype=np.float32) / 255.0
    y, cb, cr = _split(arr)

    # base/detail split: eps scales the "flat area" threshold with denoise level
    eps = 0.0004 + 0.0022 * float(np.clip(denoise, 0.0, 1.0))
    base = _guided(y, y, 4, eps)
    detail = y - base

    # stronger gain on real edges, gentle on flat noise
    edge = np.clip(_grad_mag(base) * 6.0, 0.0, 1.0)
    gain = 1.0 + float(strength) * (0.25 + 0.9 * edge)
    y2 = base + detail * gain

    # clamped unsharp: crispens transitions, cannot ring outside the local range
    blur = _box(y2, 1)
    y3 = y2 + (0.35 + 0.5 * float(strength)) * (y2 - blur)
    lo, hi = _local_range(y)
    margin = 0.045 + 0.05 * float(strength)
    y3 = np.clip(y3, np.maximum(lo - 0.004, y - margin),
                 np.minimum(hi + 0.004, y + margin))

    # chroma: light smoothing kills colour fringes on hard borders
    cb2 = 0.35 * cb + 0.65 * _box(cb, 2)
    cr2 = 0.35 * cr + 0.65 * _box(cr, 2)

    out = (_merge(np.clip(y3, 0.0, 1.0), cb2, cr2) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out, "RGB")
