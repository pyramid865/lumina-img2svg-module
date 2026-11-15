"""Cel-shading vectorisation that keeps line art and highlights alive.

The previous version flattened everything, which erased the mouth line and the
pupil. This version separates the picture into three things and treats them
differently:

* **colour regions**  - quantised from a strongly smoothed base, so skin/shirt
  gradients become clean iso-contours instead of mottled patches;
* **line work**       - pixels noticeably darker than the smoothed base (mouth,
  pupil outline, hair strands, panel lines) are re-injected as their own dark
  colour block, so they survive quantisation;
* **speculars**       - very bright, low-chroma pixels (eye catch-lights, metal
  glints) are snapped to the lightest palette entry.

Finally a single majority pass rounds the pixel staircase and vtracer fits
splines, giving smooth Beziers with no colour bleeding.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from color import assign, kmeans_palette, oklab_to_rgb01, palette_rgb, srgb_to_oklab


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
    a = (_box(guide * src, r) - mi * mp) / (_box(guide * guide, r) - mi * mi + eps)
    b = mp - a * mi
    return _box(a, r) * guide + _box(b, r)


def _local_range(y: np.ndarray, r: int = 1):
    pad = np.pad(y, ((r, r), (r, r)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, (2 * r + 1, 2 * r + 1))
    return win.min(axis=(-1, -2)), win.max(axis=(-1, -2))


# --------------------------------------------------------------------------- #
def preprocess(im: Image.Image, denoise: float = 0.45, contrast: float = 1.05,
               sat: float = 1.05, sharpen: float = 0.15) -> np.ndarray:
    """Light denoise + tone lift; keeps thin lines (denoise is deliberately mild)."""
    arr = np.asarray(im.convert("RGB"), np.float32) / 255.0
    lab = srgb_to_oklab(arr)
    L = lab[..., 0]

    base = _guided(L, L, 2, 0.0008 + 0.0015 * float(np.clip(denoise, 0, 1)))
    L = base + (L - base) * (1.0 - 0.45 * float(np.clip(denoise, 0, 1)))

    L = np.clip((L - 0.5) * float(contrast) + 0.5, 0.0, 1.0)
    if sharpen > 0:
        blur = _box(L, 1)
        L2 = L + float(sharpen) * (L - blur)
        lo, hi = _local_range(L)
        L2 = np.clip(L2, np.maximum(lo - 0.004, L - 0.06), np.minimum(hi + 0.004, L + 0.06))
        L = np.clip(L2, 0.0, 1.0)
    lab[..., 0] = L
    lab[..., 1] *= sat
    lab[..., 2] *= sat
    return lab


def _weighted_samples(lab: np.ndarray, count: int = 90000, power: float = 0.5,
                      bins: tuple = (28, 22, 22), seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pix = lab.reshape(-1, 3)
    li = np.clip((pix[:, 0] * bins[0]).astype(np.int32), 0, bins[0] - 1)
    ai = np.clip(((pix[:, 1] + 0.4) / 0.8 * bins[1]).astype(np.int32), 0, bins[1] - 1)
    bi = np.clip(((pix[:, 2] + 0.4) / 0.8 * bins[2]).astype(np.int32), 0, bins[2] - 1)
    key = (li * bins[1] + ai) * bins[2] + bi
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), 3), np.float64)
    np.add.at(sums, inv, pix)
    means = (sums / counts[:, None]).astype(np.float32)
    w = counts.astype(np.float64) ** power
    w /= w.sum()
    pick = rng.choice(len(uniq), size=min(count, 4 * len(uniq)), p=w)
    return means[pick]


def flatten(lab: np.ndarray, radius: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Return (flat_lab, base_L). Regions become smooth, edges stay aligned."""
    L = lab[..., 0]
    base = _guided(L, L, radius, 0.0045)
    cb = _box(lab[..., 1], 3)
    cr = _box(lab[..., 2], 3)
    return np.stack([base, cb, cr], -1), base


def quantize(flat: np.ndarray, colors: int, power: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    samp = _weighted_samples(flat, 90000, power)
    cent = kmeans_palette(samp, colors, iters=20, seed=5, sample=len(samp))
    rgb = Image.fromarray((oklab_to_rgb01(flat) * 255 + 0.5).astype(np.uint8), "RGB")
    pal, labels = assign(rgb, cent)
    return labels, cent


def inject_details(labels: np.ndarray, cent: np.ndarray, lab: np.ndarray,
                   base: np.ndarray, line_thr: float = 0.045,
                   spec_l: float = 0.90) -> np.ndarray:
    """Re-inject line work and speculars that the flat quantiser merged away."""
    L = lab[..., 0]
    detail = L - base
    out = labels.copy()

    # --- line work: darker than the local base, and part of a thin structure
    line = (detail < -line_thr)
    dens = _box(line.astype(np.float32), 1)
    line &= dens > 0.18                     # drop isolated noise pixels
    if line.any():
        order = np.argsort(cent[:, 0])      # darkest first
        dark = order[: max(3, len(order) // 3)]
        pts = lab[line].astype(np.float32)
        sub = cent[dark]
        d = (pts ** 2).sum(1)[:, None] - 2 * pts @ sub.T + (sub ** 2).sum(1)[None, :]
        out[line] = dark[d.argmin(1)]

    # --- speculars: near-white, low chroma
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    spec = (L > spec_l) & (chroma < 0.055) & (detail > 0.02)
    if spec.any():
        white = int(np.argmax(cent[:, 0]))
        out[spec] = white
    return out


def majority(labels: np.ndarray, k: int, iters: int = 1, radius: int = 1) -> np.ndarray:
    """Mode filter - rounds corners, keeps 2px lines (1 iteration only)."""
    cur = labels.astype(np.int32)
    for _ in range(iters):
        best = cur.copy()
        bestv = np.zeros(cur.shape, np.float32)
        for j in range(k):
            dens = _box((cur == j).astype(np.float32), radius)
            mask = dens > bestv
            best = np.where(mask, j, best)
            bestv = np.where(mask, dens, bestv)
        cur = best
    return cur


def _quant_error(flat: np.ndarray, cent: np.ndarray, labels: np.ndarray) -> float:
    d = flat - cent[labels]
    return float(np.sqrt((d ** 2).sum(-1)).mean() * 100.0)


def build(im: Image.Image, colors: int = 128, denoise: float = 0.45,
          contrast: float = 1.02, sat: float = 1.02, sharpen: float = 0.15,
          radius: int = 2, smooth_iters: int = 1, scale: int = 1,
          max_width: int = 1600, power: float = 0.5,
          line_thr: float = 0.03, target_de: float = 1.5,
          max_colors: int = 512) -> tuple[Image.Image, int, np.ndarray]:
    """Colour-block pipeline that *auto-raises* the palette until it matches.

    ``target_de`` is the mean OKLab deltaE (x100) we accept between the flat
    image and the source; if the requested palette is too small the count is
    doubled until the picture matches the original again (cap ``max_colors``).
    """
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    lab = preprocess(im, denoise, contrast, sat, sharpen)
    flat, base = flatten(lab, radius)

    k = max(8, int(colors))
    while True:
        labels, cent = quantize(flat, k, power)
        err = _quant_error(flat, cent, labels)
        if err <= target_de or k >= max_colors:
            break
        k = min(max_colors, k * 2)

    labels = inject_details(labels, cent, lab, base, line_thr=line_thr)
    labels = majority(labels, len(cent), smooth_iters, 1)
    pal = palette_rgb(cent)
    img = Image.fromarray(pal[labels], "RGB")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    used = len(np.unique(labels))
    return img, used, cent


def denoise(im: Image.Image, strength: float = 0.45) -> Image.Image:
    """Light edge-preserving denoise used by the high-fidelity mode."""
    lab = srgb_to_oklab(np.asarray(im.convert("RGB"), np.float32) / 255.0)
    L = lab[..., 0]
    base = _guided(L, L, 2, 0.0010)
    lab[..., 0] = base + (L - base) * (1.0 - 0.55 * float(np.clip(strength, 0, 1)))
    return Image.fromarray((oklab_to_rgb01(lab) * 255 + 0.5).astype(np.uint8), "RGB")


def _shift_filter(a: np.ndarray, r: int, op) -> np.ndarray:
    out = a.copy()
    for d in range(1, r + 1):
        out = op(out, np.roll(a, d, axis=1))
        out = op(out, np.roll(a, -d, axis=1))
    tmp = out.copy()
    for d in range(1, r + 1):
        out = op(out, np.roll(tmp, d, axis=0))
        out = op(out, np.roll(tmp, -d, axis=0))
    return out


def sharpen_lines(im: Image.Image, denoise: float = 0.35, line: float = 0.55,
                  sharp: float = 0.35) -> Image.Image:
    """Denoise + black-hat line boost + clamped unsharp for crisp line work."""
    lab = srgb_to_oklab(np.asarray(im.convert("RGB"), np.float32) / 255.0)
    L = lab[..., 0]
    base = _guided(L, L, 2, 0.001)
    L = base + (L - base) * (1.0 - 0.5 * float(np.clip(denoise, 0, 1)))

    # black-hat: dark structures thinner than ~5px (mouth line, lashes, panel lines)
    closing = _shift_filter(_shift_filter(L, 2, np.maximum), 2, np.minimum)
    blackhat = np.clip(closing - L, 0.0, 1.0)
    L = np.clip(L - float(line) * blackhat, 0.0, 1.0)

    if sharp > 0:
        blur = _box(L, 1)
        L2 = L + float(sharp) * (L - blur)
        lo, hi = _local_range(L)
        L2 = np.clip(L2, np.maximum(lo - 0.004, L - 0.07), np.minimum(hi + 0.004, L + 0.07))
        L = np.clip(L2, 0.0, 1.0)
    lab[..., 0] = L
    return Image.fromarray((oklab_to_rgb01(lab) * 255 + 0.5).astype(np.uint8), "RGB")


def build_print(im: Image.Image, colors: int = 12, hd_scale: float = 2.0,
                min_feature: int = 2, denoise: float = 0.55,
                max_width: int = 2000) -> tuple[Image.Image, int, np.ndarray]:
    """Palette for layered (lumina) 3D printing.

    Regions are flattened harder than the on-screen modes so the slicer gets
    solid printable blocks instead of dithery islands; thin ink lines are kept
    through the line mask only when they are wide enough to print.
    """
    from enhance import enhance as _enh

    if hd_scale and hd_scale != 1.0:
        im = _enh(im, scale=hd_scale, strength=0.45)
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)

    lab = preprocess(im, denoise=denoise, contrast=1.0, sat=1.02, sharpen=0.0)
    flat, base = flatten(lab, 6)
    labels, cent = quantize(flat, colors, power=0.6)

    # printable cleanup: median kills hair-thin islands, majority rounds corners
    pal = palette_rgb(cent)
    img = Image.fromarray(pal[labels], "RGB")
    if min_feature >= 2:
        img = img.filter(ImageFilter.MedianFilter(5 if min_feature >= 3 else 3))
        _, labels = assign(img, cent)
    labels = majority(labels, len(cent), 2, 2)
    pal = palette_rgb(cent)
    img = Image.fromarray(pal[labels], "RGB")
    return img, len(np.unique(labels)), cent


def ref_style(im: Image.Image, colors: int = 128, grid: int = 1600,
              median: int = 3) -> tuple[Image.Image, int]:
    """Look of 家教纲吉.svg: median-flattened colours at 1600 grid + 128 palette.

    The median pass is what removes the JPEG anti-alias ramps so region edges
    become hard steps (crisp in print) instead of soft gradients.
    """
    from PIL import Image as _I, ImageFilter as _IF
    im = im.resize((grid, round(im.height * grid / im.width)), _I.LANCZOS)
    if median >= 3:
        im = im.filter(_IF.MedianFilter(median))
    q = im.quantize(colors=colors, method=_I.Quantize.MEDIANCUT,
                    dither=_I.Dither.NONE).convert("RGB")
    return q, len(q.getcolors(maxcolors=1 << 20) or [])
