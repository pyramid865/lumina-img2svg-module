"""Layered colour tracer (self-developed).

Pipeline
--------
1. OKLab k-means palette, area-weighted sampling
2. per-colour indicator blur + argmax  -> mode-like smoothing
3. inscribed-half-width rule            -> thin anti-alias bands merged away
4. stacked painting: a layer's shape is itself plus every *later* layer, so
   each border is owned by exactly one shape and nothing is left uncovered
5. sub-pixel iso-contour (0.5) of the soft mask, fitted with cubic Beziers
   (least squares + recursive splitting, corners kept sharp)

The shape maths lives here so it can be tuned without touching detection code.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from skimage import measure
except Exception:  # pragma: no cover
    measure = None


# ------------------------------------------------------------------ colour
def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, np.float64)
    l = 0.4122214708 * c[..., 0] + 0.5363325363 * c[..., 1] + 0.0514459929 * c[..., 2]
    m = 0.2119034982 * c[..., 0] + 0.6806995451 * c[..., 1] + 0.1073969566 * c[..., 2]
    s = 0.0883024619 * c[..., 0] + 0.2817188376 * c[..., 1] + 0.6299787005 * c[..., 2]
    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)
    return np.stack([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ], axis=-1)


def _oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    l_ = lab[..., 0] + 0.3963377774 * lab[..., 1] + 0.2158037573 * lab[..., 2]
    m_ = lab[..., 0] - 0.1055613458 * lab[..., 1] - 0.0638541728 * lab[..., 2]
    s_ = lab[..., 0] - 0.0894841775 * lab[..., 1] - 1.2914855480 * lab[..., 2]
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    out = np.stack([
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ], axis=-1)
    return out


def build_palette(rgb: np.ndarray, colors: int, iters: int = 18, seed: int = 7) -> np.ndarray:
    """Area-weighted OKLab k-means. Returns uint8 palette (k,3)."""
    h, w = rgb.shape[:2]
    step = max(1, int(np.sqrt(h * w / 150000)))
    sample = rgb[::step, ::step].reshape(-1, 3).astype(np.float64) / 255.0
    lab = _srgb_to_oklab(sample).astype(np.float32)
    if len(lab) == 0:
        return np.zeros((1, 3), np.uint8)

    rng = np.random.default_rng(seed)
    cent = lab[rng.choice(len(lab), min(colors, len(lab)), replace=False)].copy()
    for _ in range(iters):
        d = ((lab[:, None, :] - cent[None, :, :]) ** 2).sum(2)
        assign = d.argmin(1)
        for k in range(len(cent)):
            sel = assign == k
            if sel.any():
                cent[k] = lab[sel].mean(0)
    srgb = np.clip(_oklab_to_srgb(cent.astype(np.float64)), 0, 1)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def build_palette_bands(rgb: np.ndarray, colors: int, bands: int = 6,
                        seed: int = 7) -> tuple:
    """Nested palette: lightness bands first, colour clustering inside each.

    Free k-means produces scattered layers (a dark tone may sit above a light
    one), so an accumulated "underlay" mask covers the wrong areas. Splitting by
    lightness first makes the layers nest, which is what underlay reconstruction
    needs. Returns (palette uint8 (k,3), band index per colour).
    """
    h, w = rgb.shape[:2]
    step = max(1, int(np.sqrt(h * w / 150000)))
    sample = rgb[::step, ::step].reshape(-1, 3).astype(np.float64) / 255.0
    if len(sample) == 0:
        return np.zeros((1, 3), np.uint8), np.zeros(1, np.int32)
    lab = _srgb_to_oklab(sample).astype(np.float32)
    L = lab[:, 0]

    bands = int(max(2, min(bands, colors // 2)))
    per = max(1, colors // bands)
    edges = np.quantile(L, np.linspace(0.0, 1.0, bands + 1))
    edges[0], edges[-1] = -1.0, 2.0

    rng = np.random.default_rng(seed)
    palette_lab = []
    band_of = []
    for b in range(bands):
        sel = (L >= edges[b]) & (L < edges[b + 1])
        part = lab[sel]
        if len(part) == 0:
            continue
        k = int(min(per, len(part)))
        cent = part[rng.choice(len(part), k, replace=False)].copy()
        for _ in range(14):
            d = ((part[:, None, :] - cent[None, :, :]) ** 2).sum(2)
            assign = d.argmin(1)
            for c in range(k):
                m = assign == c
                if m.any():
                    cent[c] = part[m].mean(0)
        palette_lab.extend(cent)
        band_of.extend([b] * k)

    if not palette_lab:
        return np.zeros((1, 3), np.uint8), np.zeros(1, np.int32)
    srgb = np.clip(_oklab_to_srgb(np.asarray(palette_lab, np.float64)), 0, 1)
    return (srgb * 255.0 + 0.5).astype(np.uint8), np.asarray(band_of, np.int32)


def assign_palette(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Nearest palette entry per pixel, in OKLab distance."""
    h, w = rgb.shape[:2]
    lab = _srgb_to_oklab(rgb.reshape(-1, 3).astype(np.float64) / 255.0).astype(np.float32)
    pal = _srgb_to_oklab(palette.astype(np.float64) / 255.0).astype(np.float32)
    out = np.empty(len(lab), np.int32)
    step = 400000
    for s in range(0, len(lab), step):
        chunk = lab[s:s + step]
        out[s:s + step] = ((chunk[:, None, :] - pal[None, :, :]) ** 2).sum(2).argmin(1)
    return out.reshape(h, w)


# ------------------------------------------------------------- smoothing
def unmix_edges(rgb: np.ndarray, min_delta: float = 26.0,
                band: tuple = (0.12, 0.88), passes: int = 1) -> np.ndarray:
    """Snap anti-alias pixels back to a real neighbouring colour.

    A pixel sitting on the straight line between its two opposite neighbours is
    a raster blend, not a tone. Blurring such blends away also erases genuine
    texture, so they are detected and replaced with the *actual* nearer
    neighbour instead - no invented colours.
    """
    out = rgb.astype(np.int16)
    h, w = out.shape[:2]
    for _ in range(max(1, passes)):
        cur = out.astype(np.float32)
        for axis in (1, 0):
            prev = np.roll(cur, 1, axis)
            nxt = np.roll(cur, -1, axis)
            span = nxt - prev
            denom = (span ** 2).sum(2)
            v = cur - prev
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(denom > 1e-6, (v * span).sum(2) / denom, 0.5)
            pred = prev + t[..., None] * span
            resid = np.sqrt(((cur - pred) ** 2).sum(2))
            strength = np.sqrt(denom)
            blend = (strength > min_delta) & (t > band[0]) & (t < band[1]) & (resid < min_delta * 0.5)
            if not blend.any():
                continue
            near_nxt = t >= 0.5
            src = np.where(near_nxt[..., None], nxt, prev)
            out[blend] = src[blend].astype(np.int16)
        cur = out.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def smooth_labels(labels: np.ndarray, sigma: float = 1.0, iters: int = 1) -> np.ndarray:
    """Mode-like filter: blur every colour's indicator, take the argmax.

    A plain per-pixel vote can still leave staircase edges; blurring the
    indicators gives the boundary a smooth, roughly equidistant position.
    """
    cur = labels.copy()
    uniq = np.unique(cur)
    if len(uniq) > 256 or sigma <= 0:
        return cur
    for _ in range(max(1, iters)):
        best = np.full(cur.shape, -1.0, np.float32)
        arg = cur.copy()
        for c in np.unique(cur):
            m = (cur == c).astype(np.float32)
            s = cv2.GaussianBlur(m, (0, 0), sigma)
            upd = s > best
            best[upd] = s[upd]
            arg[upd] = c
        cur = arg
    return cur


def repair_ink_gaps(rgb: np.ndarray, labels: np.ndarray,
                    threshold: float = 32.0, close: int = 3,
                    dark_luma: float = 120.0) -> tuple:
    """EXPERIMENTAL: fill the light 1-3 px gaps that break thin dark strokes.

    A black-hat responds to thin structures darker than their surroundings
    (hair separators, eyelashes, contour ink); large dark areas barely
    respond.  Inside that mask, pixels currently assigned a LIGHT label are
    the gaps that show up as white dots.  They take the nearest genuinely
    dark label, so the dashes join into a line.

    Nothing is removed from any layer - an earlier version that forced all
    ink pixels into one new label punched holes out of neighbouring layers
    and the white canvas showed through as cracks (local dE 8.67 -> 20.82).
    """
    from scipy import ndimage as ndi

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    ink = ((blackhat > threshold) & (gray < 150)).astype(np.uint8)
    if close > 1:
        ink = cv2.morphologyEx(
            ink, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    if not ink.any():
        return labels, 0

    n = int(labels.max()) + 1
    flat = labels.ravel()
    cnt = np.bincount(flat, minlength=n).astype(np.float64)
    lum_src = (rgb.astype(np.float64) *
               np.array([0.2126, 0.7152, 0.0722])).sum(2)
    lum_sum = np.bincount(flat, weights=lum_src.ravel(), minlength=n)
    mean_lum = lum_sum / np.maximum(cnt, 1.0)
    darkmap = (mean_lum < dark_luma)[labels]
    if not darkmap.any():
        return labels, 0

    dist, (iy, ix) = ndi.distance_transform_edt(~darkmap, return_indices=True)
    target = labels[iy, ix]
    gaps = ink.astype(bool) & ~darkmap & (dist <= 3.0)
    out = labels.copy()
    out[gaps] = target[gaps]
    return out, int(gaps.sum())


def unify_ink_cc(rgb: np.ndarray, labels: np.ndarray,
                 threshold: float = 28.0, close: int = 3,
                 dark_luma: float = 140.0, width_thresh: float = 3.0,
                 dark_margin: float = 28.0, max_new_ratio: float = 0.18) -> tuple:
    """Unify a fragmented thin dark stroke into clean continuous top layers.

    Why the earlier "one global ink label" failed: it painted every detected
    dark structure the same median colour, flattening the difference between
    (say) a brown hair and a black outline -> detail + dE regressed.

    This version keeps colour fidelity AND stays selective:
      1. black-hat finds THIN dark structures;
      2. the mask is closed to bridge 1-3 px gaps;
      3. each *connected component* is accepted only if it is genuinely an
         ink line - its mean luma must sit at least ``dark_margin`` below the
         mean luma of the ring just outside it.  That rule skips ordinary
         colour-block borders (where both sides are similar) and only catches
         dark lines on a lighter ground;
      4. accepted components become their OWN new top label, filled with that
         component's own mean colour, so a brown strand and a black outline
         stay distinct;
      5. width gate (2*area/perimeter < width_thresh) skips fat blobs;
      6. small area -> painted last -> sits on top, no uncovered gaps.
    """
    from scipy import ndimage as ndi

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    ink = ((blackhat > threshold) & (gray < dark_luma)).astype(np.uint8)
    if close > 1:
        ink = cv2.morphologyEx(
            ink, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    if not ink.any() or ink.mean() > max_new_ratio:
        return labels, []

    comps, ncomp = ndi.label(ink)
    out = labels.copy()
    new_id = int(labels.max()) + 1
    created = []
    ring = ndi.binary_dilation(ink) & ~ink          # the neighbourhood just outside
    ring_vals = gray[ring]
    for c in range(1, ncomp + 1):
        m = comps == c
        area = int(m.sum())
        if area < 3:
            continue
        perim = int((ndi.binary_dilation(m) & ~m).sum())  # boundary pixels
        width = 2.0 * area / max(perim, 1)
        if width >= width_thresh:        # fat blob, leave it alone
            continue
        comp_mean = float(gray[m].mean())
        surround = ring & (comps == 0)   # ring pixels not belonging to any comp
        # approximate the local ground by the ink mask's outer ring
        outer = ndi.binary_dilation(m) & ~m
        if outer.any():
            ground = float(gray[outer].mean())
        else:
            ground = 255.0
        if ground - comp_mean < dark_margin:   # not a dark line on light ground
            continue
        out[m] = new_id
        created.append(new_id)
        new_id += 1
    if not created:
        return labels, []
    return out, created


def merge_thin(labels: np.ndarray, palette: np.ndarray,
               max_half_width: float = 1.2, keep_contrast: float = 0.10) -> np.ndarray:
    """Thin strips (anti-alias bands) take the nearest thick colour.

    Strips that are clearly darker than their surroundings are real line art,
    so they survive.
    """
    from scipy import ndimage as ndi

    h, w = labels.shape
    edge = np.zeros((h, w), bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:-1] |= labels[1:] != labels[:-1]

    dt = ndi.distance_transform_edt(~edge) + 0.5
    flat = labels.ravel()
    n = int(flat.max()) + 1
    hw = ndi.maximum(dt, labels, index=np.arange(n))
    hw = np.nan_to_num(np.asarray(hw, np.float64), nan=99.0)
    thin = hw[labels] <= max_half_width
    if not thin.any():
        return labels

    _, (iy, ix) = ndi.distance_transform_edt(thin, return_indices=True)
    repl = labels[iy, ix]

    lab_pal = _srgb_to_oklab(palette.astype(np.float64) / 255.0)
    lum = lab_pal[:, 0]
    delta = lum[repl] - lum[labels]
    thin &= delta < keep_contrast          # keep genuinely dark line art

    out = labels.copy()
    out[thin] = repl[thin]
    return out


# ------------------------------------------------------------ curve fitting
def _cubic(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[:, None]
    mt = 1.0 - t
    return (mt ** 3) * ctrl[0] + 3 * (mt ** 2) * t * ctrl[1] + 3 * mt * (t ** 2) * ctrl[2] + (t ** 3) * ctrl[3]


def _chord_params(pts: np.ndarray) -> np.ndarray:
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if d[-1] <= 1e-12:
        return np.linspace(0.0, 1.0, len(pts))
    return d / d[-1]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-12 else v


def _fit_segment(pts: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Least-squares cubic through pts with fixed end tangents."""
    u = _chord_params(pts)
    a1 = np.outer(3 * (1 - u) ** 2 * u, t1)
    a2 = np.outer(3 * (1 - u) * u ** 2, t2)
    c00 = float((a1 * a1).sum())
    c01 = float((a1 * a2).sum())
    c11 = float((a2 * a2).sum())
    b0 = (1 - u) ** 3 + 3 * (1 - u) ** 2 * u
    b1 = 3 * (1 - u) * u ** 2 + u ** 3
    rest = pts - np.outer(b0, pts[0]) - np.outer(b1, pts[-1])
    x0 = float((a1 * rest).sum())
    x1 = float((a2 * rest).sum())
    det = c00 * c11 - c01 * c01
    chord = float(np.hypot(*(pts[-1] - pts[0])))
    if abs(det) > 1e-12:
        al1 = (x0 * c11 - x1 * c01) / det
        al2 = (c00 * x1 - c01 * x0) / det
    else:
        al1 = al2 = chord / 3.0
    if not (1e-6 * chord < al1 < 3 * chord) or not (1e-6 * chord < al2 < 3 * chord):
        al1 = al2 = chord / 3.0
    return np.array([pts[0], pts[0] + t1 * al1, pts[-1] + t2 * al2, pts[-1]])


def _max_error(pts: np.ndarray, ctrl: np.ndarray) -> tuple[float, int]:
    u = _chord_params(pts)
    err = np.hypot(*(_cubic(ctrl, u) - pts).T)
    i = int(err.argmax())
    return float(err[i]), i


def fit_cubics(pts: np.ndarray, t1: np.ndarray, t2: np.ndarray,
               tol: float, depth: int = 0) -> list:
    """Recursively fit cubics until every segment is within tol."""
    if len(pts) <= 2 or depth > 12:
        seg = float(np.hypot(*(pts[-1] - pts[0]))) / 3.0
        return [np.array([pts[0], pts[0] + t1 * seg, pts[-1] + t2 * seg, pts[-1]])]

    ctrl = _fit_segment(pts, t1, t2)
    err, split = _max_error(pts, ctrl)
    if err < tol:
        return [ctrl]
    split = min(max(split, 1), len(pts) - 2)
    mid_t = _unit(pts[split - 1] - pts[split + 1])
    left = fit_cubics(pts[:split + 1], t1, mid_t, tol, depth + 1)
    right = fit_cubics(pts[split:], -mid_t, t2, tol, depth + 1)
    return left + right


def _corner_indices(pts: np.ndarray, angle_deg: float = 55.0, k: int = 3) -> list:
    n = len(pts)
    if n < 2 * k + 3:
        return []
    a = pts[(np.arange(n) - k) % n] - pts
    b = pts[(np.arange(n) + k) % n] - pts
    cos = (a * b).sum(1) / np.maximum(np.hypot(*a.T) * np.hypot(*b.T), 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    sharp = ang < (180.0 - angle_deg)
    idx: list[int] = []
    for i in np.flatnonzero(sharp):
        if ang[i] <= ang[(i - 1) % n] and ang[i] <= ang[(i + 1) % n]:
            if not idx or i - idx[-1] > k:
                idx.append(int(i))
    return idx


def _resample(pts: np.ndarray, step: float = 1.0) -> np.ndarray:
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if d[-1] < step * 3:
        return pts
    s = np.arange(0.0, d[-1], step)
    return np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])]


def fit_closed(poly: np.ndarray, tol: float = 0.6, corner_deg: float = 55.0) -> list:
    """Closed polyline -> list of cubic control arrays."""
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    p = _resample(poly, 1.0)
    if len(p) < 4:
        return []

    pad = np.r_[p[-2:], p, p[:2]]
    smooth = (pad[:-4] + 2 * pad[1:-3] + 3 * pad[2:-2] + 2 * pad[3:-1] + pad[4:]) / 9.0
    corners = _corner_indices(p, corner_deg)
    if corners:
        smooth[corners] = p[corners]
    p = smooth
    n = len(p)

    if not corners:
        ring = np.r_[p, p[:1]]
        t = _unit(p[1] - p[-1])
        return fit_cubics(ring, t, -t, tol)

    segs = []
    for j, start in enumerate(corners):
        end = corners[(j + 1) % len(corners)]
        idx = np.arange(start, end + (n if end <= start else 0) + 1) % n
        q = p[idx]
        if len(q) < 2:
            continue
        t1 = _unit(q[min(2, len(q) - 1)] - q[0])
        t2 = _unit(q[max(-3, -len(q))] - q[-1])
        segs += fit_cubics(q, t1, t2, tol)
    return segs


# ---------------------------------------------------------------- tracing
def _iso_regions(soft: np.ndarray, scale: int, sigma: float, level: float = 0.45):
    """Contours of a soft mask, with the crop border replicated.

    Without padding, find_contours runs a contour along the crop edge and the
    fit closes it back on itself - that is what produced the stair-stepped white
    blobs along layer borders.
    """
    p = cv2.copyMakeBorder(soft.astype(np.float32), 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    f = cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    f = cv2.GaussianBlur(f, (0, 0), sigma * scale)
    out = []
    for c in measure.find_contours(f, level):
        xy = (c[:, ::-1] + 0.5) / scale - 2.0
        out.append(xy)
    return out


def trace(rgb: np.ndarray, colors: int = 160, detail: int = 6,
          sigma: float = 0.7, thin: float = 0.0, grow: int = 1,
          pre_smooth: bool = True, underlay: bool = False,
          nested: bool = False, bands: int = 6, report=None,
          ink: bool = True, ink_threshold: float = 28.0,
          ink_unify: bool = False, ink_cc: bool = True, ink_close: int = 3,
          ink_dark_luma: float = 140.0, ink_max_ratio: float = 0.18,
          ink_width_thresh: float = 3.0, ink_dark_margin: float = 28.0,
          ink_skip_grow: bool = True) -> dict:
    """Raster -> SVG. Returns {svg, colors, paths, layers, ms, stats}.

    ``ink``               repair thin dark strokes (experimental)
    ``ink_unify``         REPLACE the whole thin stroke with one top ink layer
                          (kills the per-pixel fragmentation that makes hair /
                          eye lines render as dotted dashes).  Safer than the
                          old "force-all-ink-to-one-label" because it is gated
                          by local darkness AND capped by ``ink_max_ratio`` so
                          it can never flatten large shadow areas.
    Everything else is byte-for-byte the 2026-11-05 baseline.
    """
    import time

    t0 = time.time()
    if cv2 is None or measure is None:
        raise RuntimeError("需要 opencv 与 scikit-image")

    from scipy import ndimage as ndi

    h, w = rgb.shape[:2]
    # edge-preserving pre-smoothing: the 2-3 px anti-alias band between two
    # fills otherwise becomes its own near-white layer, which traced into the
    # stair-stepped white blobs along borders (LAB delta 11.6 -> 5.8)
    work = cv2.bilateralFilter(rgb, 7, 30, 7) if pre_smooth else rgb
    band_of = None
    if nested:
        palette, band_of = build_palette_bands(work, colors, bands=bands)
    else:
        palette = build_palette(work, colors)
    labels = assign_palette(work, palette)
    labels = smooth_labels(labels, sigma=sigma, iters=1)
    ink_pixels = 0
    ink_ids = []
    if ink:
        if ink_cc:
            labels, ink_ids = unify_ink_cc(
                rgb, labels, threshold=ink_threshold, close=ink_close,
                dark_luma=ink_dark_luma, width_thresh=ink_width_thresh,
                dark_margin=ink_dark_margin, max_new_ratio=ink_max_ratio)
            ink_pixels = int(np.isin(labels, ink_ids).sum()) if ink_ids else 0
        elif ink_unify:
            labels, ink_ids = unify_ink(
                rgb, labels, threshold=ink_threshold, close=ink_close,
                dark_luma=ink_dark_luma, max_new_ratio=ink_max_ratio)
            ink_pixels = int(np.isin(labels, ink_ids).sum()) if ink_ids else 0
        else:
            labels, ink_pixels = repair_ink_gaps(
                rgb, labels, threshold=ink_threshold)

    if thin > 0:
        labels = merge_thin(labels, palette, max_half_width=thin, keep_contrast=0.10)

    ink_id_set = set(int(x) for x in ink_ids) if ink_ids else set()
    counts = np.bincount(labels.ravel())
    if nested and band_of is not None:
        _key = {int(c): (-int(band_of[c]) if c < len(band_of) else 0, -int(counts[c]))
                for c in range(len(counts))}
        order = sorted([int(i) for i in np.nonzero(counts)[0]], key=lambda c: _key[c])
    else:
        order = [int(i) for i in np.argsort(-counts) if counts[i] > 0]
    rank = {c: i for i, c in enumerate(order)}
    rank_img = np.full((h, w), len(order), np.int32)
    for c, r in rank.items():
        rank_img[labels == c] = r

    tol = max(0.9, 2.2 - detail * 0.12)
    min_area = max(2.0, 24.0 - detail * 2.2)
    pad = 3

    # the canvas is primed with the image's dominant colour: with stacking the
    # biggest layer is NOT necessarily the background, so priming with it tinted
    # the whole picture (measured LAB delta 40)
    small = cv2.resize(rgb, (48, 48), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    q = (small // 16).astype(np.int32)
    key = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    vals, freq = np.unique(key, return_counts=True)
    bg_colour = np.median(small[key == vals[freq.argmax()]], axis=0).astype(int)
    paths: list[str] = [
        '<path d="M0,0H%dV%dH0Z" fill="#%02x%02x%02x"/>'
        % (w, h, bg_colour[0], bg_colour[1], bg_colour[2])
    ]
    total = 0
    failed = 0
    for i, colour in enumerate(order):
        own = labels == colour
        ys, xs = np.nonzero(own)
        if len(ys) == 0:
            continue
        # each layer only needs its own neighbourhood: everything outside the
        # bounding box is painted the same way it was before this layer
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(h, int(ys.max()) + 1 + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(w, int(xs.max()) + 1 + pad)
        # each layer paints exactly its own region; paint order (big -> small)
        # decides what ends up visible, and the same-colour stroke closes the
        # hairline gap against whatever was painted before
        # underlay reconstruction (experimental, off by default).
        # Measured on 400x494 / 24 colours: free k-means 34.6 -> 61.1,
        # lightness-band palette 34.6 -> 80.3. The accumulated blob only stays
        # correct when layers are nested by *spatial containment*, which neither
        # k-means nor lightness bands provide. Needs a different quantiser.
        mask = own[y0:y1, x0:x1]
        if underlay:
            _sub_rank = rank_img[y0:y1, x0:x1]
            mask = mask | (_sub_rank > i)
            if i > 0:
                _lab, _ = ndi.label(mask)
                _ids = np.unique(_lab[own[y0:y1, x0:x1]])
                mask = np.isin(_lab, _ids[_ids > 0])
        # equivalent width 2*area/perimeter: thin strokes must not be grown (it
        # fattens and deforms them) and need a stricter fit tolerance
        _area = float(mask.sum())
        _perim = float(np.count_nonzero(mask[:, :-1] != mask[:, 1:]) +
                       np.count_nonzero(mask[:-1, :] != mask[1:, :]) + 1.0)
        thickness = 2.0 * _area / max(1.0, _perim)
        # the ink layers created by unify_ink_cc are already thin continuous
        # strokes; dilating them would fatten the lines and darken the picture
        # (local dE regressed 8.67 -> 9.20).  Skip grow for them - the 0.5
        # same-colour stroke still closes sub-pixel seams.
        grow_here = 0 if (ink_skip_grow and ink_id_set and colour in ink_id_set) else grow
        if grow_here > 0:
            # each layer overhangs its own border by `grow` px; later (smaller)
            # layers paint on top, so their overhang covers the seam instead of
            # relying on a thick stroke that also fattens thin details
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_here * 2 + 1, grow_here * 2 + 1))
            mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)

        if True:
            soft = mask.astype(np.float32)
            parts = []
            for xy in _iso_regions(soft, 2, 0.5):
                if len(xy) < 8:
                    continue
                xy = xy + np.array([x0, y0])
                area = 0.5 * abs(np.dot(xy[:-1, 0], xy[1:, 1]) - np.dot(xy[1:, 0], xy[:-1, 1]))
                if area < min_area:
                    continue
                segs = fit_closed(xy, tol=tol)
                if not segs:
                    continue
                d_ = ["M%.2f,%.2f" % tuple(segs[0][0])]
                for sg in segs:
                    d_.append("C%.2f,%.2f %.2f,%.2f %.2f,%.2f" % (*sg[1], *sg[2], *sg[3]))
                parts.append("".join(d_) + "Z")
            if not parts:
                failed += 1
                continue
            d = "".join(parts)
            total += len(parts)

        # fill with the region's real average colour: the palette entry is a
        # cluster centre and drifts, the mean of the region does not
        if own.any():
            mean = rgb[own].mean(0)
            r, g, b = (int(round(v)) for v in mean)
        else:
            r, g, b = (int(v) for v in palette[colour])
        hexc = "#%02x%02x%02x" % (r, g, b)
        # same-colour stroke: neighbouring layers are fitted independently, so a
        # sub-pixel seam would otherwise let the primed canvas show through
        paths.append('<path d="%s" fill="%s" stroke="%s" stroke-width="0.5" fill-rule="evenodd"/>'
                     % (d, hexc, hexc))
        if report and i % 40 == 0:
            report(i, len(order))

    body = "\n".join(paths)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="%d" height="%d" '
           'viewBox="0 0 %d %d">\n%s\n</svg>' % (w, h, w, h, body))
    return {"svg": svg, "colors": len(order), "paths": len(paths),
            "layers": len(paths), "ms": int((time.time() - t0) * 1000),
            "failed": failed, "stats": {"ink_pixels": ink_pixels}}
