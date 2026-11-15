"""Auto image-type detection -> engine preset selection.

Pure ``numpy`` + ``PIL`` (no cv2 / sklearn / scipy).  Cheap perceptual features
are extracted from a downscaled RGB image, the image is classified with a small
decision tree, and a tuned :class:`core.Settings` preset is returned for the best
match so the user never has to pick an engine by hand.

Categories (human labels, shown in the GUI):
    line_art      线稿 / 漫画
    flat_icon     图标 / 平面 / Logo
    anime         动漫插画 / 人物
    architecture  建筑 / 风景
    photo         写实照片
    texture       纹理 / 材质

The reference SVGs the user supplied map onto these outputs:
    ``示例-160色高清`` / ``示例-自研引擎``  -> anime / photo (faithful, high colour)
    ``示例-纹理模式``                       -> texture (fine contours keep detail)
    ``示例-新图``                           -> general faithful
so the routing favours the faithful ``qq3d`` engine for photographic / illustrated
content and switches to line-art / flat / straight-edge engines only when the
features clearly say so.

Robustness notes (why two colour-count features exist):
    * ``n_colors``  = fine 5-bit histogram count (cap 400).  Used for the photo
      rule (many distinct colours => photographic).
    * ``n_regions`` = area-weighted k-means (k=24) dominant-colour count.  Used for
      flat-icon / line-art / architecture because it MERGES the anti-aliased
      gradient rings at shape edges (each < 0.5% coverage) back into the parent
      colour, so a 3-colour logo reads as ~3 regions, not ~40.
    * ``ink_or_paper`` = fraction of pixels that are near-white OR near-black.
      Thin line-art lines survive downscaling as "deviation from paper", so this
      bimodality cue is far more stable than raw dark-pixel count for detecting
      line drawings.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import core


# --------------------------------------------------------------------------- #
# feature extraction
# --------------------------------------------------------------------------- #
def _load_rgb(src) -> np.ndarray:
    """Accept a path or a PIL image; always return an RGB uint8 array."""
    if isinstance(src, (str, bytes)):
        from core import load_rgb as _lr  # local to avoid import surprise
        im = _lr(str(src))
    else:
        im = src.convert("RGB") if hasattr(src, "convert") else src
    return np.asarray(im, dtype=np.uint8)


def _downscale(arr: np.ndarray, long_edge: int = 256) -> np.ndarray:
    h, w = arr.shape[:2]
    le = max(h, w)
    if le <= long_edge:
        return arr
    scale = long_edge / le
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    im = Image.fromarray(arr).resize((nw, nh), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _box_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Separable mean blur via cumulative sums; k = kernel size."""
    kh, kw = x.shape
    cs = np.cumsum(np.cumsum(x.astype(np.float64), axis=0), axis=1)
    out = np.zeros_like(cs)
    out[k - 1:, k - 1:] += cs[k - 1:, k - 1:]
    out[:kh - (k - 1), k - 1:] -= cs[:kh - (k - 1), k - 1:]
    out[k - 1:, :kw - (k - 1)] -= cs[k - 1:, :kw - (k - 1)]
    out[:kh - (k - 1), :kw - (k - 1)] += cs[:kh - (k - 1), :kw - (k - 1)]
    return out / (k * k)


def _dominant_count(rgb: np.ndarray, k: int = 24, frac: float = 0.005) -> int:
    """Area-weighted k-means dominant-colour count (merges anti-alias rings)."""
    h, w = rgb.shape[:2]
    n = h * w
    flat = rgb.reshape(-1, 3).astype(np.float32)
    samp = flat
    rng = np.random.default_rng(7)
    if n > 20000:
        samp = flat[rng.integers(n, size=20000)]
    # k-means++ seeding
    cent = [samp[rng.integers(len(samp))]]
    d2 = ((samp - cent[0]) ** 2).sum(1)
    for _ in range(1, k):
        t = float(d2.sum())
        nxt = samp[rng.integers(len(samp))] if t <= 0 else samp[rng.choice(len(samp), p=d2 / t)]
        cent.append(nxt)
        d2 = np.minimum(d2, ((samp - nxt) ** 2).sum(1))
    cent = np.array(cent, np.float32)
    for _ in range(12):
        d = (samp ** 2).sum(1)[:, None] - 2 * samp @ cent.T + (cent ** 2).sum(1)[None, :]
        lab = d.argmin(1)
        for j in range(k):
            sel = lab == j
            if sel.any():
                cent[j] = samp[sel].mean(0)
    # assign every pixel, count clusters by area
    d = (flat ** 2).sum(1)[:, None] - 2 * flat @ cent.T + (cent ** 2).sum(1)[None, :]
    lab = d.argmin(1)
    counts = np.bincount(lab, minlength=k)
    fracs = counts / float(counts.sum())
    return int((fracs > frac).sum())


def _thin_frac_native(arr: np.ndarray, work_edge: int = 1024) -> float:
    """Hairline share measured near native size (downsampling erases 1-px rules)."""
    a = arr
    h, w = a.shape[:2]
    le = max(h, w)
    if le > work_edge:
        scale = work_edge / le
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        a = np.asarray(Image.fromarray(a).resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    lum = (0.299 * a[..., 0].astype(np.float32) + 0.587 * a[..., 1].astype(np.float32)
           + 0.114 * a[..., 2].astype(np.float32))
    dark = lum < 90.0
    if not dark.any():
        return 0.0
    light = ~dark
    nb = np.zeros(dark.shape, np.int8)
    nb[:, 1:] += light[:, :-1]
    nb[:, :-1] += light[:, 1:]
    nb[1:, :] += light[:-1, :]
    nb[:-1, :] += light[1:, :]
    return float((dark & (nb >= 2)).sum()) / max(1, int(dark.sum()))


def _features(arr: np.ndarray) -> dict:
    """Return a dict of normalised perceptual features for classification."""
    rgb = _downscale(arr, 256)
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)

    # --- saturation / grayscale -------------------------------------------- #
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / 255.0
    mean_sat = float(sat.mean())
    is_gray = mean_sat < 0.06

    # --- luminance + ink / paper ------------------------------------------- #
    lum = (0.299 * r + 0.587 * g + 0.114 * b)
    dark_frac = float((lum < 70).mean())     # ink / shadow pixels
    light_frac = float((lum > 225).mean())   # background / paper
    ink_or_paper = float((lum > 220).mean() + (lum < 110).mean())

    # --- distinct colour count (fine 5-bit histogram, for the photo rule) --- #
    q = (rgb >> 3).astype(np.int32)  # 32 levels / channel
    key = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    vals, counts = np.unique(key, return_counts=True)
    total = float(counts.sum())
    sig = counts[counts / total > 0.005]     # ignore anti-alias gradient rings
    n_colors = int(max(1, min(max(sig.size, 400), 400)))

    # --- hairline ratio: dark px with >=2 light 4-neighbours --------------- #
    # a 1-px rule is exactly what blurring, median filters and potrace's
    # turdSize eat first, so line-art routing needs to know about it
    dark = lum < 90.0        # lum is 0..255 here, not 0..1
    if dark.any():
        light = ~dark
        nb = np.zeros(lum.shape, np.int8)
        nb[:, 1:] += light[:, :-1].astype(np.int8)
        nb[:, :-1] += light[:, 1:].astype(np.int8)
        nb[1:, :] += light[:-1, :].astype(np.int8)
        nb[:-1, :] += light[1:, :].astype(np.int8)
        thin_frac = float(((dark & (nb >= 2)).sum()) / max(1, int(dark.sum())))
    else:
        thin_frac = 0.0

    # --- dominant colour regions (k-means, for flat / line-art / arch) ------ #
    n_regions = _dominant_count(rgb, k=24, frac=0.005)

    # --- edges via central-difference gradient (0-255, ABSOLUTE threshold) -- #
    gy, gx = np.gradient(lum)  # central differences
    mag = np.sqrt(gx * gx + gy * gy)         # 0..~360
    edge_abs = float((mag > 16.0).mean())    # real lines / texture, not soft gradients
    sharp_ratio = (float((mag > 40.0).mean()) /
                   max(1e-6, float((mag > 16.0).mean())))

    # --- straight-edge ratio: orientation peakiness (gated on edge volume) -- #
    if edge_abs > 0.10:
        ang = (np.arctan2(gy, gx) % np.pi)   # 0..pi, rotation-invariant
        w = mag * (mag > 16.0)
        if w.sum() > 1e-3:
            hist, _ = np.histogram(ang[mag > 16.0], bins=18, range=(0, np.pi),
                                   weights=w[mag > 16.0])
            hist = hist / hist.sum()
            order = np.sort(hist)[::-1]
            straight_ratio = float(order[:2].sum())
        else:
            straight_ratio = 0.2
    else:
        straight_ratio = 0.2  # not enough edges to measure -> neutral

    # --- skin-tone ratio (face / character cue), tightened ---------------- #
    # g > b keeps orange/red sunsets & colour gradients out (they read as
    # "skin" under the looser rule); real skin is R > G > B.
    skin = ((r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (g > b) &
            ((mx - mn) > 15) & (np.abs(r - g) > 15))
    skin_ratio = float(skin.mean())

    # --- colour entropy (smooth gradient vs busy texture) ------------------- #
    cq = (rgb >> 4).astype(np.int32)  # 16 levels / channel -> 4096 bins
    ckey = (cq[..., 0] << 8) | (cq[..., 1] << 4) | cq[..., 2]
    cvals, ccounts = np.unique(ckey, return_counts=True)
    p = ccounts / float(ccounts.sum())
    entropy = float(-(p * np.log(p + 1e-9)).sum())
    entropy_n = entropy / float(np.log(4096))

    # --- smooth dark area share (silk / shadow on skin, low-sat dark) -------- #
    # silk stockings & soft skin-shadow read as dark + low-saturation pixels.
    # High share => the image is dominated by smooth dark gradients, which the
    # colour-block engines render as banding/streaks; lumina@256 + strong edges
    # is smoother there than qq3d (verified on the c588 stocking figure).
    smooth_dark = float(((lum < 140) & (sat < 0.12)).mean())

    return dict(
        n_colors=n_colors,
        smooth_dark=smooth_dark,
        n_regions=n_regions,
        mean_sat=mean_sat,
        is_gray=bool(is_gray),
        dark_frac=dark_frac,
        light_frac=light_frac,
        ink_or_paper=ink_or_paper,
        edge_abs=edge_abs,
        sharp_ratio=min(1.0, sharp_ratio),
        straight_ratio=straight_ratio,
        skin_ratio=skin_ratio,
        entropy=entropy_n,
        thin_frac=thin_frac,
    )


# --------------------------------------------------------------------------- #
# decision tree
# --------------------------------------------------------------------------- #
def _pick(f: dict) -> str:
    """Pick a content category from features (robust rule order)."""
    gray = f["is_gray"]
    dark = f["dark_frac"]
    light = f["light_frac"]
    edge = f["edge_abs"]
    straight = f["straight_ratio"]
    skin = f["skin_ratio"]
    ent = f["entropy"]
    n_colors = f["n_colors"]
    n_regions = f["n_regions"]
    iop = f["ink_or_paper"]

    # 1) texture: high entropy + very busy edges everywhere (no large smooth areas)
    if ent > 0.78 and edge > 0.42:
        return "texture"
    # 2) line art: grayscale INK DRAWING on (dominant) light paper, low entropy.
    #    CRITICAL: require light_frac (paper) to dominate -- NOT ink_or_paper,
    #    because ink_or_paper also counts large DARK areas, so dark night photos /
    #    shaded illustrations (dark_frac ~0.9) masquerade as "bimodal" and would
    #    be mis-sent to the sketch engine. A genuine line drawing is white-paper
    #    dominant (light_frac > 0.30), has crisp edges, and very low entropy
    #    (few tonal values). Dark-background "line art" is intentionally NOT
    #    caught here -- it falls through to anime/photo (faithful qq3d), which is
    #    safe. ent < 0.30 also drops muted light-bg illustrations (e.g. desaturated
    #    anime art) that are not line drawings.
    if gray and light > 0.30 and edge > 0.08 and ent < 0.30:
        return "line_art"
    # 3) architecture: many long straight edges, geometric.  Require enough
    #    structure (many regions OR busy straight edges OR very straight) so a
    #    2-rectangle logo (few regions, sparse edges) falls through to flat_icon.
    #    straightness must be clearly geometric (>0.60) -- a dark photo with a few
    #    bright streaks (car/road/neon) sits around 0.55 and must stay a photo.
    if (straight > 0.60 and edge > 0.08 and
            (n_regions > 6 or edge > 0.20 or straight > 0.70)):
        return "architecture"
    # 4) anime / character: skin present + organic (low straight).  Colour count
    #    is NOT a discriminator -- detailed illustrations have thousands of
    #    colours (gradients), so we rely on the skin cue + low straightness.
    if skin > 0.05 and straight < 0.45:
        return "anime"
    # 5) flat / icon: very few dominant regions, simple shapes
    if n_regions <= 14 and edge < 0.22 and not (gray and dark > 0.04):
        return "flat_icon"
    # 6) photo: many fine colours or high entropy (and not busy-texture)
    if n_colors > 95 or ent > 0.55:
        return "photo"
    # 7) default: illustrated content
    return "anime"


def _support(f: dict, cat: str) -> float:
    """Rough 0..1 confidence signal for the chosen category (for display)."""
    if cat == "line_art":
        return 1.0 if f["ink_or_paper"] > 0.7 else 0.5
    if cat == "flat_icon":
        return (1.0 - min(1.0, f["edge_abs"] / 0.22)) * (1.0 if f["n_regions"] <= 14 else 0.3)
    if cat == "anime":
        return min(1.0, 0.4 + f["skin_ratio"] / 0.10)
    if cat == "architecture":
        return min(1.0, 0.4 + (f["straight_ratio"] - 0.6) / 0.3)
    if cat == "photo":
        return min(1.0, 0.4 + f["entropy"] / 0.4)
    if cat == "texture":
        return min(1.0, 0.4 + f["edge_abs"] / 0.4)
    return 0.5


def _score_for_display(f: dict) -> dict:
    """Light additive score per category (informational; the tree picks)."""
    sat, gray = f["mean_sat"], f["is_gray"]
    n_reg = f["n_regions"]
    n, edge = f["n_colors"], f["edge_abs"]
    straight, skin, ent = f["straight_ratio"], f["skin_ratio"], f["entropy"]
    dark, iop = f["dark_frac"], f["ink_or_paper"]

    def c(x, lo=0.0, hi=1.0):
        return float(min(hi, max(lo, x)))

    s = {
        "line_art": 1.0 * (1.0 if (gray and iop > 0.55) else c(1 - sat / 0.25))
                     + 1.2 * c((edge - 0.08) / 0.15) - 1.5 * skin,
        "flat_icon": 1.2 * c((16 - n_reg) / 12) + 1.0 * c((edge - 0.02) / 0.14) * c((0.22 - edge) / 0.12 + 0.4)
                     - 1.0 * skin - 1.0 * c((straight - 0.55) / 0.20),
        "anime": 1.3 * c(skin / 0.06) + 0.8 * c((n - 30) / 60) * c((200 - n) / 80)
                  + 0.4 * c(sat / 0.30) - 1.2 * c((straight - 0.28) / 0.15),
        "architecture": 1.4 * c((straight - 0.55) / 0.15) + 0.8 * c((edge - 0.08) / 0.15)
                        - 1.2 * skin - 0.5 * (1.0 if gray else 0.0),
        "photo": 0.9 * c((n - 90) / 80) + 0.7 * c((ent - 0.55) / 0.25)
                 - 0.3 * c((straight - 0.30) / 0.15),
        "texture": 1.2 * c((edge - 0.18) / 0.15) + 0.6 * c((ent - 0.60) / 0.2)
                   + 0.6 * c((iop - 0.5) / 0.3) - 1.0 * skin,
    }
    return {k: round(v, 3) for k, v in s.items()}


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
CATEGORY_LABEL = {
    "line_art": "线稿 / 漫画",
    "flat_icon": "图标 / 平面 / Logo",
    "anime": "动漫插画 / 人物",
    "architecture": "建筑 / 风景",
    "photo": "写实照片",
    "texture": "纹理 / 材质",
}

# which engine each category routes to (mirrors MODE_* names)
CATEGORY_ENGINE = {
    "line_art": core.MODE_SKETCH,
    "flat_icon": core.MODE_FLAT,
    "anime": core.MODE_QQ3D,
    "architecture": core.MODE_LUMINA,
    "photo": core.MODE_LUMINA,
    "texture": core.MODE_QQ3D,
}


def classify(arr: np.ndarray) -> dict:
    """Classify a RGB array. Returns dict with category, label, engine, scores."""
    f = _features(arr)
    cat = _pick(f)
    support = _support(f, cat)
    scores = _score_for_display(f)
    # bias the displayed scores so the picked category reads as the winner
    scores[cat] = max(scores[cat], round(1.0 + support, 3))
    return dict(
        category=cat,
        label=CATEGORY_LABEL[cat],
        engine=CATEGORY_ENGINE[cat],
        scores=scores,
        features=f,
        confidence=round(float(min(1.0, 0.55 + 0.45 * support)), 3),
    )


def preset_for(category: str, features: dict | None = None) -> "core.Settings":
    """Tuned Settings for a detected category."""
    if category == "line_art":
        # Hairline-dominant drawings must NOT go to sketch: its luminance-band +
        # potrace path drops 1-px rules (measured: 50 rules -> empty SVG). The
        # self-developed tracer keeps them, so route by stroke fineness.
        thin = float((features or {}).get("thin_frac", 0.0))
        if thin >= 0.30:
            return core.Settings(mode=core.MODE_OWN, colors=64, detail=8,
                                 faithful_hd=False).normalised()
        return core.Settings(mode=core.MODE_SKETCH, colors=0, detail=6,
                             faithful_hd=False).normalised()
    if category == "flat_icon":
        return core.Settings(mode=core.MODE_FLAT, colors=24, grid_width=1600,
                             detail=6, gap=True, faithful_hd=False).normalised()
    if category == "anime":
        # 默认走 lumina 256色 + 强边缘 + 保真高清：vtracer cutout 共边 + 样条拟合，
        # 在 2× 超分上约 18s（qq3d 同图约 180s，慢一个数量级），且对丝袜/光滑暗区
        # 更顺（qq3d256 在脚踝有方形硬边块，已实测）。qq3d 保真引擎保留给需要
        # 极致渐变还原的场景，可在「引擎」下拉手动选。
        # v7.5.1: hd=True 必须显式给 —— auto 会用本预设整体替换 GUI 设置，
        # 之前只写了 faithful_hd（那是 qq3d 的开关），导致 lumina 没超分，
        # 输出停在原尺寸、边缘全是像素阶梯锯齿。
        return core.Settings(mode=core.MODE_LUMINA, colors=256,
                             edge_smooth="strong", faithful_hd=True,
                             hd=True, hd_scale=2.0).normalised()
    if category == "architecture":
        return core.Settings(mode=core.MODE_LUMINA, colors=128, detail=10,
                             smooth=1, straighten=True, faithful_hd=True,
                             hd=True, hd_scale=2.0).normalised()
    if category == "photo":
        # 与 anime 一致走 lumina 256色 + 强边缘 + 保真高清：实测比 qq3d 更干净、
        # 更快（qq3d 瓦片管线会在嘴角等边缘产生细线/拉丝伪影，且慢一个数量级），
        # 对写实照片渐变同样平滑。需要极致还原可在「引擎」下拉手动选 qq3d。
        # v7.5.1: 同 anime，补 hd=True 让超分真正生效（用户认可的 d926 好文件
        # 就是 1677x2981 ≈ 1.86x 超分；之前 auto 丢失 hd 导致 900x1600 锯齿版）。
        return core.Settings(mode=core.MODE_LUMINA, colors=256,
                             edge_smooth="strong", faithful_hd=True,
                             hd=True, hd_scale=2.0).normalised()
    if category == "texture":
        return core.Settings(mode=core.MODE_QQ3D, colors=256, profile="detail",
                             edge_smooth="soft", faithful_hd=True,
                             hd_scale=2.0).normalised()
    # safe default
    return core.Settings(mode=core.MODE_QQ3D, colors=256).normalised()


def suggest(src) -> dict:
    """Convenience: classify a path / PIL image and return settings + info."""
    arr = _load_rgb(src)
    info = classify(arr)
    # refine the hairline share at near-native resolution before choosing an engine
    info["features"]["thin_frac"] = _thin_frac_native(arr)
    info["settings"] = preset_for(info["category"], info["features"])
    # keep the displayed engine honest: the preset may override the category's
    # default engine (e.g. anime -> lumina for speed), so sync the label to the
    # mode that will actually run.
    info["engine"] = info["settings"].mode
    return info
