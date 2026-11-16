"""Core image -> SVG tracing pipeline (flat palette mode + high fidelity mode)."""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import uuid

import tilefix
import time
from dataclasses import dataclass, asdict

import vtracer
import numpy as np
from PIL import Image, ImageOps

try:
    from enhance import enhance as _hd_enhance
except Exception:  # package import (tests) - resolve relative module
    try:
        from .enhance import enhance as _hd_enhance
    except Exception:  # pragma: no cover
        _hd_enhance = None

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

MODE_FLAT = "flat"
MODE_HIFI = "hifi"
MODE_CEL = "cel"
MODE_WRAP = "wrap"
MODE_EXACT = "exact"
MODE_PRINT = "print"
MODE_REF = "ref"
MODE_REGION = "region"
MODE_PTRACE = "potrace"
MODE_LUMINA = "lumina"
MODE_SKETCH = "sketch"
MODE_OWN = "own"          # self-developed layered tracer (experimental)
MODE_OWN_LAB = "ownlab"   # experimental build: own + thin-line gap repair
MODE_QQ3D = "qq3d"        # qq3d-like fidelity tracer: pre-quantize + palette snap
MODE_AI_LINE = "ai_line"   # 自研形态学线条层（无需 torch / 无外部权重）；常量名保留以兼容旧配置
APP_VERSION = "v7.6.1"    # 单一版本源：GUI 标题 / 打包文件名都从这里派生，勿在别处写死

# 输出文件名统一前缀：本软件品牌名（Lumina 叠色打印 + 矢量 Vector 的缩写）。
# 所有导出 SVG 一律以它开头，形如 LumVec-20261115_132501-anime.svg。
OUT_PREFIX = "LumVec"

MODE_GEOM = "geom"        # geometric redraw: corner-cut L / arc A / quad Q big blocks
MODE_SILK = "silk"        # v7.0 smooth-strand redraw: one-block-one-path + dense short A/Q
MODE_SILK_CLEAN = "silkclean"  # v7.1 preset: 大色块/渐变背景专用（96色+强平滑+色差去斑）
MODE_SILK_PRINT = "silkprint"  # v7.0 打印预设: 上述 + 按挤出线宽清碎区（Lumina 打印可打印性）
MODE_AUTO = "auto"        # auto-detect image type, route to the best engine

MODE_LABEL = {
    MODE_CEL: "动漫色块 · 赛璐璐",
    MODE_HIFI: "精细 · 高还原",
    MODE_FLAT: "分色 · 印刷 / 多色打印",
    MODE_WRAP: "完全一致 · 封装原图（非矢量）",
    MODE_EXACT: "像素级矢量 · 最大还原",
    MODE_PRINT: "3D 叠色打印 · 分色",
    MODE_REF: "参考风格 · 家教纲吉同款",
    MODE_REGION: "区域矢量化（实验）",
    MODE_LUMINA: "vtracer 高清描摹（默认底图引擎）",
    MODE_AI_LINE: "线条层 · 自研形态学描线（无需模型）",
    MODE_SKETCH: "线稿 / 漫画模式",
    MODE_PTRACE: "potrace 描摹 · 硬边清晰",
    MODE_OWN: "自研分层描摹（基线）",
    MODE_OWN_LAB: "自研 · 细线补缝（实验）",
    MODE_QQ3D: "保真还原 · qq3d 同款",
    MODE_GEOM: "几何化重绘 · 长圆弧大色块（实验）",
    MODE_SILK: "丝状顺滑 · 对标嘻嘻宝（v7.0）",
    MODE_SILK_CLEAN: "大色块背景 · 干净无碎斑（推荐用于深红/渐变背景）",
    MODE_SILK_PRINT: "可打印 · 按线宽清碎区（Lumina 叠色打印推荐）",
    MODE_AUTO: "自动识别图片类型（推荐）",
}


@dataclass
class Settings:
    mode: str = MODE_LUMINA
    colors: int = 128          # cel: colour blocks (auto-raised to match) / flat: print palette
    grid_width: int = 1600     # tracing grid width in px (larger images get downscaled)
    detail: int = 6            # filter_speckle, 越大越丢掉细碎斑点
    smooth: int = 2            # 1..3, path smoothing
    gap: bool = False          # stroke seams so neighbouring regions never leave hairlines
    band: bool = False         # flatten gradients (line-aware blur) before tracing
    clean: bool = True         # remove small round specks before tracing
    drop_specks: bool = True   # drop leftover dark specks from the SVG
    line_layer: bool = False   # redraw lost linework as a top layer (off: looks busy)
    straighten: bool = True    # snap near-straight bezier runs to lines
    profile: str = "clean"     # qq3d 引擎色块干净度: detail / balanced / clean
    edge_smooth: str = "normal"  # 边缘顺滑: off / soft / normal / strong
    self_line: bool = False    # 自研形态学线条层（v7.5.6 起无需任何外部模型 / 权重）
    ai_line: bool = False       # 自研形态学线条层（可选，无需任何模型/权重）
    ai_line_weights: str = ""   # 保留字段以兼容旧配置；v7.5.3 起不再使用
    print_mm: float = 0.0      # >0 时启用打印瘦身：目标长边打印尺寸（毫米）
    line_mm: float = 0.42      # 挤出线宽（毫米），用于换算珠宽
    faithful_hd: bool = True   # qq3d 前先做保真高清化（只补模糊、不改清晰度）
    seamless: bool = True      # 全图单次描摹，消除分块瓦片边界的「拼接细线」
                               # (同输入 RMSE 12.15→12.01，代价是耗时约 2-3 倍)
    auto_sketch: bool = False  # force the line-art band tracer (manual switch)
    detail_layer: bool = True  # re-add thin strokes and small highlights on top
    gap_width: float = 0.8     # seam stroke width in grid units
    optimize: bool = True      # scour post-pass

    hd: bool = False           # detail-preserving upscale before tracing
    hd_scale: float = 2.0      # 2x / 3x / 4x
    hd_strength: float = 0.5   # edge crispening strength 0..1

    # 几何化重绘（MODE_GEOM）专用：大色块目标色数 / 曲线拟合容差 px
    geom_colors: int = 128
    geom_tol: float = 2.0
    # 相邻色块外扩像素（>=tol 才能无缝；1.0 为 RMSE/缝 的最优拐点）
    geom_expand: float = 1.0
    # 同色描边遮缝宽度 px（Lumina 导入时剥离，只影响预览；0=关）
    geom_seam: float = 0.8

    # 丝状顺滑引擎（MODE_SILK / v7.0）专用
    silk_colors: int = 128        # 感知量化色数（对标嘻嘻宝 79~116）
    silk_chord: float = 4.5      # 单条命令最大弦长 px（越小越密、越贴）
    silk_tol: float = 1.2        # 弧 / 二次拟合容差 px
    silk_smooth: float = 0.8     # 标签投票平滑
    silk_rare: float = 0.001     # 稀有色合并阈值（去彩色碎斑；0=关）
    silk_fit: str = "chord"      # 拟合层: chord=密集短段(对齐嘻嘻宝,推荐) / smooth=误差自适应(文件小但会露底,实验)
    silk_contrast_de: float = 0.0    # 局部色差去斑 ΔE 门槛（0=关；推荐 0.10~0.16）
    silk_contrast_area: float = 8.0  # 局部色差去斑的面积上限 px（越小越保守）
    # 打印可打印性：按物理挤出线宽清掉小于一个挤出点的连通区域（0=关）
    silk_print_mm: float = 0.0       # 成品长边 mm；>0 时启用（默认 263）
    silk_print_scale: float = 1.0    # 阈值倍率（1.0=恰好一个挤出点，实测最优）
    silk_band_sigma: float = 0.0     # 避线稿可变模糊上限 σ（0=关，实验）

    # Fields the user EXPLICITLY set (GUI control touched / CLI flag given).
    # MODE_AUTO replaces Settings with a classifier preset, so without this the
    # default auto mode silently ignores every option the user picks.
    explicit: tuple = ()

    def normalised(self) -> "Settings":
        s = Settings(**asdict(self))
        if s.mode not in (MODE_CEL, MODE_HIFI, MODE_FLAT, MODE_WRAP, MODE_EXACT, MODE_PRINT, MODE_REF, MODE_REGION, MODE_PTRACE, MODE_LUMINA, MODE_SKETCH, MODE_OWN, MODE_OWN_LAB, MODE_QQ3D, MODE_GEOM, MODE_SILK, MODE_SILK_CLEAN, MODE_SILK_PRINT, MODE_AI_LINE, MODE_AUTO):
            s.mode = MODE_LUMINA
        s.geom_colors = max(16, min(256, int(s.geom_colors)))
        s.geom_tol = max(0.5, min(6.0, float(s.geom_tol)))
        s.geom_expand = max(0.0, min(6.0, float(s.geom_expand)))
        s.geom_seam = max(0.0, min(6.0, float(s.geom_seam)))
        s.silk_colors = max(16, min(256, int(s.silk_colors)))
        s.silk_chord = max(1.2, min(20.0, float(s.silk_chord)))
        s.silk_tol = max(0.2, min(6.0, float(s.silk_tol)))
        s.silk_smooth = max(0.0, min(5.0, float(s.silk_smooth)))
        s.silk_rare = max(0.0, min(0.02, float(s.silk_rare)))
        s.silk_fit = "smooth" if s.silk_fit != "chord" else "chord"
        s.silk_contrast_de = max(0.0, min(0.5, float(s.silk_contrast_de)))
        s.silk_contrast_area = max(2.0, min(200.0, float(s.silk_contrast_area)))
        s.silk_print_mm = max(0.0, min(2000.0, float(s.silk_print_mm)))
        s.silk_print_scale = max(0.0, min(8.0, float(s.silk_print_scale)))
        s.silk_band_sigma = max(0.0, min(12.0, float(s.silk_band_sigma)))
        if s.mode == MODE_CEL:
            s.colors = max(8, min(512, int(s.colors)))
        elif s.mode == MODE_PRINT:
            s.colors = max(4, min(32, int(s.colors)))
        elif s.mode in (MODE_LUMINA, MODE_QQ3D, MODE_AUTO):
            # 高清精修：允许更高色数上限（减少渐变色带/杂色）；默认仍由预设给 256。
            s.colors = max(2, min(768, int(s.colors)))
        else:
            s.colors = max(2, min(256, int(s.colors)))
        s.grid_width = max(64, min(4000, int(s.grid_width)))
        s.detail = max(0, min(20, int(s.detail)))
        s.smooth = max(1, min(3, int(s.smooth)))
        s.gap_width = max(0.0, min(2.0, float(s.gap_width)))
        if s.profile not in ("detail", "balanced", "clean"):
            s.profile = "clean"
        if s.edge_smooth not in ("off", "soft", "normal", "strong"):
            s.edge_smooth = "normal"
        s.hd_scale = max(1.25, min(4.0, float(s.hd_scale)))
        s.hd_strength = max(0.0, min(1.0, float(s.hd_strength)))
        return s


# --------------------------------------------------------------------------- #
# image loading
# --------------------------------------------------------------------------- #
def load_rgb(path: str, bg: str = "#ffffff") -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (
        im.mode == "P" and "transparency" in im.info
    )
    if has_alpha:
        im = im.convert("RGBA")
        base = Image.new("RGB", im.size, bg)
        base.paste(im, mask=im.getchannel("A"))
        return base
    return im.convert("RGB")


def _fit_and_quantize(im: Image.Image, grid_width: int, colors: int | None) -> Image.Image:
    w, h = im.size
    gw = min(grid_width, w) if grid_width else w
    if gw != w:
        gh = max(1, round(h * gw / w))
        im = im.resize((gw, gh), Image.LANCZOS)
    if colors:
        try:
            im = _quantize_perceptual(im, colors)
        except Exception:
            pal = im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                              dither=Image.Dither.NONE)
            im = pal.convert("RGB")
    return im


def _prepare(path: str, grid_width: int, colors: int | None) -> Image.Image:
    return _fit_and_quantize(load_rgb(path), grid_width, colors)


# --------------------------------------------------------------------------- #
# perceptual palette (OKLab k-means) - keeps small vivid regions (cyan dial etc.)
# --------------------------------------------------------------------------- #
def _srgb_to_oklab(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    m = a > 0.04045
    a = np.where(m, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
    l = 0.4122214708 * a[..., 0] + 0.5363325363 * a[..., 1] + 0.0514459929 * a[..., 2]
    m2 = 0.2119034982 * a[..., 0] + 0.6806995451 * a[..., 1] + 0.1073969566 * a[..., 2]
    s = 0.0883024619 * a[..., 0] + 0.2817188376 * a[..., 1] + 0.6299787005 * a[..., 2]
    l_, m_, s_ = np.cbrt(l), np.cbrt(m2), np.cbrt(s)
    out = np.stack([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ], axis=-1)
    # 审计 B-12: 极端输入可能在 cbrt / 线性组合后产生 nan/inf，下游量化会崩。
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def _oklab_to_rgb01(lab: np.ndarray) -> np.ndarray:
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


def _kmeans_palette(pix: np.ndarray, k: int, iters: int = 14, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(pix)
    samp = pix[rng.choice(n, min(60000, n), replace=False)]
    cent = [samp[rng.integers(len(samp))]]
    d2 = ((samp - cent[0]) ** 2).sum(1)
    for _ in range(1, k):
        total = float(d2.sum())
        nxt = samp[rng.integers(len(samp))] if total <= 0 else samp[rng.choice(len(samp), p=d2 / total)]
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
            else:
                # 审计 B-09: 空簇重新随机初始化，避免色板出现重复色 / 实际色数不足。
                cent[j] = samp[rng.integers(len(samp))]
    return cent


def _quantize_perceptual(im: Image.Image, colors: int) -> Image.Image:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    lab = _srgb_to_oklab(arr)
    cent = _kmeans_palette(lab.reshape(-1, 3), max(2, int(colors)))

    pal = (_oklab_to_rgb01(cent) * 255.0).round().astype(np.uint8)
    g = 40  # sRGB -> palette lookup table resolution
    xs = np.linspace(0, 255, g).astype(np.float32)
    grid = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), -1).reshape(-1, 3) / 255.0
    glab = _srgb_to_oklab(grid)
    lut = np.empty(len(glab), dtype=np.int32)
    step = 6000
    for i in range(0, len(glab), step):
        seg = glab[i:i + step]
        d = (seg ** 2).sum(1)[:, None] - 2 * seg @ cent.T + (cent ** 2).sum(1)[None, :]
        lut[i:i + step] = d.argmin(1)
    lut = lut.reshape(g, g, g)

    q = (arr * (g - 1)).round().astype(np.int32)
    idx = lut[q[..., 0], q[..., 1], q[..., 2]]
    return Image.fromarray(pal[idx], "RGB")


# --------------------------------------------------------------------------- #
# tracing
# --------------------------------------------------------------------------- #
def unique_dst(d: str, stem: str, used=()) -> str:
    """GUI / CLI 共用：目标已存在时加序号，绝不静默覆盖上一轮成果。

    审计 F-03 / B-15 / I-05：同名不同扩展名（a.jpg / a.png）或重复运行
    旧公式会生成完全相同的 dst，后写者无声覆盖前写者。
    """
    cand = os.path.join(d, stem + ".svg")
    i = 1
    while cand in used or os.path.exists(cand):
        i += 1
        cand = os.path.join(d, "%s (%d).svg" % (stem, i))
    return cand


def out_stem() -> str:
    """统一输出文件名主干：前缀 + 本地时间戳（到秒，可排序、批量不撞名）。

    形如 ``LumVec-20261115_132501``；转换完成后由 rename_with_type 追加剧类型。
    """
    return "%s-%s" % (OUT_PREFIX, time.strftime("%Y%m%d_%H%M%S"))


def type_token(info: dict | None) -> str | None:
    """从转换结果里取一个安全的 ASCII 类型 token 用于文件名。

    优先用 auto 模式识别出的图片类别（line_art/flat_icon/anime/photo/…），
    没有类别时退回引擎 mode（本身已是短 ascii），都没有则返回 None（不加类型）。
    """
    info = info or {}
    cat = info.get("auto_category")
    if cat:
        return {
            "line_art": "line",
            "flat_icon": "icon",
            "anime": "anime",
            "photo": "photo",
            "architecture": "arch",
            "texture": "texture",
        }.get(cat, cat)
    mode = info.get("mode")
    return mode or None


def rename_with_type(dst: str, info: dict | None) -> str:
    """转换完成后，把识别出的图片类型追加进文件名。

    例：LumVec-20261115_132501.svg → LumVec-20261115_132501-anime.svg。
    永不抛异常；任何失败都原样返回 dst（绝不丢文件）；当「带类型的精确文件名」
    已存在（秒级时间戳撞名，极罕见）时保留原名、既不追加也不覆盖。
    """
    typ = type_token(info)
    if not typ:
        return dst
    d = os.path.dirname(dst)
    base = os.path.splitext(os.path.basename(dst))[0]
    if base.endswith("-" + typ):   # 已经带过类型（理论不会发生），不再叠加
        return dst
    target = os.path.join(d, "%s-%s.svg" % (base, typ))
    if os.path.exists(target):     # 撞名：保原名，不覆盖、不丢文件
        return dst
    try:
        os.replace(dst, target)
        return target
    except OSError:
        return dst


def _tmp_png(tag: str) -> str:
    """唯一临时文件：旧版只用 pid，并发会互相覆盖、异常会永久残留。审计 B-10。"""
    return os.path.join(tempfile.gettempdir(),
                        "img2svg_%s_%s.png" % (tag, uuid.uuid4().hex[:12]))


def _vtrace(png: str, out: str, **kw) -> None:
    """Trace a PNG to SVG.

    v7.5.6: the optional vtracer 1.0 ``simplify`` path was removed — it had no
    CLI/GUI entry point and an A/B showed it inflates path count 23414 -> 106339
    with visible face striping.  1.0's ``clustering``/``max_colors`` keywords are
    still translated back to 0.6's ``colormode`` for callers that supply them.

    vtracer 0.6's Rust loader panics on a **non-ASCII path + JPEG** input
    ("No image file found at specified input path"), so the 0.6 path always
    feeds it a normalised PNG in an ASCII temp directory.
    """
    import vtracer as _v06
    from PIL import Image as _Im

    _kw = dict(kw)
    if "clustering" in _kw:
        _kw["colormode"] = _kw.pop("clustering")
    for _k in ("max_colors", "palette"):
        _kw.pop(_k, None)

    _need_fix = (not png.lower().endswith(".png")) or any(ord(c) > 127 for c in png)
    _tmp = None
    _src = png
    if _need_fix:
        _tmp = _tmp_png("vt06")
        with _Im.open(png) as _im:
            _im.convert("RGB").save(_tmp)
        _src = _tmp
    try:
        _v06.convert_image_to_svg_py(_src, out, **_kw)
    finally:
        if _tmp:
            try:
                os.remove(_tmp)
            except OSError:
                pass


def _vtracer_kwargs(s: Settings) -> dict:
    smooth = {1: (58, 4.2), 2: (72, 3.6), 3: (86, 3.0)}[s.smooth]
    corner, length = smooth
    if s.mode == MODE_FLAT:
        color_precision, layer_diff = 8, 4
    else:
        color_precision, layer_diff = 7, 16
    return dict(
        clustering="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=max(4, min(8, s.detail)),
        color_precision=color_precision,
        layer_difference=layer_diff,
        corner_threshold=max(78, corner),
        length_threshold=min(3.0, length),
        max_iterations=10,
        splice_threshold=45,
        path_precision=2,
    )


_PATH_RE = re.compile(r"<path\b([^>]*?)/>")
_FILL_RE = re.compile(r'fill="(#[0-9a-fA-F]{3,8})"')


def _merge_svg_colours(svg: str, target: int, source=None) -> str:
    """Collapse near-identical fills with area-weighted k-means.

    Weighting by how often each fill occurs keeps the palette faithful: earlier
    versions either weighted every colour equally (near-white background
    variants dragged mid-tones grey) or snapped fills onto a PIL palette
    (visibly shifted colours - measured LAB delta 9.2 vs 6.5 unmerged).
    """
    import collections

    fills = re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg)
    if not fills:
        return svg
    counts = collections.Counter(c.lower() for c in fills)
    uniq = list(counts)
    if len(uniq) <= target or target < 2:
        return svg

    rgb = np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)]
                    for c in uniq], np.float32)
    w = np.array([counts[c] for c in uniq], np.float32)
    w = w / w.sum()

    rng = np.random.default_rng(7)
    cent = rgb[rng.choice(len(rgb), min(target, len(rgb)), replace=False, p=w)].copy()
    for _ in range(32):
        d = ((rgb[:, None, :] - cent[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        for k in range(len(cent)):
            sel = lab == k
            if sel.any():
                ww = w[sel]
                cent[k] = (rgb[sel] * ww[:, None]).sum(0) / ww.sum()

    mapping = {}
    for i, c in enumerate(uniq):
        r, g, b = cent[lab[i]].round().astype(int)
        mapping[c] = "#%02x%02x%02x" % (r, g, b)

    out = re.sub(r'fill="(#[0-9a-fA-F]{6})"',
                 lambda m: 'fill="%s"' % mapping.get(m.group(1).lower(), m.group(1)), svg)
    out = re.sub(r'stroke="(#[0-9a-fA-F]{6})"',
                 lambda m: 'stroke="%s"' % mapping.get(m.group(1).lower(), m.group(1)), out)
    return out


def _add_seam_strokes(svg: str, width: float) -> str:
    def repl(m):
        attrs = m.group(1)
        if 'stroke=' in attrs:
            return m.group(0)
        f = _FILL_RE.search(attrs)
        if not f:
            return m.group(0)
        c = f.group(1)
        return (
            f'<path{attrs} stroke="{c}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    return _PATH_RE.sub(repl, svg)


_ROOT_RE = re.compile(r"<svg\b[^>]*>")


def _exact_kwargs() -> dict:
    """Pixel-faithful polygon trace: no smoothing, no speckle removal.

    Deliberately does NOT use ``simplify`` -- that pass would round the
    pixel-exact polygons the user asked this mode to preserve.
    """
    return dict(
        clustering="color",
        hierarchical="stacked",
        mode="polygon",
        filter_speckle=1,
        color_precision=8,
        layer_difference=2,
        corner_threshold=60,
        length_threshold=2.0,
        max_iterations=10,
        splice_threshold=45,
        path_precision=2,
    )


def _ref_kwargs(s: Settings) -> dict:
    """Cutout trace: non-overlapping flat regions with hard edges."""
    return dict(
        clustering="color",
        hierarchical="cutout",
        mode="spline",
        filter_speckle=4,
        color_precision=8,
        layer_difference=6,
        corner_threshold=62,
        length_threshold=3.4,
        max_iterations=10,
        splice_threshold=45,
        path_precision=2,
    )


def _cel_kwargs(s: Settings) -> dict:
    """vtracer settings for flat cel artwork: long splines, few corners."""
    return dict(
        clustering="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=max(3, min(8, s.detail)),
        color_precision=8,
        layer_difference=4,
        corner_threshold=65,
        length_threshold=3.2,
        max_iterations=10,
        splice_threshold=45,
        path_precision=2,
    )


def _set_root(svg: str, width: int, height: int, scale: float, title: str) -> str:
    disp_w = max(1, round(width * scale))
    disp_h = max(1, round(height * scale))
    root = (
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{disp_w}" height="{disp_h}" '
        f'viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">\n'
        f"<title>{title}</title>\n"
    )
    return _ROOT_RE.sub(lambda _: root, svg, count=1)


def _optimize_svg(src: str, dst: str) -> bool:
    try:
        from scour.scour import parse_args, start
    except Exception:
        return False
    try:
        opts = parse_args(["-i", src, "-o", dst, "--enable-comment-strip", "-q"])
        with io.open(src, encoding="utf-8") as fin, io.open(dst, "wb") as fout:
            start(opts, fin, fout)
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def count_paths(svg_path: str) -> int:
    n = 0
    with io.open(svg_path, encoding="utf-8", errors="ignore") as f:
        for chunk in iter(lambda: f.read(1 << 20), ""):
            n += chunk.count("<path")
    return n



def _is_lineart(path: str) -> bool:
    """True when the image is dominated by a handful of flat colours (ink art)."""
    try:
        from PIL import Image as _Img

        with _Img.open(path) as im:
            small = im.convert("RGB").resize((200, 200))
        a = np.asarray(small).reshape(-1, 3).astype(np.int32)
        q = (a >> 4)
        key = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
        vals, counts = np.unique(key, return_counts=True)
        order = np.argsort(-counts)
        top = counts[order[:8]].sum() / float(max(1, counts.sum()))
        a8 = np.asarray(small, np.float32)
        sat = (a8.max(2) - a8.min(2)) / 255.0
        # line art = almost entirely greyscale ink; a coloured illustration is not
        if float((sat < 0.12).mean()) < 0.88:
            return False
        g = np.asarray(small.convert("L"), np.float32)
        dark = float((g < 90).mean())
        light = float((g > 200).mean())
        if top > 0.70:
            return True
        return bool(top > 0.55 and (dark + light) > 0.55)
    except Exception:  # noqa: BLE001
        return False


def _fix_backdrop(txt: str, src_path: str) -> "tuple[str, int]":
    """Whiten a mis-detected dark full-canvas backdrop on a light-background image.

    Some engines emit a dark-gray full-canvas ``<path>``/``<rect>`` as the
    "background" even though the artwork sits on white paper; that layer then
    shows through antialiased edges and gaps as stray color blocks.  Only acts
    when the source image's border is clearly light, so genuine dark-background
    art is left untouched.  Returns (possibly-modified svg text, n_fixed).
    """
    try:
        _im = Image.open(src_path).convert("RGB")
        _a = np.asarray(_im)
        _h, _w = _a.shape[:2]
        _border = np.concatenate([_a[0, :], _a[_h - 1, :], _a[:, 0], _a[:, _w - 1]]).astype(np.float32)
        _bg_lum = float(_border.mean())
    except Exception:  # noqa: BLE001
        return txt, 0
    if _bg_lum < 170:
        return txt, 0

    def _dark(hex6):
        try:
            r = int(hex6[1:3], 16); g = int(hex6[3:5], 16); b = int(hex6[5:7], 16)
        except Exception:
            return False
        return (0.299 * r + 0.587 * g + 0.114 * b) < 150

    _fixed = 0
    # full-canvas <rect> backdrop (matches the svg root width/height)
    _mm = re.search(r'<svg[^>]*\bwidth="(\d+)"\s+height="(\d+)"', txt)
    if _mm:
        _W, _H = int(_mm.group(1)), int(_mm.group(2))
        def _rect_fix(m):
            nonlocal _fixed
            _f = re.search(r'fill="(#[0-9a-fA-F]{6})"', m.group(0))
            if _f and _dark(_f.group(1)):
                _fixed += 1
                return re.sub(r'fill="#[0-9a-fA-F]{6}"', 'fill="#ffffff"', m.group(0), count=1)
            return m.group(0)
        txt = re.sub(r'<rect\b[^>]*\bwidth="%d"\s+height="%d"[^>]*?/>' % (_W, _H), _rect_fix, txt)
    # first <path> = bottom-most backdrop in cutout output
    _pm = re.search(r'<path\b([^>]*?)/>', txt, re.S)
    if _pm:
        _f = re.search(r'fill="(#[0-9a-fA-F]{6})"', _pm.group(1))
        if _f and _dark(_f.group(1)):
            _new = re.sub(r'fill="#[0-9a-fA-F]{6}"', 'fill="#ffffff"', _pm.group(0), count=1)
            txt = txt[:_pm.start()] + _new + txt[_pm.end():]
            _fixed += 1
    return txt, _fixed


def convert(src: str, dst: str, settings: Settings | None = None,
            progress=None) -> dict:
    """把 *src* 描摹成 *dst* SVG，并执行 GUI / CLI 共用的后处理。

    v7.5.5 / 审计 I-02：``tilefix`` 与 ``_fix_backdrop`` 过去只在
    ``gui.App._run`` 里执行，CLI 完全不做 —— 同一张图、同一组参数，两条入口
    产物不同（CLI 会保留瓦片细线与深色满屏打底块）。现在下沉到内核：
    两条调用链共用同一条产出路径，GUI 只读取 info 里的计数渲染徽标。
    """
    info = _convert_impl(src, dst, settings, progress)
    try:
        _txt = io.open(dst, encoding="utf-8", errors="ignore").read()
        if tilefix.has_tiles(_txt) and os.path.isfile(src):
            _out, _n = tilefix.fix(_txt, src)
            if _n:
                io.open(dst, "w", encoding="utf-8", newline="\n").write(_out)
                _txt = _out
                info["tilefix"] = _n
        _tf, _nb = _fix_backdrop(_txt, src)
        if _nb:
            io.open(dst, "w", encoding="utf-8", newline="\n").write(_tf)
            info["backdrop_fixed"] = _nb
        if progress and (info.get("tilefix") or info.get("backdrop_fixed")):
            progress("后处理完成（瓦片 %d 处 / 打底 %d 处）"
                     % (info.get("tilefix", 0), info.get("backdrop_fixed", 0)))
    except Exception as _exc:  # noqa: BLE001
        info["postfix_error"] = str(_exc)
    return info


def _convert_impl(src: str, dst: str, settings: Settings | None = None,
                  progress=None) -> dict:
    """Trace *src* image into *dst* svg. Returns an info dict.

    Internal routing (AUTO / legacy fallbacks) must call THIS function, not the
    public wrapper, otherwise the postfix pass would run twice.
    """
    s = (settings or Settings()).normalised()
    t0 = time.time()

    def step(msg):
        if progress:
            progress(msg)

    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    if os.path.splitext(src)[1].lower() not in SUPPORTED_EXT:
        raise ValueError("不支持的图片格式: %s" % os.path.basename(src))

    if s.mode == MODE_AUTO:
        # auto-detect the image type and route to the best engine; the engine
        # mode itself is always chosen by the classifier, but anything the user
        # explicitly set must survive the preset (otherwise the DEFAULT auto
        # mode would silently ignore every GUI / CLI option).
        import classify as _clf

        _info = _clf.suggest(src)
        _resolved = _info["settings"]
        for _k in (s.explicit or ()):
            if _k == "mode" or not hasattr(_resolved, _k):
                continue
            setattr(_resolved, _k, getattr(s, _k))
        _resolved = _resolved.normalised()
        if not s.faithful_hd:
            _resolved.faithful_hd = False
            _resolved.hd_scale = s.hd_scale
        step("自动识别：%s（引擎 %s，置信度 %.0f%%）"
             % (_info["label"], _info["engine"], _info["confidence"] * 100))
        _out = _convert_impl(src, dst, _resolved, progress)
        _out["auto_category"] = _info["category"]
        _out["auto_label"] = _info["label"]
        _out["auto_engine"] = _info["engine"]
        _out["auto_scores"] = _info["scores"]
        _out["auto_confidence"] = _info["confidence"]
        _out["auto_features"] = _info["features"]
        return _out

    step("读取图片…")
    orig_w, orig_h = Image.open(src).size

    if s.mode == MODE_WRAP:
        step("封装原图为 SVG…")
        import base64
        import mimetypes

        mime = mimetypes.guess_type(src)[0] or "image/png"
        data = base64.b64encode(open(src, "rb").read()).decode("ascii")
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
               'width="%d" height="%d" viewBox="0 0 %d %d">\n'
               '<title>Wrapped original image (%dx%d)</title>\n'
               '<image width="%d" height="%d" preserveAspectRatio="none" '
               'href="data:%s;base64,%s"/>\n</svg>\n' % (
                   orig_w, orig_h, orig_w, orig_h, orig_w, orig_h,
                   orig_w, orig_h, mime, data))
        io.open(dst, "w", encoding="utf-8", newline="\n").write(svg)
        step("完成")
        return dict(src=src, dst=dst, mode=s.mode, grid="%dx%d" % (orig_w, orig_h),
                    colors=0, hd="", paths=0, bytes=os.path.getsize(dst),
                    seconds=round(time.time() - t0, 1))

    im = load_rgb(src)

    if s.hd:
        if _hd_enhance is None:
            raise RuntimeError("高清化模块不可用（enhance.py 缺失）")
        step("高清化 %s×（%.0f%% 强度）…" % (
            ("%.2f" % s.hd_scale).rstrip("0").rstrip("."), s.hd_strength * 100))
        im = _hd_enhance(im, scale=s.hd_scale, strength=s.hd_strength)
        grid = min(im.width, 2600)
    else:
        grid = s.grid_width

    if s.mode == MODE_SKETCH or (s.mode == MODE_LUMINA and s.auto_sketch):
        step("线稿模式（亮度分带 + 颜色聚类）…")
        try:
            import sketch as _sk
            from PIL import Image as _ImgS

            # 审计 B-11: 中间产物写进系统临时目录，不污染用户输出目录。
            with _ImgS.open(src) as _ims:
                _svg, _info = _sk.build(_ims.convert("RGB"), bands=6,
                                        workdir=tempfile.gettempdir())
            if _svg:
                io.open(dst, "w", encoding="utf-8", newline="\n").write(_svg)
                step("线稿模式完成（%d 层）" % _info.get("bands", 0))
                return dict(src=src, dst=dst, mode=MODE_SKETCH,
                            grid="%dx%d" % (_info.get("width", 0), _info.get("height", 0)),
                            colors=_info.get("bands", 0), hd="",
                            # 审计 B-02: 返回 SVG 内真实 <path> 数，而非色带层数 bands。
                            paths=count_paths(dst),
                            bytes=os.path.getsize(dst), seconds=round(time.time() - t0, 1))
        except Exception as exc:  # noqa: BLE001
            # 审计 B-01: 实际落到下方通用 vtracer 分支，文案需诚实，不谎称回退 lumina。
            step("线稿模式失败（%s），回退到通用描摹引擎" % exc)

    if s.mode in (MODE_OWN, MODE_OWN_LAB):
        # self-developed tracer: OKLab palette -> mode filter -> thin-stroke
        # unification (unify_ink_cc) -> per-layer sub-pixel iso-contours ->
        # Schneider cubic fitting.  Ink layers skip `grow` so strokes stay thin.
        # MODE_OWN uses the production engine (own_tracer.py); MODE_OWN_LAB
        # uses the lab build (own_tracer_lab.py, same algorithm).
        if s.mode == MODE_OWN_LAB:
            import own_tracer_lab as own_tracer
            step("分层描摹（实验引擎）%d 色…" % s.colors)
        else:
            import own_tracer
            step("分层描摹 %d 色…" % s.colors)
        from PIL import Image as _OwnImg

        _arr = np.asarray(_OwnImg.open(src).convert("RGB"))
        res = own_tracer.trace(_arr, colors=int(s.colors), detail=int(s.detail))
        with io.open(dst, "w", encoding="utf-8") as fh:
            fh.write(res["svg"])
        step("完成 %d 层" % res["paths"])
        return {"paths": res["paths"], "bytes": len(res["svg"]),
                "colors": res["colors"], "mode": s.mode, "ms": res["ms"]}

    if s.mode == MODE_GEOM:
        # 几何化重绘（共边引擎）：硬标签量化 → 区域规整 → 共享像素边界等值线
        # → 角点切开 + L/A/Q 长曲线拟合。相邻块共享同一条边界、各自外扩重叠，
        # 从根上消除放大锯齿与亚像素缝（对标嘻嘻宝的长圆弧大色块范式）。
        # 详见 geom_shared.py / HANDOFF §10。
        step("几何化重绘引擎（共边：量化 → 区域规整 → 共享边界 → L/A/Q 拟合）…")
        import geom_shared as _gs
        res = _gs.trace(src, dst, colors=int(s.geom_colors),
                        tol=float(s.geom_tol), expand=float(s.geom_expand),
                        seam=float(s.geom_seam), progress=step)
        return dict(src=src, dst=dst, mode=s.mode,
                    grid="%dx%d" % (res["width"], res["height"]),
                    colors=res.get("colors", 0), hd="", paths=res["paths"],
                    bytes=os.path.getsize(dst) if dst else res["bytes"],
                    seconds=round(time.time() - t0, 1))

    if s.mode in (MODE_SILK, MODE_SILK_CLEAN, MODE_SILK_PRINT):
        # v7.0 丝状顺滑引擎：一块一 path + 强制细分短 A/Q 链 + 稀有色合并。
        # 解决 v6.8 共边引擎「同色环合并 + 大圆弧拟合」把发丝糊掉/切碎的问题，
        # 对标嘻嘻宝的 Q/A 主导曲线家族。详见 geom_silk.py / HANDOFF §26。
        #
        # MODE_SILK_CLEAN 是针对「大面积平滑深色背景」的实测最优预设：
        # 旧分块 vtracer 在深红渐变上会留 265 色硬边碎块，此预设用
        # 96 色 + 强标签平滑 + 色差去斑把暗色碎块从 8.91% 降到 5.94%，
        # 枫叶/金线/光点等细节仍完整。用户显式给的参数优先于预设。
        _preset = (dict(colors=96, label_smooth=1.6,
                        contrast_de=0.16, contrast_area=6.0)
                   if s.mode in (MODE_SILK_CLEAN, MODE_SILK_PRINT) else {})
        if s.mode == MODE_SILK_PRINT:
            # 实测最优（1330x1773 红背景）：窄于线宽的面积 6.88% → 4.18%，
            # 微小区域总面积 1.811% → 0.313%，RMSE 仅 +0.86，细节不丢。
            # 阈值由 print_mm / line_mm / 图像长边推导，小图自动几乎不合并。
            _preset.update(print_mm=263.0, print_scale=1.0)
        _explicit = set(s.explicit or ())
        _c = _preset.get("colors", int(s.silk_colors))
        _ls = _preset.get("label_smooth", float(s.silk_smooth))
        _de = _preset.get("contrast_de", float(s.silk_contrast_de))
        _ca = _preset.get("contrast_area", float(s.silk_contrast_area))
        _pm = _preset.get("print_mm", float(s.silk_print_mm))
        _ps = _preset.get("print_scale", float(s.silk_print_scale))
        _bs = float(s.silk_band_sigma)
        if "silk_colors" in _explicit: _c = int(s.silk_colors)
        if "silk_smooth" in _explicit: _ls = float(s.silk_smooth)
        if "silk_contrast_de" in _explicit: _de = float(s.silk_contrast_de)
        if "silk_contrast_area" in _explicit: _ca = float(s.silk_contrast_area)
        if "silk_print_mm" in _explicit: _pm = float(s.silk_print_mm)
        if "silk_print_scale" in _explicit: _ps = float(s.silk_print_scale)
        step("丝状顺滑引擎%s（一块一 path → 短 A/Q 链 → 去斑%s）…"
             % ({"silk": "", "silkclean": "· 大色块背景预设",
                 "silkprint": "· 可打印预设"}.get(s.mode, ""),
                " + 按线宽清碎区" if _pm > 0 else ""))
        import geom_silk as _gsilk
        res = _gsilk.trace(src, dst, colors=_c,
                           tol=float(s.silk_tol), max_chord=float(s.silk_chord),
                           label_smooth=_ls,
                           rare_frac=float(s.silk_rare),
                           contrast_de=_de,
                           contrast_area=_ca,
                           print_mm=_pm, print_scale=_ps,
                           band_sigma=_bs,
                           fit_mode=s.silk_fit, progress=step)
        return dict(src=src, dst=dst, mode=s.mode,
                    grid="%dx%d" % (res["width"], res["height"]),
                    colors=res.get("colors", 0), hd="", paths=res["paths"],
                    bytes=os.path.getsize(dst) if dst else res["bytes"],
                    seconds=round(time.time() - t0, 1))

    if s.mode == MODE_QQ3D:
        # qq3d 同款保真引擎：先感知量化把渐变成色带，再把 vtracer 重新求出的
        # 上千近似色吸附回色板，最后样条描摹。实测同一张写实动漫图 ΔE
        # 由 lumina 的 6.67 降到 4.25（256 色）。细节见 HANDOFF.md §12/§13。
        step("保真还原引擎（保真高清化 → 预量化 → 色板吸附 → 顺滑重拟合）…")
        import qq3d_like as _qq

        # 色数低于 256 会明显损失渐变（实测 160 色 ΔE 5.00 vs 256 色 4.12），
        # 所以该模式下把 128/160 这类默认值抬到 256；用户手动拉高则尊重。
        # 例外：打印模式下色块数 = 换料次数，256 次换料实际不可行（参考工具
        # 中位 78 色），因此强制夹到 ≤80。打印瘦身会再合并窄带。
        if float(s.print_mm or 0) > 0:
            ncol = max(16, min(80, int(s.colors)))
            step("打印模式：色块数限制为 %d（减少换料次数）" % ncol)
        else:
            ncol = max(256, int(s.colors))

        # 流程要求：原图先过一遍高清化补足模糊处，再转 SVG。
        # 用 hd_faithful 而不是旧的 aihd.preprocess —— 后者会把半尺寸模糊图
        # 以 35% 混进原图，实测颜色偏移 4.01 / 下采样误差 7.76，等于改了原图。
        _work, _hd_tmp, _scale, _disp = src, None, 1.0, None
        if s.faithful_hd:
            try:
                import hd_faithful as _hd
                if _hd.aihd.available():
                    from PIL import Image as _ImgHD
                    from ort_backend import describe as _ort_desc
                    _sc = max(1.25, float(s.hd_scale or 2.0))
                    step("AI 推理后端：%s" % _ort_desc())
                    step("保真高清化 %.1f×（只补模糊 / 不改原有清晰度）…" % _sc)
                    with _ImgHD.open(src) as _srcim:
                        _disp = _srcim.size
                        _big, _m = _hd.enhance(_srcim.convert("RGB"),
                                               scale=_sc, progress=progress)
                    _hd_tmp = _tmp_png("qq3d_hd")
                    _big.save(_hd_tmp)
                    _work, _scale = _hd_tmp, float(_m.get("scale", _sc))
                    step("清晰度校验 一致性 %.2f / 颜色偏移 %.2f / 清晰区 %.1f%% / 模糊区 +%.1f%%"
                         % (_m.get("down_rmse", 0), _m.get("color_shift", 0),
                            _m.get("sharp_clear_keep", 100), _m.get("sharp_blur_gain", 0)))
            except Exception as exc:  # noqa: BLE001
                step("保真高清化跳过（%s）" % exc)
                _work, _scale, _disp = src, 1.0, None
        try:
            res = _qq.trace(_work, dst, colors=ncol, profile=s.profile,
                            edge_smooth=s.edge_smooth, scale=_scale,
                            print_mm=float(s.print_mm or 0.0), line_mm=float(s.line_mm or 0.42),
                            self_line=bool(getattr(s, 'self_line', False)),
                            display_size=_disp,
                            seamless=bool(getattr(s, "seamless", True)),
                            seam_width=0.0, progress=progress)
        finally:
            if _hd_tmp:
                try:
                    os.remove(_hd_tmp)
                except OSError:
                    pass
        step("完成 %d 色块 / %d 色（%s 档 / 边缘 %s / 高清 %.1f×）"
             % (res["paths"], res["colors"], s.profile, s.edge_smooth, _scale))
        return dict(src=src, dst=dst, mode=s.mode,
                    grid="%dx%d" % (res["width"], res["height"]),
                    colors=res["colors"], hd="", paths=res["paths"],
                    bytes=os.path.getsize(dst) if dst else res["bytes"],
                    seconds=round(time.time() - t0, 1))

    if s.mode in (MODE_LUMINA, MODE_AI_LINE):
        # MODE_AI_LINE = lumina 底色描摹 + 叠加自研形态学线条层（分支末尾处理）。
        # vtracer (cutout, spline) produces complete coverage and keeps gradients;
        # the lumina engine stays as a fallback only.
        step("vtracer 引擎（cutout 共边 + 样条拟合）…")
        import vtracer as _vt
        from PIL import Image as _ImgV

        _work = src
        _hd_tmp = None
        if s.hd:
            try:
                # v7.5.7: 超分从 Real-ESRGAN (aihd) 换成颜色安全的 enhance 同款
                # 「只补模糊、不动原色」超分。实测 Real-ESRGAN 会把饱和度压掉
                # ~20%、亮度压低 ~5%（原图 0.384 → 0.305），导致动漫/照片图
                # 转 SVG 后整体发淡；enhance 同款 2× 后饱和度 0.384 → 0.383，
                # 几乎零损失且仍保留 2× 细节。
                if _hd_enhance is None:
                    raise RuntimeError("高清化模块不可用（enhance.py 缺失）")
                from PIL import Image as _Img

                _sc = max(1.25, float(s.hd_scale or 2.0))
                step("高清化 %.1f×（保色，不洗淡）…" % _sc)
                with _Img.open(src) as _im:
                    _big = _hd_enhance(_im.convert("RGB"), scale=_sc,
                                       strength=0.0, denoise=1.0)
                # vtracer 按超分后尺寸描摹；超过 500 万像素则回缩，避免路径数 /
                # 内存暴涨（与旧 aihd 5M 上限一致）。
                _cap = 5_000_000
                if _big.width * _big.height > _cap:
                    _r = (_cap / float(_big.width * _big.height)) ** 0.5
                    _big = _big.resize(
                        (max(1, int(_big.width * _r)),
                         max(1, int(_big.height * _r))),
                        Image.LANCZOS)
                _hd_tmp = _tmp_png("vt")
                _big.save(_hd_tmp)
                _work = _hd_tmp
            except Exception as exc:  # noqa: BLE001
                step("高清化跳过（%s）" % exc)

        try:
            # v7.6.1 画质参数（从桌面版 backport，2026-11-16）：
            # 保色超分 + case7 描摹参数（color_precision=8 / layer_diff=8 /
            # corner=86 / filter_speckle=4）。实测同一张高细节插画：
            # 旧参数脸发灰、发丝糊成浅色一片；case7 后脸部不发灰、镜框更黑、
            # 发丝层次清楚。颜色随后由 _merge_svg_colours 收束到 256 色大色块。
            # 不引入任何模型或二进制（模块 backend 为 text-only）。
            _vtrace(
                _work, dst,
                clustering="color", hierarchical="cutout", mode="spline",
                filter_speckle=4, color_precision=8, layer_difference=8,
                corner_threshold=86, length_threshold=3.0,
                max_iterations=10, splice_threshold=45, path_precision=2)
        finally:
            if _hd_tmp:
                try:
                    os.remove(_hd_tmp)
                except OSError:
                    pass

        txt = io.open(dst, encoding="utf-8").read()
        # vtracer writes width/height only; a viewBox is required for correct zooming
        try:
            _m = re.search(r"<svg([^>]*)>", txt)
            if _m and "viewBox" not in _m.group(1):
                _mm = re.search(r'width="([\d.]+)"\s+height="([\d.]+)"', _m.group(1))
                if _mm:
                    txt = txt.replace(_m.group(0),
                                      '<svg%s viewBox="0 0 %s %s">'
                                      % (_m.group(1), _mm.group(1), _mm.group(2)), 1)
        except Exception:  # noqa: BLE001
            pass
        paths = txt.count("<path")
        # collapse near-identical colours: shapes keep their detail, the palette
        # is brought back to the requested colour count
        try:
            txt = _merge_svg_colours(txt, int(s.colors), _work)
        except Exception:  # noqa: BLE001
            pass
        try:
            _sw = 0.6

            def _stroke(m):
                _attrs = m.group(1)
                _f = re.search(r'fill="(#[0-9a-fA-F]{6})"', _attrs)
                if not _f:
                    return m.group(0)
                return ('<path%s stroke="%s" stroke-width="%.2f" stroke-linejoin="round"/>'
                        % (_attrs, _f.group(1), _sw))

            txt = re.sub(r"<path([^>]*?)/>", _stroke, txt)
        except Exception:  # noqa: BLE001
            pass
        # backdrop + width, so no hairline gap ever shows white
        try:
            with _ImgV.open(_work) as _imw:
                _arr = np.asarray(_imw.convert("RGB").resize((120, 120)), np.float32)
            # 审计 §4.1: 原写在 float32 数组上做 >>4，会抛 TypeError 被 except 吞掉，
            # 导致防露白背景 rect 静默失效。先转 uint8 再取高 4 位。
            _q = (_arr.astype(np.uint8).reshape(-1, 3) >> 4).astype(np.int32)
            _key = (_q[:, 0] << 8) | (_q[:, 1] << 4) | _q[:, 2]
            _vals, _counts = np.unique(_key, return_counts=True)
            _k = int(_vals[_counts.argmax()])
            _bg = "#%02x%02x%02x" % (((_k >> 8) & 15) * 16 + 8,
                                     ((_k >> 4) & 15) * 16 + 8, (_k & 15) * 16 + 8)
            _m = re.search(r'width="(\d+)"\s+height="(\d+)"', txt)
            if _m:
                _w, _h = int(_m.group(1)), int(_m.group(2))
                # v7.5.4 / 审计 B-04: vtracer 0.6 输出以 <?xml ...?> 与
                # <!-- Generator --> 开头，替换「全文第一个 >」会把 rect
                # 插到根元素之外、产出非法 XML。定位根 <svg ...> 结束处再插。
                _root = re.search(r"<svg\b[^>]*>", txt)
                if _root:
                    _rect = ('<rect x="0" y="0" width="%d" height="%d" fill="%s"/>'
                             % (_w, _h, _bg))
                    txt = txt[:_root.end()] + _rect + txt[_root.end():]
        except Exception:  # noqa: BLE001
            pass
        io.open(dst, "w", encoding="utf-8", newline="\n").write(txt)
        step("vtracer 完成（%d 路径）" % paths)
        if s.straighten:
            try:
                import poststraight as _ps2

                txt = io.open(dst, encoding="utf-8").read()
                txt, _st = _ps2.straighten_svg(txt, tol=0.6)
                io.open(dst, "w", encoding="utf-8", newline="\n").write(txt)
                step("拉直 %d 段" % _st["curves_straightened"])
            except Exception as exc:  # noqa: BLE001
                step("直线化跳过（%s）" % exc)
        if s.mode == MODE_AI_LINE:
            return _apply_ai_line(
                s, src, dst,
                dict(src=src, dst=dst, mode=s.mode, grid="", colors=s.colors,
                     hd=("%dx%d" % (_big.width, _big.height)) if _hd_tmp else "",
                     paths=paths, bytes=os.path.getsize(dst),
                     seconds=round(time.time() - t0, 1)))
        return dict(src=src, dst=dst, mode=s.mode, grid="", colors=s.colors,
                    hd=("%dx%d" % (_big.width, _big.height)) if _hd_tmp else "",
                    paths=paths, bytes=os.path.getsize(dst),
                    seconds=round(time.time() - t0, 1))

    if s.mode == MODE_PTRACE:
        # 参考文件都是「先高清化，再描摹」：放大 + 边界锐化后再切色块，
        # 边缘才会像 家教纲吉.svg / P5合集.svg 那样干净。
        from enhance import enhance as _enh

        # 审计 B-05: 关闭高清时不应再强制 2× 上采样，否则「关高清」无效。
        hd_scale = s.hd_scale if s.hd else 1.0
        step("高清化 %.1f× …" % hd_scale)
        im = _enh(im, scale=hd_scale, strength=0.0, denoise=1.0)
        step("potrace 描摹（%d 色 · %d 网格）…" % (s.colors, min(im.width, 3200)))
        from ptr import vectorise as _ptr_vectorise

        svg_text, rinfo = _ptr_vectorise(im, colors=s.colors, grid=min(im.width, 3200),
                                         median=3, turd=4, alphamax=1.0, opttol=0.2,
                                         gap=(2.0 if s.gap else 0.0), grow=0,
                                         smooth_r=0, merge_area=300, merge_de=0.08,
                                         label_grid=0,
                                         # 审计 B-11: 中间产物写进系统临时目录。
                                         workdir=tempfile.gettempdir())
        gw, gh = rinfo["grid"]
        scale = orig_w / gw
        svg_text = _set_root(svg_text, gw, gh, scale,
                             "potrace colour artwork (%dx%d grid)" % (gw, gh))
        io.open(dst, "w", encoding="utf-8", newline="\n").write(svg_text)
        step("完成")
        return dict(src=src, dst=dst, mode=s.mode, grid="%dx%d" % (gw, gh),
                    colors=rinfo["colors"], hd="", paths=rinfo["contours"],
                    bytes=os.path.getsize(dst), seconds=round(time.time() - t0, 1))

    if s.mode == MODE_REGION:
        step("区域矢量化（128 色 · 硬边 · 无缝）…")
        from regionvec import vectorise as _region_vectorise

        svg_text, rinfo = _region_vectorise(im, colors=s.colors, grid=1200,
                                            median=3, eps=0.9, min_area=8,
                                            gap=1.0, chaikin=0, min_hole=24)
        gw, gh = rinfo["grid"]
        scale = orig_w / gw
        svg_text = _set_root(svg_text, gw, gh, scale,
                             "Region vectorised artwork (%dx%d grid)" % (gw, gh))
        io.open(dst, "w", encoding="utf-8", newline="\n").write(svg_text)
        step("完成")
        return dict(src=src, dst=dst, mode=s.mode, grid="%dx%d" % (gw, gh),
                    colors=rinfo["colors"], hd="", paths=rinfo["contours"],
                    bytes=os.path.getsize(dst), seconds=round(time.time() - t0, 1))

    if s.mode == MODE_REF:
        step("参考风格：%d 色 · 1600 网格 · 补缝描边…" % max(32, s.colors))
        from cel import ref_style as _ref_style

        im, used = _ref_style(im, colors=max(32, s.colors))
        s.colors = used
    elif s.mode == MODE_PRINT:
        from cel import sharpen_lines as _sharpen_print

        scale = s.hd_scale if s.hd else 1.6
        step("高清化 %s× + 线条增强（打印用）…" % ("%.2f" % scale).rstrip("0").rstrip("."))
        im = _sharpen_print(im, denoise=0.35, line=0.45, sharp=0.4)
        if scale > 1.01:
            tw = min(int(round(im.width * scale)), 2600)
            th = max(1, int(round(im.height * tw / im.width)))
            im = im.resize((tw, th), Image.LANCZOS)
        grid = min(im.width, 2600)
    elif s.mode == MODE_CEL:
        step("色块化处理（%d 色 · 去噪 · 平滑边界）…" % s.colors)
        from cel import build as _cel_build

        cap = 2600 if s.hd else 1600
        im, used, _cent = _cel_build(im, colors=s.colors, max_width=cap)
        s.colors = used
    else:
        if s.mode == MODE_HIFI:
            from cel import sharpen_lines as _sharpen_lines

            im = _sharpen_lines(im, denoise=0.3, line=0.5, sharp=0.45)
        im = _fit_and_quantize(im, grid, s.colors if s.mode == MODE_FLAT else None)
    gw, gh = im.size

    tmpdir = tempfile.mkdtemp(prefix="imgsvg_")
    png = os.path.join(tmpdir, "trace_in.png")
    raw = os.path.join(tmpdir, "trace_out.svg")
    try:
        im.save(png, "PNG", optimize=False, compress_level=1)

        step("矢量化描摹中（%d×%d，%s）…" % (
            gw, gh,
            ("动漫色块 %d 色" % s.colors) if s.mode == MODE_CEL else
            ("分色 %d 色" % s.colors if s.mode == MODE_FLAT else "全彩")))
        if s.mode == MODE_REF:
            kw = _ref_kwargs(s)
        elif s.mode == MODE_PRINT:
            kw = _ref_kwargs(s)
        elif s.mode == MODE_CEL:
            kw = _cel_kwargs(s)
        elif s.mode == MODE_EXACT:
            kw = _exact_kwargs()
        else:
            kw = _vtracer_kwargs(s)
        _vtrace(png, raw, **kw)

        step("整理路径与画板…")
        svg = io.open(raw, encoding="utf-8").read()
        if s.mode == MODE_REF:
            svg = _add_seam_strokes(svg, 0.8)
        elif s.mode == MODE_PRINT:
            svg = _add_seam_strokes(svg, s.gap_width)
        elif s.gap and s.mode == MODE_FLAT:
            svg = _add_seam_strokes(svg, s.gap_width)
        scale = orig_w / gw if gw else 1.0
        svg = _set_root(svg, gw, gh, scale,
                        "Vectorised illustration (%d×%d grid)" % (gw, gh))
        io.open(dst, "w", encoding="utf-8", newline="\n").write(svg)

        if s.optimize:
            step("压缩优化 SVG…")
            opt = os.path.join(tmpdir, "opt.svg")
            if _optimize_svg(dst, opt):
                svg_opt = io.open(opt, encoding="utf-8").read()
                svg_opt = _set_root(svg_opt, gw, gh, scale,
                                    "Vectorised illustration (%d×%d grid)" % (gw, gh))
                io.open(dst, "w", encoding="utf-8", newline="\n").write(svg_opt)
    finally:
        for p in (png, raw):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    info = dict(
        src=src,
        dst=dst,
        mode=s.mode,
        grid="%dx%d" % (gw, gh),
        colors=(s.colors if s.mode in (MODE_FLAT, MODE_CEL, MODE_PRINT) else 0),
        hd=("%.2gx" % s.hd_scale) if s.hd else "",
        paths=count_paths(dst),
        bytes=os.path.getsize(dst),
        seconds=round(time.time() - t0, 1),
    )
    step("完成")
    return info


def _apply_ai_line(s: "Settings", src: str, dst: str, st: dict) -> dict:
    """Append the self-developed morphology line layer on top of the traced SVG.

    v7.5.3: no torch, no third-party weights, no vendored code.  If OpenCV is
    unavailable we simply return the base stats so the colour SVG is untouched.
    """
    try:
        import ai_ink
    except Exception as exc:
        st["ai_line"] = "import_failed:%s" % exc
        return st
    if not ai_ink.available():
        st["ai_line"] = "cv2_missing"
        return st
    try:
        from PIL import Image as _I
        with _I.open(src) as _im:
            _rgb = np.asarray(_im.convert("RGB"))
        _txt = io.open(dst, encoding="utf-8").read()
        _layer = ai_ink.build_line_layer(_rgb, _txt)
        if _layer:
            _idx = _txt.rfind("</svg>")
            if _idx >= 0:
                _txt = _txt[:_idx] + _layer + _txt[_idx:]
                io.open(dst, "w", encoding="utf-8", newline="\n").write(_txt)
                st["ai_line_paths"] = _layer.count("<path")
                st["ai_line"] = "ok"
            else:
                st["ai_line"] = "no_svg_close"
        else:
            st["ai_line"] = "empty"
    except Exception as exc:  # noqa: BLE001
        st["ai_line"] = "error:%s" % exc
    return st
