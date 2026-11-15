"""自研「保真还原」引擎 —— 复刻 qq3d 转换器的视觉效果。

逆向 qq3d 参考 SVG 后归纳出的四条关键技术：

1. 预量化(posterize)：先把原图量化成 64~256 级相近纯色，让平滑渐变/光泽/
   柔影变成一串离散色带。vtracer 直接吃连续渐变时，每个连通区只取一个平均
   色，会把所有明暗压平（实测 RAW 只有 3 色，量化后 22~26 色）。
2. 边缘保持降噪：先清掉 JPEG 孤立噪点，否则它们会变成一堆"杂色块"。
3. 样条描摹：vtracer spline + 贝塞尔拟合，边界是曲线而非折线，不锯齿。
4. 成对补缝(qq3d-gap-filler)：每个 fill path 紧跟一条同 d 的 stroke-only
   path（fill=none / stroke=同色 / stroke-width=2 / linejoin=round）。它把
   相邻色块之间 1~2px 的抗锯齿缝隙用区域自身颜色填上，同时把边界外扩 1px，
   因此线条既不变粗也不变细、看不到白缝、边缘平滑。

只依赖 numpy / Pillow / vtracer，便于打包进现有 exe。
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import threading
import time

import numpy as np
from PIL import Image, ImageFilter


def _start_heartbeat(step, label: str, interval: float = 15.0):
    """长时间阻塞阶段（vtracer / 曲线重拟合）期间定期回报进度。

    只发消息、不碰任何计算，因此输出 SVG 与无心跳时逐字节一致。GUI 在工作
    线程里把这些消息放进 queue，避免 80 秒无声让用户以为程序卡死。
    """
    stop = threading.Event()
    t0 = time.time()

    def beat():
        while not stop.wait(interval):
            try:
                step("%s 中… 已 %.0f 秒" % (label, time.time() - t0))
            except Exception:  # noqa: BLE001
                return

    threading.Thread(target=beat, daemon=True).start()
    return stop.set

# --------------------------------------------------------------------------- #
# OKLab 感知色空间（与 core.py 保持一致，这里复制一份使引擎可独立运行）
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
    # 与 core.py 同步（审计 B-12）: 极端输入可能在 cbrt / 线性组合后产生 nan/inf。
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


def _kmeans_palette(pix: np.ndarray, k: int, iters: int = 16, seed: int = 7) -> np.ndarray:
    """k-means++ 初始化，在 OKLab 空间中聚类。"""
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
    return cent


def quantize_perceptual(im: Image.Image, colors: int, seed: int = 7,
                        return_labels: bool = False):
    """把连续渐变量化成 N 级纯色（关键步骤：保留光泽/柔影）。

    return_labels=True 时额外返回 (标签图, 色板数组)，供区域规整使用。
    """
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    lab = _srgb_to_oklab(arr)
    cent = _kmeans_palette(lab.reshape(-1, 3), max(2, int(colors)), seed=seed)
    pal = (_oklab_to_rgb01(cent) * 255.0).round().astype(np.uint8)

    # 3D LUT 一次性算好映射，避免逐像素算 N 个距离
    g = 48
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
    out = Image.fromarray(pal[idx], "RGB")
    if return_labels:
        return out, idx, pal
    return out


def defringe(im: Image.Image, strength: float = 1.0, ksize: int = 7,
             threshold: float = 2.0, softness: float = 6.0) -> Image.Image:
    """压掉边缘 1–3 px 的**色度尖峰**（JPEG 色差 / 抗锯齿青紫彩边）。

    与 ``denoise`` 的区别：去噪只在「非边缘」处替换孤立噪点；青紫彩边恰恰
    出现在**真实边缘**上，denoise 判据会把权重压到 0，所以治不了它。
    这里把 RGB 转到 YUV，对 U/V 做 5–7 px 中值得到「这块本该有的颜色」，
    只把明显偏离的色度拉回中值，**亮度 Y 一字节不动** —— 边缘锐度、明暗
    层次、真实大面积颜色全部保留，只有 1–3 px 的彩色过渡被中性化，于是
    量化不会再把它们固化成独立的青/紫色块。

    实测（原尺寸，177 / 362cb）：mean|dY| 0.022 / 0.023（≈0），
    OKLab dE 0.0107 / 0.0069，色度尖峰像素占比 2.3% → 0.9%。
    """
    if strength <= 0:
        return im
    import cv2
    a = np.asarray(im.convert("RGB"), np.uint8)
    yuv = cv2.cvtColor(a, cv2.COLOR_RGB2YUV).astype(np.float32)
    uv = yuv[..., 1:]
    med = np.stack([cv2.medianBlur(uv[..., i].astype(np.uint8), ksize)
                    for i in range(2)], -1).astype(np.float32)
    d = np.abs(uv - med).mean(2)
    w = np.clip((d - threshold) / softness, 0.0, 1.0) * float(strength)
    yuv[..., 1:] = uv * (1.0 - w[..., None]) + med * w[..., None]
    return Image.fromarray(
        cv2.cvtColor(np.clip(yuv, 0, 255).astype(np.uint8), cv2.COLOR_YUV2RGB),
        "RGB")


def denoise(im: Image.Image, strength: float = 1.0) -> Image.Image:
    """边缘保持降噪：只替换孤立噪点，保留细线与高光。

    判据：某像素明显偏离邻域中值（噪点），但邻域中值与局部均值接近
    （不是真实边缘）时才平滑。这样不会把发丝、高光、描边糊掉。
    """
    if strength <= 0:
        return im
    a = np.asarray(im.convert("RGB"), np.float32)
    med = np.asarray(im.filter(ImageFilter.MedianFilter(3)), np.float32)
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(1.0)), np.float32)

    d_noise = np.abs(a - med).mean(2)      # 单像素偏离邻域 → 噪点
    d_edge = np.abs(med - blur).mean(2)    # 邻域结构变化 → 真实边缘

    t = np.clip((d_noise - 4.0) / 18.0, 0, 1) * np.clip((14.0 - d_edge) / 14.0, 0, 1)
    t = (t * strength)[..., None]
    out = a * (1 - t) + med * t
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def _drop_fringe_labels(labels: np.ndarray, pal: np.ndarray,
                        min_same: int = 2,
                        chroma_floor: float = 0.05) -> np.ndarray:
    """⚠ 实验函数，**未被产品流程采用**（失败原因见 HANDOFF §13）。保留作参考。

    色板层清 1px 边界过渡色（青紫彩边），**不改输入图、不改色板数组**。

    v6.7 的 ``defringe`` 在 YUV 上压色度尖峰，但量化仍会把残余过渡色固化成
    独立色板项 → 红/白边界还剩一条淡青紫线。诊断（``_dev/probe_palette.py``）
    显示：这些过渡色像素在空间上几乎全是**沿两种主色边界的 1px 细线**，
    而真实细节色（粉色花、金色装饰）是块状的。所以判据用空间形态而非面积：

    * ``same_neighbors <= 2``：8 邻域里同色像素不超过 2 个 = 孤立像素 / 细线；
    * ``OKLab 色度 > 0.05``：只清彩色过渡，保护灰阶描边线（实测不加这条会
      把衣服上的灰色轮廓线抹虚，见 ``palette_merge_probe.png`` 的 B/D 格）。

    命中像素替换成 3×3 邻域中值 label；色板数组不变，只是某些色板项不再被
    使用 → 输出色数自然下降（换料更少）。被误伤的极细真彩线由后续
    ``unify_ink_cc`` 在真实图上补回，与 ``_label_smooth`` 的既有设计一致。
    """
    from scipy import ndimage as ndi

    H, W = labels.shape
    same = np.zeros((H, W), np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            same += (np.roll(labels, (dy, dx), axis=(0, 1)) == labels)
    thin = same <= min_same
    if chroma_floor > 0:
        lab = _srgb_to_oklab(pal[None, :, :].astype(np.float32) / 255.0)[0]
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        thin &= chroma[labels] > chroma_floor
    if not thin.any():
        return labels
    med = ndi.median_filter(labels, size=3)
    out = labels.copy()
    out[thin] = med[thin]
    return out


def _label_smooth(labels: np.ndarray, k: int, sigma: float,
                  protect: np.ndarray | None = None) -> np.ndarray:
    """逐色高斯投票（软多数决）：清掉孤立杂色斑、把像素阶梯磨圆，且所有区域
    共享同一条边界（天然无接缝）。细暗线在这里会被侵蚀，但随后由
    unify_ink_cc 在真实图上检测并补回，故线条不糊。

    protect: 原图梯度算出的「真实边缘」掩码。这些像素保留原标签，避免加大
    sigma 后把轮廓 / 发丝 / 花纹一起糊掉（HANDOFF §23）。默认 None 时与旧版
    逐字节一致。
    """
    from scipy.ndimage import gaussian_filter
    H, W = labels.shape
    best_score = np.zeros((H, W), np.float32)
    best_label = np.zeros((H, W), np.int32)
    for c in range(k):
        mask = (labels == c).astype(np.float32)
        if not mask.any():
            continue
        sm = gaussian_filter(mask, sigma, mode="nearest")
        upd = sm > best_score
        best_score[upd] = sm[upd]
        best_label[upd] = c
    if protect is not None:
        return np.where(protect, labels, best_label)
    return best_label


# --------------------------------------------------------------------------- #
# SVG 组装：qq3d 式「填充 + 同色补缝」成对 path
# --------------------------------------------------------------------------- #

_PATH_RE = re.compile(r"<path\b([^>]*?)/>")
_ATTR_RE = re.compile(r'(\w[\w-]*)="([^"]*)"')


def _rgb_hex(c: str) -> str | None:
    c = c.strip().lower()
    if re.fullmatch(r"#[0-9a-f]{6}", c):
        return c
    if re.fullmatch(r"#[0-9a-f]{3}", c):
        return "#" + "".join(ch * 2 for ch in c[1:])
    return None


def _stroke_color(fill: str, alpha: float) -> str:
    h = _rgb_hex(fill)
    if h is None:
        return fill
    if alpha >= 0.999:
        return h
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return "rgba(%d,%d,%d,%.3f)" % (r, g, b, alpha)


def _palette_snap(palette: list[str]):
    """构造 hex -> 色板最近色 的映射（在 OKLab 空间比较）。

    vtracer 会给每个区域重新求平均色，输出上千个近似色；吸附回量化色板
    才能保证颜色忠于原图、且色数可控（qq3d 的颜色准就靠这一步）。
    """
    cent = _srgb_to_oklab(
        np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)]
                  for c in palette], np.float32) / 255.0
    )
    cache: dict[str, str] = {}
    c2 = (cent ** 2).sum(1)

    def snap(h: str) -> str:
        if h in cache:
            return cache[h]
        v = _srgb_to_oklab(
            np.array([[[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]]], np.float32) / 255.0
        ).reshape(1, 3)[0]
        d = (v ** 2).sum() - 2 * cent @ v + c2
        out = palette[int(d.argmin())]
        cache[h] = out
        return out

    return snap


def _consolidate_ink(labels: np.ndarray, pal: np.ndarray,
                     created: list[int], k: int = 3) -> tuple:
    """把「每条暗线一个颜色」压成 k 档统一墨线色。

    ``unify_ink_cc`` 会把每个暗线连通域分配一个独立 label + 它自己的平均色。
    在细线密集的插画上这会追加**几百**种颜色（用户两张样张实测 447 / 428 色，
    其中 ~190 色来自墨线），每条线又各自成为一个小 path，4× 放大后就是
    深色网状碎块和彩色锯齿边。

    这里按**亮度面积加权分位数**把连通域聚成 k 档（默认深/中/浅墨），同档
    共用一个 label 与一个颜色：vtracer 于是把它们合成连续线条层，色数从
    447 降到 ~259，路径数与碎块同步大减，棕色发丝与黑色描边仍能分开。
    返回 ``(labels, pal, extra_ids)``。
    """
    if not created:
        return labels, pal, set()
    n = len(created)
    base = int(created[0])
    cols = pal[-n:].astype(np.float32)
    lum = 0.299 * cols[:, 0] + 0.587 * cols[:, 1] + 0.114 * cols[:, 2]
    counts = np.bincount(labels.ravel(), minlength=len(pal))
    areas = counts[base:base + n].astype(np.float64)
    total = float(areas.sum()) or 1.0
    order = np.argsort(lum)
    cs = np.cumsum(areas[order])
    cuts = np.searchsorted(cs, total * np.arange(1, k) / k)
    sections = [sec for sec in np.split(order, cuts) if len(sec)]

    lut = np.arange(int(labels.max()) + 1, dtype=np.int32)
    new_cols: list[np.ndarray] = []
    for gi, sec in enumerate(sections):
        w = areas[sec]
        col = (cols[sec] * w[:, None]).sum(0) / max(float(w.sum()), 1.0)
        new_cols.append(col.round().astype(np.uint8))
        for j in sec:
            lut[base + int(j)] = base + gi
    return (lut[labels],
            np.concatenate([pal[:base], np.array(new_cols, np.uint8)]),
            set(range(base, base + len(sections))))


def build_svg(raw_svg: str, width: int, height: int, bg: str,
              seam_width: float, seam_alpha: float,
              palette: list[str] | None = None) -> tuple[str, int, int]:
    """把 vtracer 输出改写成 qq3d 结构，返回 (svg, 填充路径数, 颜色数)。"""
    fills: list[str] = []
    colors: set[str] = set()
    nfill = 0
    snap = _palette_snap(palette) if palette else (lambda h: h)

    # 分块并行给每块瓦片补的底矩形（<rect>，不是 <path>）：颜色吸附回色板，
    # 放在全局底 rect 之后、所有填充 path 之前。全局 rect 被它们完全盖住，
    # 因此瓦片内没被 path 覆盖的「洞」显示的是本区域主色，而不是全图底色。
    tile_rects: list[str] = []
    for m in re.finditer(r"<rect\b[^>]*/>", raw_svg):
        tag = m.group(0)
        fm = re.search(r'fill="(#[0-9a-fA-F]{6})"', tag)
        if fm:
            h = _rgb_hex(fm.group(1))
            if h:
                tag = tag.replace(fm.group(1), snap(h))
        tile_rects.append(tag)

    for m in _PATH_RE.finditer(raw_svg):
        attrs = dict(_ATTR_RE.findall(m.group(1)))
        d = attrs.get("d")
        f = attrs.get("fill")
        if not d or not f:
            continue
        h = _rgb_hex(f)
        if h is None:  # 跳过透明填充 / none
            continue
        h = snap(h)
        extra = ' transform="%s"' % attrs["transform"] if attrs.get("transform") else ""
        # 1) 主填充区域
        fills.append('<path d="%s" fill="%s"%s/>' % (d, h, extra))
        nfill += 1
        # 2) 同 d 的补缝描边（qq3d-gap-filler）。密集碎块上叠加半透明描边
        #    会累积变暗形成脏斑，默认改用不透明同色（qq3d 卡布达版写法）。
        if seam_width > 0:
            fills.append(
                '<path class="qq3d-gap-filler" d="%s" fill="none" stroke="%s" '
                'stroke-width="%.2f" stroke-linejoin="round"%s/>'
                % (d, _stroke_color(h, seam_alpha), seam_width, extra)
            )
        colors.add(h)

    head = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        'width="%d" height="%d" viewBox="0 0 %d %d">\n'
        '<rect x="0" y="0" width="%d" height="%d" fill="%s"/>\n'
        % (width, height, width, height, width, height, bg)
    )
    if tile_rects:
        head += "\n".join(tile_rects) + "\n"
    return head + "\n".join(fills) + "\n</svg>\n", nfill, len(colors)


# ----------------------------------------------------------------------------
# 样条描摹：把量化后的大图切成网格瓦片，每块独立跑 vtracer 后拼回。
# vtracer 是纯单线程 Rust（无 threads 参数、无并行后端），占整条管线 ~40%，
# 是"没用上多核"的真正大头。每块瓦片用 <path transform="translate(dx,dy)">
# 平移回原坐标，下游 build_svg 已支持逐路径 transform，无需改动。瓦片带重叠，
# 保证跨边界的色块被完整描入某一块、不会在接缝处被裁断，因此画质等价于非分块
# 结果（重叠区只是同色重复描边，渲染无差）。IMG2SVG_NO_TILE=1 可强制单进程回退。
# ----------------------------------------------------------------------------
_VTR_PATH_RE = re.compile(r"<path\b([^>]*?)/>")
_PATH_TOK_RE = re.compile(r"([a-zA-Z])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _fmt_num(x: float) -> str:
    s = "%.2f" % float(x)
    return s.rstrip("0").rstrip(".") if "." in s else s


def _translate_path_d(d: str, dx: float, dy: float) -> str:
    """把路径 d 里所有坐标整体平移 (dx, dy)。dx/dy 为整像素，绝对精确。

    直接在坐标数字上平移（而非用 transform 属性），这样非重叠区的路径与
    单进程 vtracer 输出逐坐标一致、只是平移了整数，渲染结果逐字节等价；
    重叠区只是同色重复描边。vtracer 输出为绝对命令（M/C/L/Z…），这里同时
    兼容相对命令（按当前点折算成绝对后再平移）。
    """
    toks = list(_PATH_TOK_RE.finditer(d))
    out = []
    cx = cy = 0.0
    cmd = None
    i = 0
    n = len(toks)

    def emit(v):
        out.append(" ")
        out.append(_fmt_num(v))

    while i < n:
        m = toks[i]
        if m.group(1):  # 命令字母
            cmd = m.group(1)
            out.append(cmd.upper())  # 一律按绝对命令输出
            i += 1
            if cmd.upper() == "M":
                cmd = "L"  # M 之后的成对坐标按 L 处理
            continue
        if cmd is None:
            i += 1
            continue
        C = cmd.upper()
        rel = cmd.islower()
        if C == "Z":
            i += 1
            continue
        if C == "H":
            size = 1
        elif C == "V":
            size = 1
        elif C == "A":
            size = 7
        else:
            size = 2 if C in ("M", "L", "T") else 4 if C in ("Q", "S") else 6
        vals = []
        for _ in range(size):
            if i >= n or toks[i].group(1):
                break
            vals.append(float(toks[i].group(0)))
            i += 1
        if not vals:
            continue
        if C == "H":
            ax = (cx + vals[0]) if rel else vals[0]
            ax += dx
            emit(ax); cx = ax
        elif C == "V":
            ay = (cy + vals[0]) if rel else vals[0]
            ay += dy
            emit(ay); cy = ay
        elif C == "A":
            x, y = vals[5], vals[6]
            ax = (cx + x) if rel else x
            ay = (cy + y) if rel else y
            ax += dx; ay += dy
            for v in (vals[0], vals[1], vals[2], vals[3], vals[4], ax, ay):
                emit(v)
            cx, cy = ax, ay
        else:
            for k in range(0, len(vals), 2):
                x, y = vals[k], vals[k + 1]
                ax = (cx + x) if rel else x
                ay = (cy + y) if rel else y
                ax += dx; ay += dy
                emit(ax); emit(ay)
                cx, cy = ax, ay
    return "".join(out)


def _vtrace_png(png, out, **kw):
    """Trace PNG->SVG with vtracer 0.6 (v7.5.6: vtracer10 path removed)."""
    import vtracer as _v06
    _kw = dict(kw)
    if "clustering" in _kw:
        _kw["colormode"] = _kw.pop("clustering")
    for _k in ("max_colors", "palette"):
        _kw.pop(_k, None)
    _v06.convert_image_to_svg_py(png, out, **_kw)


def _vtracer_tile_worker(job):
    """多进程 worker：对单块瓦片跑 vtracer，返回已平移回绝对坐标的 <path> 标签列表。

    注意：曾试过给瓦片四周加「哨兵色」边框，逼 vtracer 把主色也输出为 path。
    洞确实更小，但 vtracer 的整条颜色分层被改变，形状大面积变形、全图 RMSE
    反而从 15.89 劣化到 29.36（见 HANDOFF §21.3），**已回退，不要重试**。
    正确做法是在所有 path 之下补每块自己的底 <rect>（见 _vtracer_tiled）。
    """
    import os
    import tempfile
    from PIL import Image
    import vtracer

    crop, dx, dy, params = job
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        Image.fromarray(crop).save(tmp.name)
        raw = tmp.name + ".svg"
        _vtrace_png(tmp.name, raw, **params)
        s = io.open(raw, encoding="utf-8", errors="ignore").read()
    finally:
        for _p in (tmp.name, tmp.name + ".svg"):
            try:
                os.remove(_p)
            except OSError:
                pass
    out = []
    for m in _VTR_PATH_RE.finditer(s):
        attrs = m.group(1)
        if "transform=" in attrs:
            attrs = re.sub(r'transform="([^"]*)"',
                          lambda mm: 'transform="translate(%.6g %.6g) %s"' % (dx, dy, mm.group(1)),
                          attrs, count=1)
        else:
            attrs = attrs + ' transform="translate(%.6g %.6g)"' % (dx, dy)
        out.append("<path%s/>" % attrs)
    return out


def _tile_bg_hex(crop: np.ndarray) -> str:
    """瓦片主色（4-bit 量化众数）——vtracer 把这个色当「透明背景」不输出 path。

    分块并行时每个瓦片各有一个这样的背景，但全图只有一张全局 ``<rect>``；
    不补底的话，瓦片里没被 path 覆盖的区域会统一露出全局底色，在人物/装甲上
    形成粉色斑、白色空缺、黑块（见 HANDOFF §21）。这里把每块的主色补成
    独立底矩形，洞里就显示本区域该有的颜色。
    """
    a = crop.reshape(-1, 3).astype(np.int16)
    q = (a >> 4).astype(np.int32)
    key = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    cnt = np.bincount(key, minlength=4096)
    v = int(cnt.argmax())
    return "#%02x%02x%02x" % (((v >> 8) & 15) * 16 + 8,
                              ((v >> 4) & 15) * 16 + 8, (v & 15) * 16 + 8)


def _cell_bg_rgb(cell: np.ndarray) -> np.ndarray:
    """格子主色：先找 4-bit 众数桶，再取桶内像素的**精确**平均色。

    直接用桶中心（``_tile_bg_hex``）会让相邻格在放大后跳成两级色、出现明显
    方形块；取桶内真实均值后颜色连续得多，且该值本就来自量化图。
    """
    a = cell.reshape(-1, 3).astype(np.int16)
    q = (a >> 4).astype(np.int32)
    key = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    cnt = np.bincount(key, minlength=4096)
    v = int(cnt.argmax())
    sel = a[key == v]
    col = sel.mean(0).round() if len(sel) else a.mean(0).round()
    return col.astype(np.float32)


def _cell_bg_hex(cell: np.ndarray) -> str:
    c = _cell_bg_rgb(cell)
    return "#%02x%02x%02x" % (int(np.clip(c[0], 0, 255)),
                              int(np.clip(c[1], 0, 255)),
                              int(np.clip(c[2], 0, 255)))


def _quad_grid(grid: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               xs: list[int], ys: list[int], tau: float,
               out: list[str], depth: int = 0) -> None:
    """颜色一致的区域合成一张大 rect，只在真实颜色交界处四等分细分。"""
    sub = grid[y0:y1, x0:x1].reshape(-1, 3)
    spread = float(np.max(sub.max(0) - sub.min(0)))
    if (spread <= tau or depth >= 5 or (x1 - x0) <= 1 or (y1 - y0) <= 1):
        col = sub.mean(0)
        out.append('<rect x="%d" y="%d" width="%d" height="%d" fill="#%02x%02x%02x"/>'
                   % (xs[x0], ys[y0], xs[x1] - xs[x0], ys[y1] - ys[y0],
                      int(np.clip(col[0], 0, 255)), int(np.clip(col[1], 0, 255)),
                      int(np.clip(col[2], 0, 255))))
        return
    xm, ym = (x0 + x1) // 2, (y0 + y1) // 2
    for a, b, c, d in ((x0, y0, xm, ym), (xm, y0, x1, ym),
                       (x0, ym, xm, y1), (xm, ym, x1, y1)):
        if a < c and b < d:
            _quad_grid(grid, a, b, c, d, xs, ys, tau, out, depth + 1)


def _base_grid_rects(q_arr: np.ndarray, n: int = 48,
                     tau: float = 8.0) -> list[str]:
    """全图自适应兜底层：细网格主色 + 四叉树近色合并，铺在所有 path 之下。

    vtracer ``stacked`` 永远会把某些区域当「透明背景」不输出 path；无论分块还是
    单进程，只靠一张全图单色 <rect> 兜底，这些洞就会露出错误底色（eva 粉斑、
    40bf 白空缺、33df 黑块、HANDOFF §21）。这层只在洞里可见（已覆盖区域被上面
    的 path 完全盖住），却能把洞填成该处本来的颜色。

    规则 48×48 网格会在 4× 放大里留下淡的方形过渡线；按颜色一致性做四叉树
    合并后，平坦区是一整张大 rect、只在真实颜色交界处细分，方块痕迹明显变淡
    （HANDOFF §22/§23：2304 → 约 800 张 rect，RMSE 代价 +0.39）。
    用 <rect> 而非 <path>，edge_smooth 不会把它磨圆。
    """
    H, W = q_arr.shape[:2]
    xs = [i * W // n for i in range(n)] + [W]
    ys = [j * H // n for j in range(n)] + [H]
    grid = np.empty((n, n, 3), np.float32)
    for j in range(n):
        for i in range(n):
            grid[j, i] = _cell_bg_rgb(q_arr[ys[j]:ys[j + 1], xs[i]:xs[i + 1]])
    out: list[str] = []
    _quad_grid(grid, 0, 0, n, n, xs, ys, float(tau), out)
    return out


def _vtracer_tiled(q_arr, params, max_workers=None, overlap=16):
    import multiprocessing as mp
    import os
    H, W = q_arr.shape[:2]
    cw = max(1, min(max_workers or (os.cpu_count() or 4), 16))
    # 按图幅比例取接近方形的网格，竖图多切几行
    nh = max(1, round((cw * H / W) ** 0.5))
    nw = max(1, round(cw / nh))
    nw = max(1, min(nw, 8))
    nh = max(1, min(nh, 8))
    jobs = []
    tile_rects = []
    for j in range(nh):
        y0 = (j * H) // nh
        y1 = H if j == nh - 1 else ((j + 1) * H) // nh
        for i in range(nw):
            x0 = (i * W) // nw
            x1 = W if i == nw - 1 else ((i + 1) * W) // nw
            x0c = max(0, x0 - overlap)
            x1c = min(W, x1 + overlap)
            y0c = max(0, y0 - overlap)
            y1c = min(H, y1 + overlap)
            crop = q_arr[y0c:y1c, x0c:x1c].copy()
            jobs.append((crop, x0c, y0c, params))
            # 每块自带底色：vtracer 把该块主色视为透明背景不输出 path，
            # 没有这层底就会露出全图唯一的 <rect>（挖洞/黑块/白空缺）。
            # 用 <rect> 而非 <path>，edge_smooth 只处理 <path>，不会被磨圆。
            tile_rects.append(
                '<rect x="%d" y="%d" width="%d" height="%d" fill="%s"/>'
                % (x0c, y0c, x1c - x0c, y1c - y0c, _tile_bg_hex(crop)))
    # 由粗到细：全局 rect → 瓦片底 → 细网格底 → 所有 path。
    # 细网格在瓦片底之后绘制，洞里最终显示的是更贴近局部原图的颜色。
    base = tile_rects + _base_grid_rects(q_arr)
    if len(jobs) == 1:
        return "".join(base) + "".join(_vtracer_tile_worker(jobs[0]))
    # 网格划分（jobs）保持原样，只按可用内存限制「同时运行的进程数」：
    # pool.map 依旧按序返回全部结果，输出逐字节不变，画质零影响。
    import mem_budget as _mb
    with mp.Pool(_mb.workers(min(len(jobs), cw))) as pool:
        results = pool.map(_vtracer_tile_worker, jobs)
    out = "".join(base) + "".join(p for r in results for p in r)
    del results
    return out


def _drop_full_canvas_bg(s: str, W: int, H: int):
    """剔除 vtracer 单进程输出里那张「覆盖整幅的底色 path」。

    全图只跑一次 vtracer 时，它会把整图主色当成背景、输出一张 bbox≈全幅
    且极短的矩形 path；这张 path 盖在细网格兜底层之上，于是大片区域露出
    错误底色（HANDOFF §21.3 记的「顶部一大片露出底色」、实测 RMSE 50.94）。
    分块描摹时每块各取自己的主色、且被邻块覆盖，所以不会暴露这个问题。

    去掉它之后，未被 path 覆盖的洞由 ``_base_grid_rects``（逐格精确主色）
    填色，实测 RMSE 50.94 → 12.01。返回 (新 SVG, 剔除数量)。
    """
    out, n, pos = [], 0, 0
    for m in _VTR_PATH_RE.finditer(s):
        out.append(s[pos:m.start()])
        drop = False
        dm = re.search(r'\bd="([^"]*)"', m.group(1))
        if dm and len(dm.group(1)) <= 400:   # 只可能是极短的整幅矩形
            nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", dm.group(1))]
            if len(nums) >= 8:
                xs, ys = nums[0::2], nums[1::2]
                if ((max(xs) - min(xs)) > 0.95 * W and
                        (max(ys) - min(ys)) > 0.95 * H):
                    drop = True
        if drop:
            n += 1
        else:
            out.append(m.group(0))
        pos = m.end()
    out.append(s[pos:])
    return "".join(out), n


def _vtracer_svg(q_arr, params, step, seamless: bool = False):
    """样条描摹入口。

    seamless=True：全图单次描摹。分块多核并行虽快，但每块独立裁切会让
    成千上万条 path 的端点堆在瓦片边界上，形成贯穿画面的「拼接细线」
    （用户 2026-11-12 反馈，2bbf018d 实测显示坐标 x389/786、y374/757/1139
    的二阶差峰正是 3×5 瓦片起点）。全图单次描摹没有瓦片边界，细线消失；
    同输入公平对比 RMSE 12.15→12.01（画质不降反微升），代价是耗时增加。
    """
    if seamless:
        step("无缝描摹：全图单次描摹（消除分块拼接细线，耗时较长）…")
    elif not os.environ.get("IMG2SVG_NO_TILE"):
        try:
            return _vtracer_tiled(q_arr, params)
        except Exception as _e:  # noqa: BLE001
            step("分块描摹失败，回退单进程（%s）" % _e)
    # 单进程全图（seamless 或回退路径，与旧实现逐字节一致）
    import tempfile
    from PIL import Image
    import vtracer
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        Image.fromarray(q_arr).save(tmp.name)
        raw = tmp.name + ".svg"
        _vtrace_png(tmp.name, raw, **params)
        s = io.open(raw, encoding="utf-8", errors="ignore").read()
    finally:
        for _p in (tmp.name, raw):
            try:
                os.remove(_p)
            except OSError:
                pass
    if seamless:
        s, _n = _drop_full_canvas_bg(s, q_arr.shape[1], q_arr.shape[0])
        if _n:
            step("无缝描摹：剔除 %d 张全图底 path，改用逐格兜底色" % _n)
    return "".join(_base_grid_rects(q_arr)) + s


def _bg_color(im: Image.Image) -> str:
    """取四边窄带的众数色当底色（兜底层间缝，不能取错色）。

    旧版只取四角 32x32 块：遇到四角带装饰色标的构图（例如像素画框四角的
    蓝色小角标）会把整张 SVG 底色设成蓝色，四条边于是露出明显蓝线。改成
    四条边整条参与投票后，边框主体色（面积远大于角标）稳定胜出。
    """
    a = np.asarray(im.convert("RGB"))
    h, w = a.shape[:2]
    k = max(2, min(24, min(h, w) // 24))
    strips = np.concatenate([
        a[:k, :].reshape(-1, 3), a[-k:, :].reshape(-1, 3),
        a[:, :k].reshape(-1, 3), a[:, -k:].reshape(-1, 3),
    ], axis=0)
    q = (strips >> 4).astype(np.int32)
    key = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    v = int(vals[counts.argmax()])
    return "#%02x%02x%02x" % (((v >> 8) & 15) * 16 + 8,
                              ((v >> 4) & 15) * 16 + 8, (v & 15) * 16 + 8)


# 三档「色块干净度」：layer_difference 控制相近色合并成多大一片，
# filter_speckle 控制丢弃多小的碎块。qq3d 参考图约 3000 个大块，就是 clean 档。
# 实测（荷叶精灵图 2059×1197，256 色）：
#   detail   ld=6  sp=4 → 25117 块 / ΔE 4.02（碎斑多、观感斑驳）
#   balanced ld=18 sp=6 → 10798 块 / ΔE 4.12
#   clean    ld=24 sp=8 →  6000 块 / ΔE 4.63（≈ qq3d 4.61，皮肤干净、水珠发丝仍在）
PROFILES = {
    "detail":   dict(layer_difference=6,  speckle=4),
    "balanced": dict(layer_difference=18, speckle=6),
    "clean":    dict(layer_difference=24, speckle=8),
}

# 边缘顺滑档：vtracer 输出的是逐段追随像素阶梯的短三次贝塞尔，放大后呈锯齿。
# 旧实现（一次弱移动平均 + 大 eps RDP）只磨掉毛刺、保留原始锯齿顶点，放大后
# 仍是逐像素短折线 → 观感毛糙。新实现（见 edge_smooth.smooth_svg）改为
# 「多轮移动平均把阶梯松弛成光滑中心线 → 尖角钉回 → 中等 eps RDP 安全压缩」，
# 既流畅又保形。实测（荷叶图 2059×1197，clean 档 + 2× 保真高清化）：
#   off      ΔE 4.23（锯齿最明显）
#   soft     iters2/win5/eps1.5  ΔE 4.90（轻度顺滑，形状几乎不动）
#   normal   iters3/win5/eps2.5  ΔE 5.17（流畅且形状/细线宽度保住，推荐）★ 8× 观感最佳平衡
#   strong   iters3/win5/eps3.5  ΔE 5.27（最流畅，细线略变细，要极顺滑时用）
# 8× 放大对照 qq3d 参考：normal/strong 已接近 qq3d 的流畅长直线，而旧 normal 仍
# 呈阶梯/抖动；heavy(old win9/eps4) 虽顺但 ΔE 6.15 且线条变粗，已弃用。
EDGE_SMOOTH = {
    "off":    None,
    "soft":   dict(smooth_win=5, eps=1.5, corner_deg=120.0, bend_min_deg=18.0, iters=2),
    "normal": dict(smooth_win=5, eps=2.5, corner_deg=120.0, bend_min_deg=18.0, iters=3),
    "strong": dict(smooth_win=5, eps=3.5, corner_deg=120.0, bend_min_deg=18.0, iters=3),
}


def _cap_print_palette(labels, pal, limit=80, protect=()):
    """打印模式：把最终出现的颜色数压到 ``limit`` 以内（色块数 = 换料次数）。

    只在入口限制量化色数是不够的：墨线统一 / 模型线条层之后还会往色板追加
    颜色。这里按使用面积淘汰最小的色块，合并到 OKLab 最近的保留色；
    ``protect`` 中的索引（线条层）一律保留，否则刚补上的线条会被合并掉。
    """
    flat = np.maximum(labels.ravel().astype(np.int64), 0)
    used = np.bincount(flat, minlength=len(pal))
    used_idx = [int(i) for i in np.nonzero(used)[0]]
    if len(used_idx) <= limit:
        return labels, 0
    protect = {int(p) for p in protect}
    keep_line = [i for i in used_idx if i in protect]
    rest = [i for i in used_idx if i not in protect]
    rest.sort(key=lambda i: used[i], reverse=True)
    keep = set(keep_line + rest[:max(0, limit - len(keep_line))])
    drop = [i for i in used_idx if i not in keep]
    if not drop:
        return labels, 0
    keep_list = sorted(keep)
    k_lab = _srgb_to_oklab(pal[keep_list].astype(np.float32) / 255.0)
    d_lab = _srgb_to_oklab(pal[drop].astype(np.float32) / 255.0)
    out = labels.copy()
    for j, src in enumerate(drop):
        tgt = keep_list[int(np.argmin(((k_lab - d_lab[j]) ** 2).sum(1)))]
        out[labels == src] = tgt
    return out, len(drop)


def trace(src: str, dst: str | None = None, *, colors: int = 256,
          denoise_strength: float = 1.0, defringe_strength: float = 1.0,
          seam_width: float = 1.2,
          seam_alpha: float = 1.0, speckle: int | None = None,
          layer_difference: int | None = None, profile: str = "clean",
          edge_smooth: str = "normal",
          print_mm: float = 0.0, line_mm: float = 0.42, self_line: bool = False,
          scale: float = 1.0,
          display_size: tuple[int, int] | None = None,
          max_side: int = 0, seed: int = 7, seamless: bool = False,
          progress=None) -> dict:
    """描摹图片，返回 {"svg", "paths", "colors", "width", "height", "ms"}。

    profile 取 detail / balanced / clean；显式传 speckle 或 layer_difference
    时覆盖对应档位的值。edge_smooth 取 off / soft / normal / strong。

    scale: 输入图相对显示尺寸的放大倍数（保真高清化后一般为 2）。像素类参数
    （filter_speckle / length_threshold / 顺滑窗口与容差）按倍率同步放大，
    否则高分辨率输入会被当成噪点过度清理、曲线也相对更碎。

    display_size: 若给出，则 SVG 的 width/height 用它，而 viewBox 仍保留高清
    坐标 —— 显示尺寸与原图一致，内部路径精度翻倍（与 qq3d 的 scale(2.02)
    同款做法）。
    """
    preset = PROFILES.get(profile, PROFILES["clean"])
    sc = max(1.0, float(scale))
    sp = int(round((speckle if speckle is not None else preset["speckle"]) * sc))
    ld = int(layer_difference if layer_difference is not None else preset["layer_difference"])
    # 细节保留力度（HANDOFF §19）：vtracer 的 filter_speckle 会把窄细线 / 高光
    # 当成噪点整块丢弃，length_threshold 过长则会把细线拟合成粗曲线。旧值
    # (speckle=6*scale≈9, length=3.0*scale≈4.5) 实测把眼睛高光、发丝、鞋带排线
    # 全部抹平，白色渐变还被切成斜向色带。四类样张（177 人物 / 362cb 插画 /
    # 繁复纹理 / anime074）A/B 全部双赢：原尺寸 RMSE 降 10–18%，4× 放大细节
    # 明显更清晰，无碎斑爆炸。因此默认全局启用，无需任何用户参数或图像判据
    # （判据曾试过 classify._thin_frac_native，177 只有 0.042 会漏判，已弃用）。
    # IMG2SVG_FINE_DETAIL=0 可强制回退到旧参数，用于 A/B 与紧急回退。
    if os.environ.get("IMG2SVG_FINE_DETAIL", "1") != "0":
        sp = max(2, int(round(sp * 0.45)))
        _lt_scale = 1.34
    else:
        _lt_scale = 3.0
    t0 = time.time()

    def step(msg):
        if progress:
            progress(msg)

    import vtracer

    step("读取图片…")
    im = Image.open(src).convert("RGB")
    if max_side and max(im.size) > max_side:
        r = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))), Image.LANCZOS)
    w, h = im.size

    if denoise_strength > 0:
        step("边缘保持降噪（去杂色块）…")
        im = denoise(im, denoise_strength)
    # IMG2SVG_DEFRINGE=0 可一键关闭做 A/B（默认开，不需要用户手动选）
    _df = float(os.environ.get("IMG2SVG_DEFRINGE", defringe_strength))
    if _df > 0:
        step("边缘色度去边（压青紫彩边 / 亮度不变）…")
        im = defringe(im, _df)

    step("感知量化 %d 级（保留渐变 / 光泽 / 柔影）…" % int(colors))
    q, labels, pal = quantize_perceptual(im, int(colors), seed=seed, return_labels=True)

    # 区域规整：标签平滑去杂斑/磨圆阶梯，再在真实图上把细暗线统一成实心墨线层。
    # 平滑会侵蚀 1-2px 暗线，墨线统一负责补回，二者配合才既干净又线条清晰。
    step("区域规整（标签平滑去杂斑 + 墨线统一保细线）…")
    im_arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    # 源头抹阶梯：加大逐色高斯投票的 sigma 把量化像素阶梯磨圆，同时用**原图**
    # 梯度保护真实轮廓 / 发丝 / 花纹，避免无差别软化糊掉细节。所有 path 仍从
    # 同一张标签图生成，边界天然共享，不会像「成品上做曲线后处理」那样露底色
    # （HANDOFF §22/§23）。档位 0 = 旧行为（0.4*sc、无保护），逐字节不变。
    import label_regularize as _lr
    _lreg = _lr.from_env()
    # 档位表：sigma 倍数（相对旧版 0.4*sc）与「真实边缘」保护分位。
    # 0 = 旧行为；1–3 温和；4–5 激进；6 = 不保护（仅作上界参考，会糊细节）。
    _sigma_mult = (1.0, 1.5, 2.0, 2.5, 4.0, 6.0, 8.0)[_lreg]
    _pct = (0.0, 70.0, 70.0, 65.0, 80.0, 90.0, 0.0)[_lreg]
    _sigma = 0.4 * sc * _sigma_mult
    _protect = None
    if 0 < _lreg <= 5:
        _protect = _lr.edge_protect(im_arr, _pct)
        step("标签图规整（档 %d：sigma=%.2f，真实边缘保护 %.1f%%）…"
             % (_lreg, _sigma, 100.0 * float(_protect.mean())))
    labels = _label_smooth(labels, len(pal), _sigma, protect=_protect)
    # 注意：不要再在这里加「色板层去 1px 过渡色」。见 HANDOFF §13 —— 该实验
    # 在 1× 原型上有效，但产品先做 1.5–2× 保真高清化，过渡带被放大成 2–3px，
    # 判据失效；且 v6.7 的 defringe 已把色度尖峰压到 0.39%（原图 2.63%），
    # 放大图里残留的淡紫线是低频色偏，不是同一类问题。加了只会让 RMSE 变差。
    from own_tracer import unify_ink_cc
    # Colours appended on top of the quantised palette (ink lines / model line
    # layer).  Print mode must NOT merge these away -- they are the structure.
    extra_ids: set[int] = set()
    # 高细节插画（青海波 / 网点 / 密集发丝）会被黑帽误判出几百条"墨线"，
    # 全保留会把背景刷成深色网状碎块。只留对比度最高的 80 条真描边。
    labels, created = unify_ink_cc(
        im_arr, labels, threshold=28.0, close=max(3, int(2 * sc) + 1),
        dark_luma=140.0, width_thresh=3.0 * sc, dark_margin=50.0,
        max_new_ratio=0.18, max_lines=80)
    if created:
        # 每个墨线连通域一个颜色会让细线密集的插画色数爆炸（实测 447 色）；
        # 聚成 8 档统一墨线色，保留深浅层次的同时压住换料次数。
        flat = labels.ravel()
        px = im_arr.reshape(-1, 3).astype(np.float64)
        cnt = np.bincount(flat, minlength=len(created) + int(created[0]))
        sums = np.stack([np.bincount(flat, weights=px[:, ch],
                                     minlength=len(cnt)) for ch in range(3)], 1)
        base = int(created[0])
        extra = (sums[base:base + len(created)] /
                 np.maximum(cnt[base:base + len(created), None], 1.0))
        pal = np.concatenate([pal, extra.round().astype(np.uint8)])
        labels, pal, extra_ids = _consolidate_ink(labels, pal, created, k=8)
        step("墨线层择优保留 %d 条 / 归并为 %d 档（避免纹理误判与色数爆炸）"
             % (len(created), len(extra_ids)))
    # --- 自研线条层（可选）：形态学黑帽找暗细结构，补一层顶部墨线 ---
    if self_line:
        try:
            import line_layer as _ll
            # v7.5.6: 完全自研形态学描线，不再加载任何外部模型。
            # 原 PiDiNet ONNX 权重仅授权研究用途，商业分发不可随附。
            _m = _ll.stroke_mask(im_arr)
            _src = "自研形态学"
            if _m.any() and float(_m.mean()) <= 0.12:
                _id = int(labels.max()) + 1
                labels = labels.copy()
                labels[_m > 0] = _id
                _col = im_arr[_m > 0].mean(0).round().astype(np.uint8)
                pal = np.concatenate([pal, _col[None, :]], axis=0)
                extra_ids.add(int(_id))
                step("自研线条层（%s）：补 %d px（%.2f%%）" % (_src, int(_m.sum()), 100 * float(_m.mean())))
            else:
                step("自研线条层跳过（覆盖 %.1f%% 超出合理范围）" % (100 * float(_m.mean()) if _m.any() else 0))
        except Exception as _e:  # noqa: BLE001
            step("自研线条层跳过（%s）" % _e)

    # --- 打印模式：比一个挤出珠还窄的色带，按"是不是一条线"决定合并还是加粗 ---
    if print_mm and print_mm > 0:
        try:
            import print_mode as _pm
            _bp = _pm.bead_px(max(im.size), float(print_mm), float(line_mm))
            if _bp > 1.2:
                step("打印瘦身（%.0fmm / 线宽 %.2fmm → 珠宽 %.2fpx）…"
                     % (print_mm, line_mm, _bp))
                labels, pal, _st = _pm.apply_print_mode(labels, pal, _bp)
                step("打印瘦身：合并 %d 条窄带，加粗 %d 条线"
                     % (_st.get("merged", 0), _st.get("thickened", 0)))
        except Exception as _exc:  # noqa: BLE001
            step("打印瘦身跳过（%s）" % _exc)
        # Hard cap on the FINAL colour count.  Capping the quantisation palette
        # alone is not enough: ink-line / model-line layers append colours after
        # it, which pushed a 80-colour palette to 93 in the acceptance test.
        labels, _dropped = _cap_print_palette(labels, pal, 80, extra_ids)
        if _dropped:
            step("打印模式：再合并 %d 个小色块，最终色数 ≤ 80" % _dropped)

    q = Image.fromarray(pal[labels], "RGB")
    # 内存：区域规整已结束，原图数组与标签图（各几十 MB）不再使用，
    # 立即释放再启动多进程描摹，避免与 worker 常驻内存叠加。
    import gc
    del im_arr, labels
    gc.collect()

    import mem_budget as _mb
    step(_mb.describe())
    step("样条描摹（%s 档：layer_difference=%d, speckle=%d，%s）…"
         % (profile, ld, sp,
            "无缝全图单次" if seamless else "分块多核并行（可用 seamless 消除拼接线）"))
    # hierarchical: stacked = 层叠覆盖（默认，画质基线）；cutout = 共边镶嵌。
    # 后者让洞区更小、色块更少，但会改变颜色分层（见 HANDOFF §22）；
    # 用环境变量做 A/B 与回退，默认行为逐字不变。
    _hier = os.environ.get("IMG2SVG_HIERARCHICAL", "stacked").strip().lower()
    if _hier not in ("stacked", "cutout"):
        _hier = "stacked"
    params = dict(clustering="color", hierarchical=_hier, mode="spline",
                 filter_speckle=int(max(1, min(32, sp))), color_precision=8,
                 layer_difference=ld, corner_threshold=65, length_threshold=_lt_scale * sc,
                 max_iterations=10, splice_threshold=45, path_precision=2)
    qa = np.asarray(q)
    _hb = _start_heartbeat(step, "样条描摹")
    try:
        raw_svg = _vtracer_svg(qa, params, step, seamless=bool(seamless))
    finally:
        _hb()
    del qa
    gc.collect()

    step("颜色吸附回色板" + ("+ qq3d 式补缝" if seam_width > 0 else "（stacked 天然无重叠缝，跳过补缝）") + "…")
    uniq = np.unique(np.asarray(q).reshape(-1, 3), axis=0)
    palette = ["#%02x%02x%02x" % (int(c[0]), int(c[1]), int(c[2])) for c in uniq]
    svg, npaths, ncolors = build_svg(raw_svg, w, h, _bg_color(q),
                                     float(seam_width), float(seam_alpha),
                                     palette=palette)
    # vtracer 原始 SVG 文本（几十 MB）与量化图在组装后不再需要；
    # 顺滑重拟合还要再持有一份成品 SVG，先释放可显著压低峰值。
    del raw_svg, q
    gc.collect()
    opts = EDGE_SMOOTH.get(edge_smooth)
    if opts:
        step("边缘顺滑重拟合（%s 档）…" % edge_smooth)
        _hb = _start_heartbeat(step, "边缘顺滑重拟合")
        try:
            import edge_smooth as _es
            # 实测（荷叶图 2× 高清化输入）：eps 是点数简化容差、win 是采样点
            # 窗口，都不是物理距离。高清化后 length_threshold 已按倍率放大、
            # 每段曲线更长、采样点间距同步变大，因此这里**不能再按 scale
            # 放大**，否则 win=10/eps=5 会把细线削细、色块改形（ΔE 4.23→6.00）。
            # 保持档位原值时 eps2.5/win5 正好落在最佳区间（ΔE≈4.8）。
            svg = _es.smooth_svg(svg, **dict(opts))
        except Exception as exc:  # noqa: BLE001
            step("边缘顺滑跳过（%s）" % exc)
        finally:
            _hb()
    if display_size and tuple(display_size) != (w, h):
        # 内部路径保持高清坐标，显示尺寸回到原图（qq3d 的 scale(2.02) 同款）
        _old = 'width="%d" height="%d" viewBox="0 0 %d %d"' % (w, h, w, h)
        _new = 'width="%d" height="%d" viewBox="0 0 %d %d"' % (
            int(display_size[0]), int(display_size[1]), w, h)
        svg = svg.replace(_old, _new, 1)
    if dst:
        io.open(dst, "w", encoding="utf-8", newline="\n").write(svg)

    return {
        "svg": svg, "paths": npaths, "colors": ncolors,
        "width": w, "height": h,
        "bytes": os.path.getsize(dst) if dst else len(svg.encode("utf-8")),
        "ms": int((time.time() - t0) * 1000),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        info = trace(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
                     progress=lambda m: print("  ", m, flush=True))
        print({k: v for k, v in info.items() if k != "svg"})
