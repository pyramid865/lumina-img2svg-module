# -*- coding: utf-8 -*-
"""渐变覆盖（v7.17.15）—— 用线性渐变根治「量化色阶硬边」。

问题：一张天空/水面这类平滑渐变，被量化成 N 种颜色后，颜色不是连续变化，
而是一圈层一层层的「台阶」。台阶之间的色差只有 1~3 ΔE（人眼刚好能看出边），
但边界是一条硬线，放大就是一圈圈等高线。补缝、加色数都治不了——只要还是
「一块一块涂色」，台阶就永远存在。

做法：在量化图上找「大面积平滑、内部色差小、颜色沿一个方向单调变化」的区域，
用一条线性渐变（linearGradient）直接覆盖它。颜色在两个端点之间连续插值，
台阶自然消失。原来的色块 path 保留在下面（被盖住），不再逐个删除——
删错一条就是画面破洞，风险远大于收益。

全部自研：numpy + OpenCV（项目已有依赖），不引入任何新库。
"""
from __future__ import annotations

import re

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# ── 参数（每条的「调大/调小」影响都写在下面） ────────────────────────
#   GRAD_FILL      总开关。默认**关**。原因：实测覆盖不可靠——区域轮廓是
#                  多边形近似，与相邻色块错位 1~2 px；渐变方向在不规则区域
#                  （云雾、湖岸）也难对准。结果墙面 p95 从 2.3 涨到 6.0、
#                  p99 从 3.4 涨到 10.6。留作实验，不参与默认管线。
#   GRAD_MIN_FRAC  区域门槛：面积至少占全图多少才处理。调大（0.05）＝只处理
#                  大面积天空/水面，改动最少；调小（0.005）＝小块渐变也覆盖，
#                  但误判成渐变的概率上升（可能把该保留的色块抹平）。
#   GRAD_LINK_DE   成链门槛：相邻两档颜色差多少以内算「同一段渐变」。
#                  调大（6）＝更多色档被连成长链，覆盖更全，但可能把两个
#                  不同物体的平整面串成一段；调小（2）＝链变碎，只收最平滑
#                  的那几档，安全但覆盖少。
#   GRAD_MIN_CHAIN 一条链至少几档颜色才算渐变。调大＝更严格（只收明显渐变）；
#                  调小＝两三个色块的过渡也会被收，误判率上升。
#   GRAD_FIT_TOL   拟合残差上限（ΔE）：颜色沿方向偏离直线的程度。调大＝更宽
#                  容（径向渐变也会被硬套成线性，方向可能不对）；调小＝只收
#                  真正笔直的渐变，覆盖率下降。
#   GRAD_MAX_ZONES 单张图最多覆盖几个区域。调大＝覆盖更全、耗时增加。
GRAD_FILL = False
#   GRAD_FLAT_DE   刷平上限：色链两端色差小于它时，整条链刷成一个中位色。
#                  调大（8）＝更多浅色阶被刷平，条纹消失但可能出现「一片色块」；
#                  调小（2）＝只刷真正均匀的区域，安全但对 2~3 ΔE 的台阶无效。
#                  设 0＝禁用刷平。
#   GRAD_ALLOW_LINEAR  是否允许用线性渐变覆盖（v7.17.15 实测失败，默认关）。
GRAD_FLAT_DE = 6.0
GRAD_ALLOW_LINEAR = False
GRAD_MIN_FRAC = 0.012
GRAD_LINK_DE = 4.0
GRAD_MIN_CHAIN = 4
GRAD_FIT_TOL = 3.0
GRAD_MAX_ZONES = 6

_MIN_LABEL_PX = 60      # 单个色块至少这么多像素才参与成链（去噪点）

_FLAT_WIN = 7        # 平滑判定窗口（像素）。调大＝更保守，只认大片平坦区
_FLAT_STD = 3.0      # 窗口内标准差低于它算平滑。调大＝更多区域入选（含纹理）
_DILATE = 2          # 覆盖轮廓外扩像素。调大＝盖得更死但更容易溢出到邻居


def _lab(rgb: np.ndarray) -> np.ndarray:
    import color as _c
    return _c.srgb_to_oklab(np.asarray(rgb, np.float32) / 255.0)


def _des(a, b) -> float:
    """两个 OKLab 颜色的色差（ΔE×100，人眼刚好能分辨约 1）。"""
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32))) * 100.0


def _de(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1) * 100.0


def _median_color(px: np.ndarray) -> np.ndarray:
    return np.median(px, axis=0)


def _fit_zone(rgb: np.ndarray, mask: np.ndarray):
    """在一个区域上拟合线性渐变。返回 (x1,y1,x2,y2,c0,c1,residual) 或 None。"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mean = pts.mean(0)
    cov = np.cov((pts - mean).T)
    if not np.isfinite(cov).all():
        return None
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    if not np.isfinite(axis).all() or float(np.hypot(*axis)) < 1e-6:
        return None
    axis = axis / float(np.hypot(*axis))
    t = (pts - mean) @ axis

    lo, hi = np.percentile(t, [2, 98])
    if hi - lo < 8:
        return None
    p0 = mean + axis * lo
    p1 = mean + axis * hi

    lab = _lab(rgb)
    t_all = ((np.stack([xs, ys], 1) - mean) @ axis)
    span = hi - lo
    edge = max(2.0, span * 0.04)
    m0 = np.abs(t_all - lo) <= edge
    m1 = np.abs(t_all - hi) <= edge
    c0 = _median_color(lab[ys[m0], xs[m0]])
    c1 = _median_color(lab[ys[m1], xs[m1]])
    flat_only = (_des(c0, c1) < GRAD_FLAT_DE) if GRAD_FLAT_DE > 0 else False

    # 线性拟合残差：把区域按主轴分成 9 段，每段取中位色，和线性插值比
    q = np.clip((t_all - lo) / span, 0.0, 1.0)
    bins = np.minimum((q * 8).astype(int), 8)
    errs = []
    for b in range(9):
        sel = bins == b
        if sel.sum() < 30:
            continue
        cb = _median_color(lab[ys[sel], xs[sel]])
        lin = c0 + (c1 - c0) * ((b + 0.5) / 9.0)
        errs.append(float(_de(cb, lin)))
    resid = float(np.percentile(errs, 90)) if errs else 999.0
    if flat_only:
        return (0.0, 0.0, 0.0, 0.0, c0, c1, 0.0, "flat")
    if not GRAD_ALLOW_LINEAR:
        return None
    if resid > GRAD_FIT_TOL:
        return None

    return (float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]),
            c0, c1, resid, "grad")


def _oklab_to_hex(c) -> str:
    import color as _c
    lab = np.asarray(c, np.float32).reshape(1, 1, 3)
    rgb = _c.oklab_to_rgb01(lab).reshape(3)
    rgb = np.clip(np.round(rgb * 255.0), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(rgb)


def _zone_path(mask: np.ndarray, eps: float = 2.0):
    """区域掩码 → 简化多边形 path（只保留外轮廓）。"""
    m = mask.astype(np.uint8)
    if _DILATE:
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=_DILATE)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return ""
    ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(ap) < 3:
        return ""
    return "M" + "L".join("%d %d" % (int(p[0]), int(p[1])) for p in ap) + "Z"


def build(rgb: np.ndarray, index: np.ndarray, pal: np.ndarray):
    """返回 (defs片段, path片段, 覆盖区域数)。rgb/index 为工作网格尺寸。"""
    if not GRAD_FILL or cv2 is None:
        return "", "", 0
    h, w = index.shape
    total = float(h * w)

    g = np.asarray(rgb, np.float32)
    gray = g.mean(2)
    mu = cv2.blur(gray, (_FLAT_WIN, _FLAT_WIN))
    mu2 = cv2.blur(gray * gray, (_FLAT_WIN, _FLAT_WIN))
    std = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    flat = std < _FLAT_STD
    flat = cv2.morphologyEx(flat.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8)).astype(bool)

    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(flat.astype(np.uint8), 8)
    pal_lab = _lab(pal)
    defs, paths, used = [], [], 0
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1

    for ci in order:
        if used >= GRAD_MAX_ZONES:
            break
        area = int(stats[ci, cv2.CC_STAT_AREA])
        if area < GRAD_MIN_FRAC * total:
            continue
        region = lab_cc == ci
        # 紧凑度：区域面积 / 外接矩形面积。太低说明是环状/带状，覆盖会吃掉洞
        x, y, bw, bh = (int(stats[ci, k]) for k in range(4))
        if area / float(max(1, bw * bh)) < 0.45:
            continue
        idxs = np.unique(index[region])
        counts = {int(i): int((index[region] == i).sum()) for i in idxs}
        keep = [i for i in idxs if counts[int(i)] >= _MIN_LABEL_PX]
        if len(keep) < GRAD_MIN_CHAIN:
            continue
        # 按亮度排序成「色链」：相邻两档色差 ≤ GRAD_LINK_DE 就连起来。
        # 一条链就是一段渐变（天空的 20 档淡蓝、墙面的 15 档米色）；
        # 链与链之间色差大，说明是不同物体，不合并。
        ordv = [keep[int(i)] for i in np.argsort(pal_lab[np.asarray(keep), 0])]
        chains, cur = [], [ordv[0]]
        for i in ordv[1:]:
            if _des(pal_lab[i], pal_lab[cur[-1]]) <= GRAD_LINK_DE:
                cur.append(int(i))
            else:
                chains.append(cur)
                cur = [int(i)]
        chains.append(cur)

        for ch in chains:
            if used >= GRAD_MAX_ZONES:
                break
            if len(ch) < GRAD_MIN_CHAIN:
                continue
            px = sum(counts[i] for i in ch)
            if px < GRAD_MIN_FRAC * total:
                continue
            zone = region & np.isin(index, np.asarray(ch, np.int32))
            fit = _fit_zone(g, zone)
            if fit is None:
                continue
            d = _zone_path(zone)
            if not d:
                continue
            x1, y1, x2, y2, c0, c1, _res, kind = fit
            if kind == "flat":
                # 近均匀区（两端色差 < 2 ΔE，人眼分不出）：直接刷成中位色。
                # 这一片的所谓「色带」本来就是 1~2 ΔE 的量化噪声，刷平只会更干净。
                paths.append('<path fill="%s" d="%s"/>' % (_oklab_to_hex(c0), d))
            else:
                gid = "gf%d" % used
                defs.append(
                    '<linearGradient id="%s" gradientUnits="userSpaceOnUse" '
                    'x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f">'
                    '<stop offset="0" stop-color="%s"/><stop offset="1" stop-color="%s"/>'
                    '</linearGradient>' % (gid, x1, y1, x2, y2,
                                           _oklab_to_hex(c0), _oklab_to_hex(c1)))
                paths.append('<path fill="url(#%s)" d="%s"/>' % (gid, d))
            used += 1

    if not paths:
        return "", "", 0
    return "<defs>%s</defs>" % "".join(defs), "".join(paths), used


def apply_svg(svg: str, rgb: np.ndarray, index: np.ndarray,
              pal: np.ndarray) -> tuple:
    """把渐变覆盖层插到所有色块之后、</svg> 之前（后画＝盖在上面）。"""
    defs, paths, n = build(rgb, index, pal)
    if not n:
        return svg, 0
    if "</defs>" in svg:
        svg = svg.replace("</defs>", defs.replace("<defs>", "").replace("</defs>", "") + "</defs>", 1)
    elif "<defs" not in svg:
        svg = re.sub(r"(<svg\b[^>]*>)", r"\1" + defs, svg, count=1)
    body = "<g>%s</g>\n</svg>" % paths
    if svg.count("</svg>") != 1:
        return svg, 0
    return svg.replace("</svg>", body, 1), n
