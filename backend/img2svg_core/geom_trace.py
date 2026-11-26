# -*- coding: utf-8 -*-
"""几何化重绘引擎 (MODE_GEOM) —— 对标「嘻嘻宝」的长圆弧大色块。

与 vtracer/own_tracer 的根本区别：那些引擎让短三次贝塞尔逐段追随整数像素
阶梯（放大后必留毛刺）；本引擎先把图规整成少量大色块，再把每条边界**当成
几何图形重绘** —— 角点处切开，角点之间的平滑段优先用一条直线 L、一段圆弧
A、或一条二次贝塞尔 Q 表达，实在复杂才递归细分或用三次 C 兜底。

流程：
    OKLab 感知量化(默认 96 色) → 标签平滑 → 小/窄色带并入最近邻
    → 逐色亚像素等值线(marching squares) → 角点检测 + 基元拟合
    → 大色块 SVG(高精度浮点坐标，按面积降序叠放)

只依赖 numpy / PIL / cv2 / scipy / skimage（打包环境已全部具备）。
详见 HANDOFF.md §10。
"""
from __future__ import annotations

import io
import math
import os
import time

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# 数字与折线工具
# --------------------------------------------------------------------------- #
def _f(v) -> str:
    """高精度浮点坐标（嘻嘻宝是 13–16 位，这里 15 位有效数字）。"""
    x = float(v)
    if abs(x) < 1e-12:
        return "0"
    return "%.15g" % x


def _pt(p) -> str:
    return "%s %s" % (_f(p[0]), _f(p[1]))


def _resample_ring(p: np.ndarray, step: float = 1.0) -> np.ndarray:
    """闭合环按弧长等距重采样。"""
    q = np.vstack([p, p[:1]])
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
    if d[-1] < step * 3:
        return p
    s = np.arange(0.0, d[-1], step)
    return np.c_[np.interp(s, d, q[:, 0]), np.interp(s, d, q[:, 1])]


def _corner_indices(p: np.ndarray, angle_deg: float = 60.0, k: int = 6) -> list[int]:
    """真实转角（不是锯齿）：前后 k 点夹角小于 180-angle_deg 的局部极小。"""
    n = len(p)
    if n < 2 * k + 3:
        return []
    a = p[(np.arange(n) - k) % n] - p
    b = p[(np.arange(n) + k) % n] - p
    cos = (a * b).sum(1) / np.maximum(np.hypot(*a.T) * np.hypot(*b.T), 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    sharp = ang < (180.0 - angle_deg)
    out: list[int] = []
    for i in np.flatnonzero(sharp):
        if ang[i] <= ang[(i - 1) % n] and ang[i] <= ang[(i + 1) % n]:
            if not out or i - out[-1] > k:
                out.append(int(i))
    return out


def _line_error(P: np.ndarray) -> tuple[float, int]:
    """点到首尾连线的最大垂距 + 最远点索引。"""
    if len(P) < 3:
        return 0.0, len(P) // 2
    a, b = P[0], P[-1]
    v = b - a
    L2 = float(v @ v)
    if L2 < 1e-9:
        d = np.hypot(*(P - a).T)
        return float(d.max()), int(d.argmax())
    t = np.clip(((P - a) @ v) / L2, 0.0, 1.0)
    d = np.hypot(*(P - (a + t[:, None] * v)).T)
    return float(d.max()), int(d.argmax())


def _arc_len_t(P: np.ndarray) -> np.ndarray:
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    return d / max(1e-9, d[-1])


# --------------------------------------------------------------------------- #
# 几何基元拟合
# --------------------------------------------------------------------------- #
def _circle3(p0, pm, p1):
    """三点定圆 -> (cx, cy, r) 或 None。"""
    ax, ay = p0
    bx, by = pm
    cx, cy = p1
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    r = math.hypot(ux - ax, uy - ay)
    if not (0 < r < 1e6):
        return None
    return ux, uy, r


def _try_arc(P: np.ndarray, tol: float) -> str | None:
    """若整段落在同一个圆上，返回一条 A 命令（SVG 椭圆弧，rx=ry=r）。

    用「生成候选圆弧采样点 + 与原点列比较」来验证 sweep/large-arc，避免
    符号推导出错导致圆弧跑到轮廓另一侧。
    """
    if len(P) < 5:
        return None
    p0, pm, p1 = P[0], P[len(P) // 2], P[-1]
    cir = _circle3(p0, pm, p1)
    if cir is None:
        return None
    cx, cy, r = cir
    C = np.array([cx, cy])
    dev = np.abs(np.hypot(*(P - C).T) - r)
    if float(dev.max()) > tol:
        return None
    chord = float(np.linalg.norm(p1 - p0))
    # 近似直线的段三点定圆会得到巨大半径；那种段应该交给 Q/L，不要输出
    # 一个半径几万像素的假圆弧。
    if r > 12.0 * max(chord, 1.0) or 2 * math.asin(min(1.0, chord / (2 * r))) < math.radians(4):
        return None
    a0 = math.atan2(p0[1] - cy, p0[0] - cx)
    a1 = math.atan2(p1[1] - cy, p1[0] - cx)
    small = (a1 - a0) % (2 * math.pi)
    if small > math.pi:
        small = 2 * math.pi - small
    # four (sweep, large) candidates; pick the one whose sampled arc matches P
    t = _arc_len_t(P)
    best, best_err = None, 1e18
    for sweep in (0, 1):
        for large in (0, 1):
            if sweep == 1:
                delta = small if not large else 2 * math.pi - small
            else:
                delta = -(small if not large else 2 * math.pi - small)
            # large=1 needs the complementary angle, guard against degenerate
            if abs(delta) < 1e-6 or abs(delta) > 2 * math.pi - 1e-6:
                continue
            ang = a0 + t * delta
            pts = C + r * np.c_[np.cos(ang), np.sin(ang)]
            err = float(np.hypot(*(pts - P).T).max())
            if err < best_err:
                best_err, best = err, (sweep, large)
    if best is None or best_err > max(tol, 0.8):
        return None
    sweep, large = best
    return "A %s %s 0 %d %d %s" % (_f(r), _f(r), large, sweep, _pt(p1))


def _try_quad(P: np.ndarray, tol: float) -> str | None:
    """最小二乘单控制点二次贝塞尔（端点固定）。"""
    if len(P) < 4:
        return None
    t = _arc_len_t(P)
    p0, p1 = P[0], P[-1]
    B = 2.0 * (1.0 - t) * t
    denom = float((B * B).sum())
    if denom < 1e-9:
        return None
    rhs = P - ((1.0 - t) ** 2)[:, None] * p0 - (t ** 2)[:, None] * p1
    ctrl = (B[:, None] * rhs).sum(0) / denom
    fit = (((1.0 - t) ** 2)[:, None] * p0 + B[:, None] * ctrl
           + (t ** 2)[:, None] * p1)
    err = float(np.hypot(*(P - fit).T).max())
    if err > tol:
        return None
    return "Q %s %s" % (_pt(ctrl), _pt(p1))


def _fit_cubic(P: np.ndarray) -> str:
    """最小二乘三次贝塞尔兜底（端点固定，解两个控制点）。"""
    t = _arc_len_t(P)
    p0, p1 = P[0], P[-1]
    u = 1.0 - t
    B1, B2 = 3 * u * u * t, 3 * u * t * t
    rhs = P - (u ** 3)[:, None] * p0 - (t ** 3)[:, None] * p1
    A = np.stack([B1, B2], 1)
    try:
        ctrl = np.linalg.lstsq(A, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        d = (p1 - p0) / 3.0
        ctrl = np.array([p0 + d, p1 - d])
    return "C %s %s %s" % (_pt(ctrl[0]), _pt(ctrl[1]), _pt(p1))


def _fit_polyline(P: np.ndarray, tol: float, depth: int = 0,
                  max_depth: int = 7) -> list[str]:
    """把一段折线递归拟合成最少的 L / A / Q / C 命令。"""
    if len(P) < 2:
        return []
    err, split = _line_error(P)
    # 优先「画成曲线」再退回直线：嘻嘻宝的 Q/A 远多于 L。若先判直线，
    # 平滑不足时每个像素阶梯都会变成一条短 L，等于忠实描摹锯齿。
    if depth < max_depth:
        arc = _try_arc(P, tol)
        if arc is not None:
            return [arc]
        quad = _try_quad(P, tol)
        if quad is not None:
            return [quad]
    if err <= tol:
        return ["L %s" % _pt(P[-1])]
    if depth < max_depth and len(P) > 10 and 1 < split < len(P) - 2:
        return (_fit_polyline(P[:split + 1], tol, depth + 1, max_depth)
                + _fit_polyline(P[split:], tol, depth + 1, max_depth))
    return [_fit_cubic(P)]


def _fit_closed(poly: np.ndarray, tol: float, corner_deg: float,
                smooth: float, min_len: float = 6.0,
                min_area: float = 16.0) -> str:
    """闭合轮廓 -> 一条 M...Z 子路径（角点处断开）。

    太小的轮廓直接丢弃：它们是量化残留，留着会被 gap-filler 描边描成小圆点
    / 碎线（4× 放大里额头那串橙色点就是这么来的）。丢掉后露出下层大色块，
    反而更干净。
    """
    P = np.asarray(poly, np.float64)
    if not np.allclose(P[0], P[-1], atol=1e-6):
        P = np.vstack([P, P[:1]])
    if len(P) < 8:
        return ""
    area = 0.5 * abs(float(np.dot(P[:-1, 0], P[1:, 1])
                           - np.dot(P[:-1, 1], P[1:, 0])))
    if area < min_area:
        return ""
    ring = _resample_ring(P[:-1], 1.0)
    n = len(ring)
    if n < 8 or float(np.hypot(*np.diff(np.vstack([ring, ring[:1]]), axis=0).T).sum()) < min_len:
        return ""
    # 轻度平滑抹掉 marching-squares 的逐像素抖动，真实角点钉回原位
    if smooth > 0:
        from scipy.ndimage import gaussian_filter1d
        ext = np.vstack([ring[-4:], ring, ring[:4]])
        sm = gaussian_filter1d(ext, smooth, axis=0, mode="nearest")[4:4 + n]
    else:
        sm = ring.copy()
    # 不把原始 ring 的角点钉回：钉回会把 marching-squares 的锯齿一并恢复，
    # 正是 4× 放大里头发毛刺的来源。强平滑后只在真正的尖角处断开即可。
    cs = _corner_indices(sm, corner_deg)
    if not cs:
        d = np.hypot(*(ring - ring[0]).T)
        j = int(d.argmax())
        cs = sorted({0, j}) if j > 0 else [0, n // 2]
    parts: list[str] = []
    for a, b in zip(cs, cs[1:] + [cs[0] + n]):
        idx = np.arange(a, b + 1) % n
        run = sm[idx]
        if len(run) < 2:
            continue
        parts += _fit_polyline(run, tol)
    if not parts:
        return ""
    return "M %s %s Z" % (_pt(sm[cs[0]]), " ".join(parts))


# --------------------------------------------------------------------------- #
# 图像 → 大色块
# --------------------------------------------------------------------------- #
def _iso_contours(mask: np.ndarray, scale: int = 2,
                  sigma: float = 0.55, level: float = 0.5,
                  offset: tuple[float, float] = (0.0, 0.0)) -> list[np.ndarray]:
    """软掩膜的亚像素等值线；padding 用 replicate 避免贴边产生伪轮廓。

    mask 通常是某个颜色的 bbox 裁剪，offset 把坐标平移回全图。
    """
    import cv2
    from skimage import measure
    p = cv2.copyMakeBorder(mask.astype(np.float32), 2, 2, 2, 2,
                           cv2.BORDER_REPLICATE)
    f = cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    f = cv2.GaussianBlur(f, (0, 0), sigma * scale)
    out = []
    for c in measure.find_contours(f, level):
        xy = (c[:, ::-1] + 0.5) / scale - 2.0
        xy[:, 0] += offset[0]
        xy[:, 1] += offset[1]
        if len(xy) >= 8:
            out.append(xy)
    return out


def _oklab_palette(pal: np.ndarray) -> np.ndarray:
    """(K,3) uint8 -> OKLab，用于合并时判断颜色远近。"""
    from qq3d_like import _srgb_to_oklab
    return _srgb_to_oklab(pal[None, :, :].astype(np.float32) / 255.0)[0]


def merge_small(labels: np.ndarray, pal: np.ndarray, min_area: int = 60,
                min_hw: float = 2.4, iters: int = 3,
                keep_de: float = 0.055) -> np.ndarray:
    """把过小 / 过窄的色带并入颜色最近的大邻居（参考 Lumina-Vec regions2）。

    与邻居颜色差很大的小高光/小细节会被保留（这是刻意的细节，不是噪点）。
    """
    from scipy import ndimage as ndi
    if iters <= 0:
        return labels
    out = labels.copy()
    labpal = _oklab_palette(pal)
    _cross = ndi.generate_binary_structure(2, 1)   # 4 邻接，等价 measure.label(connectivity=1)
    for _ in range(iters):
        # v7.19.0：去掉 skimage 依赖（打包环境里没有 scikit-image，xixi/flatstyle/smooth
        # 会在这一步直接 ModuleNotFoundError）。scipy 的 label 同结构结果一致。
        reg, _nl = ndi.label(out.astype(np.int32) + 1, structure=_cross)
        n = int(reg.max()) + 1
        area = np.bincount(reg.ravel(), minlength=n)
        edge = np.zeros(reg.shape, bool)
        edge[:, 1:] |= reg[:, 1:] != reg[:, :-1]
        edge[:, :-1] |= reg[:, 1:] != reg[:, :-1]
        edge[1:] |= reg[1:] != reg[:-1]
        edge[:-1] |= reg[1:] != reg[:-1]
        dt = ndi.distance_transform_edt(~edge)
        hw = np.asarray(ndi.maximum(dt, reg, index=np.arange(n)))
        bad = (area < min_area) | (hw < min_hw)
        bad[0] = False
        if not bad.any():
            break
        objs = ndi.find_objects(reg)
        changed = 0
        order = sorted(np.flatnonzero(bad), key=lambda r: area[r])
        for r in order:
            sl = objs[r - 1]
            if sl is None:
                continue
            y0, y1 = max(0, sl[0].start - 1), min(reg.shape[0], sl[0].stop + 1)
            x0, x1 = max(0, sl[1].start - 1), min(reg.shape[1], sl[1].stop + 1)
            sub = reg[y0:y1, x0:x1]
            m = sub == r
            ring = ndi.binary_dilation(m, structure=np.ones((3, 3), np.uint8)) & ~m
            nb = np.unique(sub[ring])
            nb = nb[(nb > 0) & (nb < n)]
            good = nb[~bad[nb]] if len(nb) else nb
            if not len(good):
                good = nb
            if not len(good):
                continue
            vals = out[y0:y1, x0:x1][m]
            if not len(vals):
                continue
            my = int(np.bincount(vals, minlength=len(pal)).argmax())
            cand = []
            for q in good:
                sel = (sub == q) & ring
                v = out[y0:y1, x0:x1][sel]
                if not len(v):
                    continue
                c = int(np.bincount(v).argmax())
                d = float(((labpal[c] - labpal[my]) ** 2).sum())
                cand.append((d, -float(area[q]), c))
            if not cand:
                continue
            cand.sort()
            d, _, best = cand[0]
            if math.sqrt(d) > keep_de and area[r] >= 8:
                continue
            out[y0:y1, x0:x1][m] = best
            changed += 1
        if not changed:
            break
    return out


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def trace(src: str, dst: str | None = None, *, colors: int = 64,
          tol: float = 2.8, corner_deg: float = 60.0, smooth: float = 2.6,
          min_area: int = 170, min_hw: float = 3.5, merge_iters: int = 4,
          label_smooth: float = 2.6, iso_scale: int = 2, iso_sigma: float = 1.5,
          contour_min_area: float = 18.0, keep_de: float = 0.18,
          seam: float = 1.2, dilate: int = 1, max_side: int = 0, seed: int = 7,
          progress=None) -> dict:
    """几何化重绘，返回 {"svg","paths","colors","width","height","bytes","ms"}。"""
    t0 = time.time()

    def step(msg):
        if progress:
            progress(msg)

    import qq3d_like as _qq

    step("读取图片…")
    im = Image.open(src).convert("RGB")
    if max_side and max(im.size) > max_side:
        r = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))),
                       Image.LANCZOS)
    w, h = im.size

    step("感知量化 %d 级（几何化重绘：少量大色块）…" % int(colors))
    q, labels, pal = _qq.quantize_perceptual(im, int(colors), seed=seed,
                                             return_labels=True)
    if label_smooth > 0:
        step("标签平滑（磨掉像素阶梯 / 杂斑）…")
        labels = _qq._label_smooth(labels, len(pal), label_smooth)
    if merge_iters > 0:
        step("区域规整（小 / 窄色带并入最近邻）…")
        labels = merge_small(labels, pal, min_area=min_area, min_hw=min_hw,
                             iters=merge_iters, keep_de=keep_de)

    from scipy import ndimage as ndi
    counts = np.bincount(labels.ravel(), minlength=len(pal))
    order = [int(i) for i in np.argsort(-counts) if counts[i] > 0]
    boxes = ndi.find_objects(labels.astype(np.int32) + 1)
    parts: list[str] = []
    used_colors: set[str] = set()
    npaths = 0
    step("几何基元拟合（角点切开 → L / A / Q / C）…")
    for ci, c in enumerate(order):
        if counts[c] < max(4, min_area // 3):
            continue
        sl = boxes[c] if c < len(boxes) else None
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - 3), min(h, sl[0].stop + 3)
        x0, x1 = max(0, sl[1].start - 3), min(w, sl[1].stop + 3)
        if y1 - y0 < 3 or x1 - x0 < 3:
            continue
        crop = (labels[y0:y1, x0:x1] == c).astype(np.float32)
        if dilate > 0:
            crop = ndi.binary_dilation(crop, iterations=dilate).astype(np.float32)
        try:
            contours = _iso_contours(crop, scale=iso_scale, sigma=iso_sigma,
                                     offset=(x0, y0))
        except Exception:
            continue
        subs = []
        for xy in contours:
            d = _fit_closed(xy, tol, corner_deg, smooth,
                            min_area=contour_min_area)
            if d:
                subs.append(d)
        if not subs:
            continue
        col = "#%02x%02x%02x" % tuple(int(v) for v in pal[c])
        d = " ".join(subs)
        # 消缝：dilate 让相邻色块重叠贴合；seam>0 时再叠一道「同色」描边把残余
        # 亚像素缝抹平（同色 → 无对比伪影；去掉 non-scaling-stroke 避免圆头在
        # 拐点炸出星芒尖刺）。这是 Lumina 的 qq3d-gap-filler 布局，导入时剥离。
        if seam > 0:
            parts.append(
                '<path class="qq3d-gap-filler" d="%s" fill="none" stroke="%s" '
                'stroke-width="%.2f" stroke-linejoin="round" '
                'stroke-linecap="round"/>' % (d, col, seam))
        parts.append('<path d="%s" fill="%s"/>' % (d, col))
        used_colors.add(col)
        npaths += 1
        if progress and (ci + 1) % 20 == 0:
            step("  已拟合 %d/%d 个色层…" % (ci + 1, len(order)))

    bg = "#%02x%02x%02x" % tuple(int(v) for v in im.getpixel((0, 0)))
    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
           'width="%d" height="%d" viewBox="0 0 %d %d">\n'
           '<rect x="0" y="0" width="%d" height="%d" fill="%s"/>\n'
           % (w, h, w, h, w, h, bg)
           + "\n".join(parts) + "\n</svg>\n")
    if dst:
        io.open(dst, "w", encoding="utf-8", newline="\n").write(svg)
    step("完成 %d 个大色块 / %d 色（几何化重绘）" % (npaths, len(used_colors)))
    return {
        "svg": svg, "paths": npaths, "colors": len(used_colors),
        "width": w, "height": h,
        "bytes": os.path.getsize(dst) if dst else len(svg.encode("utf-8")),
        "ms": int((time.time() - t0) * 1000),
    }
