# -*- coding: utf-8 -*-
"""自研混合描摹引擎（v7.7.0，MODE_HYBRID）——高细节插画碎线攻关落地。

组合两项已验证的自研能力：
  1. own_tracer.unify_ink_cc：黑帽 + 连通域筛选，把被量化切碎的暗细线
     （发丝 / 眼线 / 描边）统一成连续顶层；
  2. geom_trace 的长曲线基元拟合（L/A/Q/C，15 位坐标）用于墨线层，
     替代短 C 逐像素追阶梯——这是碎线的几何根因。

其余大色块走「角点切分 + 每段单条三次贝塞尔」快速拟合（v7.7.0 速度优化：
递归细分是旧原型 3.5 分钟/张的主瓶颈，实测占一半以上耗时）。

全程自研，零外部代码、零模型权重。详见 HANDOFF §27。
"""
from __future__ import annotations

import time

import numpy as np

import own_tracer as ot
import geom_trace as gt


def _fit_hybrid_closed(poly, tol, corner_deg, smooth):
    """闭合轮廓 -> M...Z：墨线层用长曲线基元（L/A/Q/C，高精度坐标）。"""
    P = np.asarray(poly, np.float64)
    if not np.allclose(P[0], P[-1], atol=1e-6):
        P = np.vstack([P, P[:1]])
    if len(P) < 8:
        return ''
    ring = gt._resample_ring(P[:-1], 1.0)
    n = len(ring)
    if n < 8:
        return ''
    if smooth > 0:
        from scipy.ndimage import gaussian_filter1d
        ext = np.vstack([ring[-4:], ring, ring[:4]])
        sm = gaussian_filter1d(ext, smooth, axis=0, mode='nearest')[4:4 + n]
    else:
        sm = ring.copy()
    cs = gt._corner_indices(sm, corner_deg)
    if not cs:
        d = np.hypot(*(ring - ring[0]).T)
        j = int(d.argmax())
        cs = sorted({0, j}) if j > 0 else [0, n // 2]
    parts = []
    for a, b in zip(cs, cs[1:] + [cs[0] + n]):
        idx = np.arange(a, b + 1) % n
        run = sm[idx]
        if len(run) < 2:
            continue
        parts += gt._fit_polyline(run, tol)
    if not parts:
        return ''
    return 'M %s %s Z' % (gt._pt(sm[cs[0]]), ' '.join(parts))


def _fast_closed_d(poly, corner_deg=55.0):
    """大色块快速拟合：角点切分 + 每段单条 C（不递归）。

    与 own_tracer.fit_closed 同样的预平滑与角点保护，但每段只解一次
    最小二乘三次贝塞尔——递归细分在 5MP 图上要 50 万次调用，是速度主瓶颈。
    大色块边界经标签平滑后已较光顺，单段 C 的偏差远小于像素级容差关注区。
    """
    P = np.asarray(poly, np.float64)
    if np.allclose(P[0], P[-1], atol=1e-6):
        P = P[:-1]
    p = ot._resample(P, 1.0)
    if len(p) < 4:
        return ''
    pad = np.r_[p[-2:], p, p[:2]]
    sm = (pad[:-4] + 2 * pad[1:-3] + 3 * pad[2:-2] + 2 * pad[3:-1] + pad[4:]) / 9.0
    corners = ot._corner_indices(p, corner_deg)
    if corners:
        sm[corners] = p[corners]
    n = len(sm)
    if not corners:
        ring = np.r_[sm, sm[:1]]
        t = ot._unit(sm[1] - sm[-1])
        ctrl = ot._fit_segment(ring, t, -t)
        segs = [ctrl]
    else:
        segs = []
        for start in corners:
            end = corners[(corners.index(start) + 1) % len(corners)]
            idx = np.arange(start, end + (n if end <= start else 0) + 1) % n
            q = sm[idx]
            if len(q) < 2:
                continue
            t1 = ot._unit(q[min(2, len(q) - 1)] - q[0])
            t2 = ot._unit(q[max(-3, -len(q))] - q[-1])
            segs.append(ot._fit_segment(q, t1, t2))
    if not segs:
        return ''
    d_ = ['M%.2f,%.2f' % tuple(segs[0][0])]
    for sg in segs:
        d_.append('C%.2f,%.2f %.2f,%.2f %.2f,%.2f' % (*sg[1], *sg[2], *sg[3]))
    return ''.join(d_) + 'Z'


def trace(rgb: np.ndarray, colors: int = 256, detail: int = 6,
          ink_kw: dict | None = None, ink_tol: float = 1.0,
          ink_corner: float = 50.0, ink_smooth: float = 0.8,
          iso_scale: int = 1, kmeans_iters: int = 10,
          progress=None) -> dict:
    """Raster -> SVG（混合引擎）。返回 {svg, layers, subpaths, ink_layers, ms}。"""
    import cv2
    t0 = time.time()

    def step(msg):
        if progress:
            progress(msg)

    h, w = rgb.shape[:2]
    step("混合引擎：保边预平滑…")
    work = cv2.bilateralFilter(rgb, 7, 30, 7)
    step("混合引擎：感知量化 %d 色…" % int(colors))
    palette = ot.build_palette(work, int(colors), iters=int(kmeans_iters))
    labels = ot.assign_palette(work, palette)
    labels = ot.smooth_labels(labels, sigma=0.7, iters=1)
    kw = dict(threshold=28.0, close=3, dark_luma=140.0, width_thresh=3.0,
              dark_margin=28.0, max_new_ratio=0.18)
    if ink_kw:
        kw.update(ink_kw)
    step("混合引擎：墨线统一（发丝/眼线连成整条）…")
    labels, ink_ids = ot.unify_ink_cc(rgb, labels, **kw)
    ink_set = set(int(x) for x in ink_ids)
    counts = np.bincount(labels.ravel())
    order = [int(i) for i in np.argsort(-counts) if counts[i] > 0]
    min_area = max(2.0, 24.0 - detail * 2.2)
    pad = 3
    small = cv2.resize(rgb, (48, 48), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    q = (small // 16).astype(np.int32)
    key = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    vals, freq = np.unique(key, return_counts=True)
    bg = np.median(small[key == vals[freq.argmax()]], axis=0).astype(int)
    paths = ['<path d="M0,0H%dV%dH0Z" fill="#%02x%02x%02x"/>'
             % (w, h, bg[0], bg[1], bg[2])]
    nsub = 0
    ink_sub = 0
    step("混合引擎：大色块快速拟合 + 墨线长曲线拟合…")
    for i, colour in enumerate(order):
        own = labels == colour
        ys, xs = np.nonzero(own)
        if len(ys) == 0:
            continue
        y0 = max(0, int(ys.min()) - pad); y1 = min(h, int(ys.max()) + 1 + pad)
        x0 = max(0, int(xs.min()) - pad); x1 = min(w, int(xs.max()) + 1 + pad)
        mask = own[y0:y1, x0:x1]
        is_ink = colour in ink_set
        if not is_ink:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
        soft = mask.astype(np.float32)
        parts = []
        for xy in ot._iso_regions(soft, int(iso_scale), 0.5):
            if len(xy) < 8:
                continue
            xy = xy + np.array([x0, y0])
            area = 0.5 * abs(np.dot(xy[:-1, 0], xy[1:, 1])
                             - np.dot(xy[1:, 0], xy[:-1, 1]))
            if area < (2.0 if is_ink else min_area):
                continue
            if is_ink:
                d = _fit_hybrid_closed(xy, ink_tol, ink_corner, ink_smooth)
                if d:
                    parts.append(d)
                    ink_sub += 1
            else:
                d = _fast_closed_d(xy)
                if d:
                    parts.append(d)
        if not parts:
            continue
        d = ''.join(parts)
        nsub += len(parts)
        mean = rgb[own].mean(0)
        hexc = '#%02x%02x%02x' % tuple(int(round(v)) for v in mean)
        sw = '0.3' if is_ink else '0.5'
        paths.append('<path d="%s" fill="%s" stroke="%s" stroke-width="%s" '
                     'fill-rule="evenodd"/>' % (d, hexc, hexc, sw))
        if progress and (i + 1) % 100 == 0:
            step("  已拟合 %d/%d 层…" % (i + 1, len(order)))
    body = '\n'.join(paths)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="%d" '
           'height="%d" viewBox="0 0 %d %d">\n%s\n</svg>' % (w, h, w, h, body))
    step("完成 %d 层（墨线 %d 层）" % (len(paths) - 1, len(ink_set)))
    return {'svg': svg, 'layers': len(paths) - 1, 'subpaths': nsub,
            'ink_layers': len(ink_set), 'ink_subpaths': ink_sub,
            'ms': int((time.time() - t0) * 1000)}