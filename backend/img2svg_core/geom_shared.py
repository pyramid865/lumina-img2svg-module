# -*- coding: utf-8 -*-
"""共边几何化重绘（seam-free mosaic）—— MODE_GEOM 的下一代原型。

v6.5 的 ``geom_trace`` 对每个颜色**独立**提软掩膜等值线：高斯模糊会让细窄
色带的峰值跌破 0.5 而消失，三色交界处各层轮廓也不互补，于是层间露出底色
→ 只能靠 ``qq3d-gap-filler`` 描边遮缝；描边又把微小轮廓描成小圆点，就是
4× 放大里发际线那串橙点 / 深色网状碎斑的来源。

本模块改成**硬标签 + 共享边界**：

* 标签图把每个像素严格分给一个颜色，本身是一张无缝拼图；
* 对二值（0/1）掩膜在 ``level=0.5`` 做 marching squares，不做高斯、不做
  LINEAR 上采样 —— 相邻两色在同一条像素分界线上得到同一组点（方向相反），
  天然共边；
* 平滑只作用在**轮廓折线**上（高斯 1D + 角点断开 + L/A/Q 拟合），不改变
  区域拓扑，所以既平滑又不留缝；
* 不再用 stroke 描边，改为在**几何层面**把每个环向外扩 ``expand`` 像素，
  相邻块重叠约 2*expand，彻底杜绝亚像素缝，同时小轮廓不会被描成圆点。

只依赖 numpy / PIL / cv2 / scipy / skimage / geom_trace / qq3d_like。
"""
from __future__ import annotations

import io
import os
import time

import numpy as np
from PIL import Image

import geom_trace as g
import qq3d_like as qq


def _expand_ring(poly: np.ndarray, delta: float) -> np.ndarray:
    """把闭合折线整体向外扩 delta 像素（保留尖角，不像 stroke 那样变圆点）。

    用相邻两条边的单位法线之和作顶点法线，并按夹角补偿长度；最后用面积
    变化自动判定方向，避免图像坐标系 y 向下导致的内外法线符号搞反。
    """
    P = np.asarray(poly, np.float64)
    if len(P) < 4:
        return P
    if np.allclose(P[0], P[-1], atol=1e-6):
        P = P[:-1]

    def signed_area(Q):
        return 0.5 * float(np.dot(Q[:, 0], np.roll(Q[:, 1], -1))
                           - np.dot(Q[:, 1], np.roll(Q[:, 0], -1)))

    e1 = P - np.roll(P, 1, axis=0)
    e2 = np.roll(P, -1, axis=0) - P
    l1 = np.maximum(np.hypot(*e1.T), 1e-9)
    l2 = np.maximum(np.hypot(*e2.T), 1e-9)
    n1 = np.c_[e1[:, 1], -e1[:, 0]] / l1[:, None]
    n2 = np.c_[e2[:, 1], -e2[:, 0]] / l2[:, None]
    n = n1 + n2
    nn = np.maximum(np.hypot(*n.T), 1e-9)
    n /= nn[:, None]
    # 尖角处法线变短，需按 1/cos(half-angle) 放大才能保持等距偏移
    scale = np.clip(2.0 / np.maximum(nn, 0.35), 1.0, 3.0)
    Q = P + delta * n * scale[:, None]
    if abs(signed_area(Q)) < abs(signed_area(P)):
        Q = P - delta * n * scale[:, None]
    return Q


def _hard_contours(labels: np.ndarray, c: int,
                   box: tuple[int, int, int, int], pad: int = 2) -> list[np.ndarray]:
    """硬标签的共边等值线。padding 用 constant 0，让贴边色块沿画布外沿闭合。"""
    import cv2
    from skimage import measure

    y0, y1, x0, x1 = box
    crop = (labels[y0:y1, x0:x1] == c).astype(np.float32)
    p = cv2.copyMakeBorder(crop, pad, pad, pad, pad,
                           cv2.BORDER_CONSTANT, value=0.0)
    # 关键：不 resize、不 GaussianBlur。二值 0/1 在 0.5 处的等值线正好落在
    # 像素分界线上，相邻颜色共享同一条线。
    out: list[np.ndarray] = []
    for cc in measure.find_contours(p, 0.5, fully_connected="low"):
        xy = cc[:, ::-1].copy()
        xy[:, 0] += x0 - pad
        xy[:, 1] += y0 - pad
        if len(xy) >= 8:
            out.append(xy)
    return out


def trace(src: str, dst: str | None = None, *, colors: int = 64,
          tol: float = 2.8, corner_deg: float = 60.0, smooth: float = 2.6,
          label_smooth: float = 2.2, min_area: int = 170,
          min_hw: float = 3.5, merge_iters: int = 4, keep_de: float = 0.18,
          contour_min_area: float = 18.0, expand: float = 0.65,
          seam: float = 0.8,
          max_side: int = 0, seed: int = 7, progress=None) -> dict:
    """共边几何化重绘，返回与 geom_trace.trace 同结构的 dict。"""
    t0 = time.time()

    def step(msg):
        if progress:
            progress(msg)

    step("读取图片…")
    im = Image.open(src).convert("RGB")
    if max_side and max(im.size) > max_side:
        r = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))),
                       Image.LANCZOS)
    w, h = im.size

    step("感知量化 %d 级…" % int(colors))
    q, labels, pal = qq.quantize_perceptual(im, int(colors), seed=seed,
                                            return_labels=True)
    if label_smooth > 0:
        step("标签平滑…")
        labels = qq._label_smooth(labels, len(pal), label_smooth)
    if merge_iters > 0:
        step("区域规整（小 / 窄色带并入最近邻）…")
        labels = g.merge_small(labels, pal, min_area=min_area,
                               min_hw=min_hw, iters=merge_iters,
                               keep_de=keep_de)

    from scipy import ndimage as ndi
    counts = np.bincount(labels.ravel(), minlength=len(pal))
    order = [int(i) for i in np.argsort(-counts) if counts[i] > 0]
    boxes = ndi.find_objects(labels.astype(np.int32) + 1)
    parts: list[str] = []
    used: set[str] = set()
    npaths = 0
    nsub = 0
    step("共边轮廓 + 几何基元拟合（L / A / Q）…")
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
        subs: list[str] = []
        for xy in _hard_contours(labels, c, (y0, y1, x0, x1)):
            if expand:
                xy = _expand_ring(xy, expand)
            d = g._fit_closed(xy, tol, corner_deg, smooth,
                              min_area=contour_min_area)
            if d:
                subs.append(d)
        if not subs:
            continue
        col = "#%02x%02x%02x" % tuple(int(v) for v in pal[c])
        d = " ".join(subs)
        # 预览遮缝：同色描边（Lumina 导入时剥离，不进打印）。不用
        # non-scaling-stroke（会导致圆头在拐点炸出星芒尖刺），用固定宽度 +
        # 圆角连接即可把残余亚像素缝抹成同色，预览与嘻嘻宝一致。
        if seam > 0:
            parts.append(
                '<path class="qq3d-gap-filler" d="%s" fill="none" stroke="%s" '
                'stroke-width="%.2f" stroke-linejoin="round" '
                'stroke-linecap="round"/>' % (d, col, seam))
        parts.append('<path d="%s" fill="%s"/>' % (d, col))
        used.add(col)
        npaths += 1
        nsub += len(subs)
        if progress and (ci + 1) % 20 == 0:
            step("  已拟合 %d/%d 个色层…" % (ci + 1, len(order)))

    bg = qq._bg_color(q)
    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
           'width="%d" height="%d" viewBox="0 0 %d %d">\n'
           '<rect x="0" y="0" width="%d" height="%d" fill="%s"/>\n'
           % (w, h, w, h, w, h, bg)
           + "\n".join(parts) + "\n</svg>\n")
    if dst:
        io.open(dst, "w", encoding="utf-8", newline="\n").write(svg)
    step("完成 %d 个共边色层 / %d 子路径 / %d 色" % (npaths, nsub, len(used)))
    return {"svg": svg, "paths": npaths, "subpaths": nsub,
            "colors": len(used), "width": w, "height": h,
            "bytes": os.path.getsize(dst) if dst else len(svg.encode("utf-8")),
            "ms": int((time.time() - t0) * 1000)}
