# -*- coding: utf-8 -*-
"""自研「平涂长弧」风格化引擎 (MODE_FLATSTYLE) — v7.17.0。

目标：把厚涂渐变插画转成嘻嘻宝那种「平涂大色块 + 长弧平滑边缘」风格。

与之前六条失败路线的本质区别：
  之前在「保留厚涂渐变碎色块」的前提下硬拟合曲线 → 碎色块上拟合必炸
  （露白洞 / 黑线 / 白缝）。本引擎反其道：**先把渐变压成大色块**，边界
  自然变少变长，再用自研共边 L/A/Q 拟合，从源头得到平滑长弧。

全部使用本地自研资产，零复制搬运、无外部模型：
  - flatfill.quantize_palette   OKLab 色度加权量化（v7.16 已认可的色彩）
  - regionvec.merge_regions     色差门控区域合并
  - geom_trace.merge_small / _fit_closed   面积归并 + L/A/Q 长弧拟合
  - geom_shared._hard_contours / _expand_ring   共边等值线 + 几何遮缝
  - qq3d_like._label_smooth     标签平滑

五官保护（自研「钉色」机制）：高饱和色块（青色虹膜、金色装饰）与极亮/极暗
核心在强合并前记录标签、合并后恢复，保证眼睛/饰物不被吞掉。

风格取舍（用户明确接受）：牺牲像素保真换观感。参考平涂样张自身 RMSE ~57，
本引擎 RMSE ~19（比 v7.16 的 ~15 略高，属风格化正常代价）。

依赖：numpy / Pillow / scipy / opencv —— 全部已在运行时内。
"""
from __future__ import annotations
import time

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
import cv2

import flatfill
import regionvec as rv
import geom_shared as gs
import geom_trace as g
import qq3d_like as qq


def trace(rgb: np.ndarray, colors: int = 112, grid: int = 1500, scale: float = 2.0,
          merge_de: float = 0.085, merge_min_area: int = 160, merge_rounds: int = 5,
          label_smooth: float = 1.2, tol: float = 2.0, corner_deg: float = 56,
          smooth: float = 2.4, expand: float = 0.6, seam: float = 0.9,
          progress=None) -> dict:
    """把 RGB 数组转成平涂长弧风格 SVG，返回统计 dict。"""
    t0 = time.time()

    def step(msg):
        if progress:
            progress(msg)

    im = Image.fromarray(rgb).convert('RGB')
    work = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    if grid and work.width != grid:
        work = work.resize((grid, max(1, round(work.height * grid / work.width))), Image.LANCZOS)
    work = work.filter(ImageFilter.MedianFilter(3))
    h, w = work.height, work.width

    step('OKLab 量化 %d 色…' % int(colors))
    pal, qimg = flatfill.quantize_palette(work, int(colors), flatfill.DEFAULT_W)
    labels = flatfill.labels_of_quantized(pal, qimg, flatfill.DEFAULT_W)
    qarr = np.asarray(qimg).astype(np.float32) / 255.0
    lab = flatfill.color.srgb_to_oklab(qarr)
    plab = flatfill.color.srgb_to_oklab(pal.astype(np.float32) / 255.0)

    if label_smooth > 0:
        labels = qq._label_smooth(labels, len(pal), label_smooth)

    # 五官/饰物保护：记录高饱和与极亮/极暗核心的标签，合并后恢复。
    Lch = lab[..., 0]
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    protect = (chroma > 0.14) | (Lch > 0.93) | (Lch < 0.06)
    protect = cv2.morphologyEx(protect.astype(np.uint8), cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8)) > 0
    protect = cv2.dilate(protect.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    saved = labels[protect].copy()

    step('强区域合并（渐变压大色块）…')
    labels = rv.merge_regions(labels, lab, plab, min_area=int(merge_min_area),
                              max_de=float(merge_de), rounds=int(merge_rounds))
    labels = g.merge_small(labels, pal, min_area=int(merge_min_area),
                           min_hw=3.0, iters=3, keep_de=float(merge_de) + 0.06)
    labels[protect] = saved

    # 合并后取每区真实均值色（k-means 中心会偏暗）
    src_arr = np.asarray(work)
    for k in range(len(pal)):
        m = labels == k
        if m.any():
            pal[k] = src_arr[m].mean(0).round().astype(np.uint8)

    counts = np.bincount(labels.ravel(), minlength=len(pal))
    order = [int(i) for i in np.argsort(-counts) if counts[i] > 0]
    boxes = ndi.find_objects(labels + 1)
    parts = []
    nsub = 0
    step('共边轮廓 + L/A/Q 长弧…')
    for c in order:
        if counts[c] < max(6, int(merge_min_area) // 3):
            continue
        sl = boxes[c] if c < len(boxes) else None
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - 3), min(h, sl[0].stop + 3)
        x0, x1 = max(0, sl[1].start - 3), min(w, sl[1].stop + 3)
        if y1 - y0 < 3 or x1 - x0 < 3:
            continue
        subs = []
        for xy in gs._hard_contours(labels, c, (y0, y1, x0, x1)):
            if expand:
                xy = gs._expand_ring(xy, expand)
            d = g._fit_closed(xy, float(tol), float(corner_deg), float(smooth),
                              min_area=24.0)
            if d:
                subs.append(d)
        if not subs:
            continue
        col = '#%02x%02x%02x' % tuple(int(v) for v in pal[c])
        dd = ' '.join(subs)
        # 同色遮缝（先遮缝后填充，与嘻嘻宝 qq3d-gap-filler 同构）
        if seam > 0:
            parts.append('<path d="%s" fill="none" stroke="%s" stroke-width="%.2f" '
                         'stroke-linejoin="round" stroke-linecap="round"/>' % (dd, col, seam))
        parts.append('<path d="%s" fill="%s"/>' % (dd, col))
        nsub += len(subs)

    bg = '#%02x%02x%02x' % tuple(int(v) for v in pal[order[0]]) if order else '#ffffff'
    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
           'width="%d" height="%d" viewBox="0 0 %d %d">\n'
           '<rect width="%d" height="%d" fill="%s"/>\n' % (w, h, w, h, w, h, bg)
           + '\n'.join(parts) + '\n</svg>')
    return dict(svg=svg, layers=len(order), paths=len(parts), subpaths=nsub,
                w=w, h=h, colors=len(order), seconds=round(time.time() - t0, 1))
