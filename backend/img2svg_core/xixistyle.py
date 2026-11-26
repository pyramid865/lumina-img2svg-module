# -*- coding: utf-8 -*-
"""自研「嘻嘻宝平涂」风格引擎（MODE_XIXI）— v7.18.1。

风格规格来自 **312 个嘻嘻宝样本（V:\\QQ + 桌面）的统计中位值**：

| 项 | 嘻嘻宝中位 |
|---|---|
| 色数 | 78（p90 136） |
| 子路径 ÷ 色数 | 56 |
| 曲线构成 | Q 43% / A 26% / C 13% / L 18%（长弧 69%） |
| 描边 | 100% 样本都是「每区域 1 条填充 + 1 条同色描边」，描边占 path 50% |
| 描边宽度 | 100% 全是 2px（画布 2048~5760，相对宽约 0.08%） |
| 特效 | 0 渐变 / 0 滤镜 / 0 位图 |

注意：其 2px 是**固定值**，直接搬到我们 1600 网格会因相对宽度过大而吃区域
（实测 RMSE 9.89→14.20），故本引擎按画布宽度**归一化**：0.08% × 画布宽，最小 0.8px。

已知代价（用户明确接受）：这是「重绘成平涂」而不是保真转换。简单/平涂图上
观感干净清脆（强边保真可达 104~116%），**复杂图会明显丢细节**（嘻嘻宝自己在
复杂图上强边保真只有 29%）。

全部使用项目自有算法资产：flatfill 量化 / color Oklab / regionvec 合并 /
geom_shared 共边轮廓 / geom_trace L/A/Q 最简拟合 / qq3d_like 标签平滑。
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

# 嘻嘻宝样本中位规格
XIXI_COLORS = 78
STROKE_RATIO = 0.0004      # 相对画布宽度；实测 0.08%(1.92px@2400) 偏粗会吃区域，0.04% 最优


def detail_score(arr: np.ndarray) -> float:
    """粗略「细节密度」：强梯度像素占比（%）。越高越不适合平涂化。"""
    g = cv2.cvtColor(np.asarray(arr, np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2) / 4.0
    return float((mag > 40).mean() * 100)


def trace(rgb: np.ndarray, colors: int = XIXI_COLORS, grid: int = 1600,
          scale: float = 1.0, merge_de: float = 0.05, min_area: int = 150,
          merge_rounds: int = 2, label_smooth: float = 1.2, tol: float = 1.2,
          corner_deg: float = 56.0, smooth: float = 2.4, expand: float = 0.6,
          erode_px: int = 2, progress=None) -> dict:
    """把 RGB 数组转成嘻嘻宝风格 SVG，返回统计 dict。"""
    t0 = time.time()

    def step(msg):
        if progress:
            progress(msg)

    im = Image.fromarray(rgb).convert("RGB")
    src_arr = np.asarray(im)
    W0 = int(round(im.width * max(scale, 1.0)))
    work = im.resize((W0, max(1, round(im.height * W0 / im.width))), Image.LANCZOS)
    if grid and work.width != int(grid):
        work = work.resize((int(grid), max(1, round(work.height * int(grid) / work.width))),
                           Image.LANCZOS)
    work = work.filter(ImageFilter.MedianFilter(3))
    h, w = work.height, work.width
    arr = np.asarray(work).astype(np.float32)

    step("嘻嘻宝风格：OKLab 量化 %d 色…" % int(colors))
    pal, _q, idx = flatfill.quantize_palette(work, int(colors), flatfill.DEFAULT_W,
                                            with_index=True)
    labels = qq._label_smooth(idx.astype(np.int32), len(pal), label_smooth)

    lab = flatfill.color.srgb_to_oklab(arr / 255.0)
    plab = flatfill.color.srgb_to_oklab(pal.astype(np.float32) / 255.0)

    # 五官/饰物保护：高饱和 + 极亮/极暗核心，合并后恢复
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    protect = (chroma > 0.14) | (lab[..., 0] > 0.93) | (lab[..., 0] < 0.06)
    protect = cv2.morphologyEx(protect.astype(np.uint8), cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8)) > 0
    protect = cv2.dilate(protect.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    saved = labels[protect].copy()

    # 目标「子路径÷色数 ≈ 56」：按画布面积归一化最小区域
    ma = int(max(24, round(int(min_area) * (w / 1600.0) ** 2)))
    step("区域归并（目标每色约 56 块）…")
    labels = rv.merge_regions(labels, lab, plab, min_area=ma,
                              max_de=float(merge_de), rounds=int(merge_rounds))
    labels = g.merge_small(labels, pal, min_area=ma, min_hw=3.0, iters=3,
                           keep_de=float(merge_de) + 0.06)
    labels[protect] = saved

    # 核心取色（v7.17.1 同款）：腐蚀掉抗锯齿过渡再取均值
    pal = flatfill.core_colors(pal, labels, arr, erode_px=int(erode_px))

    counts = np.bincount(labels.ravel(), minlength=len(pal))
    order = [int(i) for i in np.argsort(-counts) if counts[i] > 0]
    boxes = ndi.find_objects(labels + 1)
    stroke = max(0.8, STROKE_RATIO * w)
    parts, nsub = [], 0
    step("共边轮廓 + L/A/Q 长弧（嘻嘻宝 69% 长弧）…")
    for c in order:
        if counts[c] < max(6, ma // 3):
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
        col = "#%02x%02x%02x" % tuple(int(v) for v in pal[c])
        dd = " ".join(subs)
        # 与嘻嘻宝同构：每区域一对（同色描边 + 填充），描边占 path 的 50%
        parts.append('<path d="%s" fill="none" stroke="%s" stroke-width="%.2f" '
                     'stroke-linejoin="round" stroke-linecap="round"/>' % (dd, col, stroke))
        parts.append('<path d="%s" fill="%s"/>' % (dd, col))
        nsub += len(subs)

    # 背景色取原图边框真实色（不能用调色板最大区域色，k-means 离群色会出纯蓝）
    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]])
    bg = "#%02x%02x%02x" % tuple(int(v) for v in np.median(border, 0).round())
    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="%d" height="%d" '
           'viewBox="0 0 %d %d">\n<title>xixi flat style</title>\n'
           '<rect width="%d" height="%d" fill="%s"/>\n' %
           (w, h, w, h, w, h, bg)) + "\n".join(parts) + "\n</svg>"

    ds = detail_score(arr)
    warn = ""
    if ds > 8.0:
        warn = "细节密度 %.1f%%（偏高）：平涂化会明显丢细节" % ds
    return dict(svg=svg, w=w, h=h, colors=len(order), paths=len(parts),
                subpaths=nsub, stroke=round(stroke, 2), detail=round(ds, 1),
                warning=warn, seconds=round(time.time() - t0, 1))
