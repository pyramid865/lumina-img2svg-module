# -*- coding: utf-8 -*-
"""自研「平滑区大色块」描摹引擎（v7.9.0，MODE_SMOOTH）——高细节插画画质攻关落地。

## 设计依据（2026-11-16，V:\\QQ 302 个嘻嘻宝 SVG 逆向实证）
参考工具（用户称「嘻嘻宝」）在高细节插画上的签名：
  - 平滑区（皮肤 / 天空 / 纯色背景）画成**极少数大色块**，一整块干净色；
  - 细节区（发丝 / 碎晶 / 眼睛）才精细描摹；
  - 全部边界用 Q/A 长弧 + 高精度几何拟合。
我们的旧输出把整图统一按色差量化：皮肤被切成十几个色阶 → 块状色斑；
边界用短 L/C 段追像素阶梯 → 发丝碎线、暗部灰白噪点。

## 本引擎做法（纯自研，零外部代码 / 零模型权重）
1. OKLab 感知量化到 ~110 色（cel.flatten / cel.quantize，项目自有）；
2. **局部梯度幅值**识别「平滑区」（低梯度连通大区）——不依赖颜色判据，泛化好；
   平滑区内强制统一为该区主色（消除皮肤色阶斑块）；高梯度区保留原量化（保发丝 / 晶体）；
3. regionvec.merge_regions 区域合并（项目自有）；
4. regionvec._arc_path 做 L/A/Q 长弧拟合（项目自有，与参考工具同款命令家族）。

## v7.9.0 阴影修正（用户 2026-11-16 手臂阴影 / 发丝反馈）
- v7.8.0 把**任何**低梯度连通区强制统一成主色，导致手臂/腋下的柔和阴影被
  抹成一整块硬边粉色（用户反馈「阴影处理不好」）。v7.9 新增**梯度方差护栏**：
  平滑区内 OKLab 亮度 p92-p8 超过 `gradient_spread` 时视为「渐变阴影区」，
  不做整块统一，保留量化色阶；只有真正平坦区才统一主色消色斑。
- 渐变区另外保留 `gradient_speckle`（默认 0 = 不吞并，避免把阴影色阶当斑点删掉）。
- 区域合并阈值默认 0.02（v7.8 为 0.04），减少相邻阴影色被并成一块。
- 自动预设 anime / photo 已路由到本引擎（此前走 vtracer，头发碎线）。
- 对比图：`_dev/arm_shadow_probe/arm_full_compare.png`、`hair_v4_compare.png`、
  `armpit_zoom.png` / `armpit_v4.png`。

## 成果（59Q 点唇图，原尺寸 941×1672 实测）
皮肤从十几个色阶斑块 → 一整块干净浅色，接近参考工具；暗部灰白噪点消失；
发丝连贯成片；眼睛 / 晶体细节保留。约 22 秒/张。
对比图：`_dev/gojo_probe2/smooth_cmp_*.png`、`FINAL_*.png`。

## 合规
全部算法为本项目独立实现；依赖为已有的 numpy / OpenCV / Pillow（宽松许可），
未复制任何第三方源码。详见 THIRD_PARTY_NOTICES.md。
"""
from __future__ import annotations

import re
import time

import cv2
import numpy as np
from PIL import Image, ImageFilter

from cel import flatten, quantize as _quantize
from color import palette_rgb, srgb_to_oklab
import regionvec as rv


def trace(rgb: np.ndarray, colors: int = 110, detail: int = 6,
          smooth_grad: float = 10.0, min_smooth_area: int = 300,
          merge_de: float = 0.02, arc_tol: float = 1.0,
          gradient_spread: float = 0.055, gradient_speckle: int = 0,
          progress=None) -> dict:
    """Raster -> SVG（平滑区大色块引擎）。

    rgb: H×W×3 uint8 原图（已由 core 完成保色超分）。
    返回 {svg, layers, subpaths, colors, ms}。
    """
    t0 = time.time()

    def step(msg):
        if progress:
            progress(msg)

    im = Image.fromarray(rgb).convert('RGB')
    # 工作网格 1200 宽：与既有引擎一致，兼顾细节与速度
    grid = 1200
    if im.width != grid:
        im = im.resize((grid, max(1, round(im.height * grid / im.width))), Image.LANCZOS)
    im = im.filter(ImageFilter.MedianFilter(3))
    arr = np.asarray(im)

    step("平滑大色块：感知量化 %d 色…" % int(colors))
    lab = srgb_to_oklab(arr.astype(np.float32) / 255.0)
    flat, _base = flatten(lab, 2)
    labels, cent = _quantize(flat, int(colors), power=0.5)

    # 平滑区检测：Sobel 梯度幅值低 + 形态学闭合 + 连通域面积筛选
    step("平滑大色块：识别平滑区并统一主色…")
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.hypot(gx, gy), (9, 9), 0)
    smooth = (mag < float(smooth_grad)).astype(np.uint8)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    ncc, cc = cv2.connectedComponents(smooth)
    unified = 0
    L = lab[..., 0]
    gradient_kept = 0
    for ci in range(1, ncc):
        m = cc == ci
        if int(m.sum()) < int(min_smooth_area):
            continue
        vals = labels[m]
        if not len(vals):
            continue
        main = int(np.bincount(vals, minlength=len(cent)).argmax())

        # v7.9 梯度护栏：低梯度 ≠ 无明暗变化。手臂阴影 / 腋下暗部亮度跨度大，
        # 整块统一会把柔和渐变压成硬边大色块。这类连通区保留量化色阶。
        lv = L[m]
        spread = float(np.percentile(lv, 92) - np.percentile(lv, 8))
        if spread > float(gradient_spread):
            if int(gradient_speckle) > 0:
                for k in np.unique(vals):
                    if int(k) == main:
                        continue
                    mk = (m & (labels == int(k))).astype(np.uint8)
                    n2, cc2 = cv2.connectedComponents(mk)
                    for j in range(1, n2):
                        comp = cc2 == j
                        if int(comp.sum()) <= int(gradient_speckle):
                            labels[comp] = main
                            unified += int(comp.sum())
            gradient_kept += int(m.sum())
            continue

        labels[m] = main
        unified += int(m.sum())

    labels = rv.merge_regions(labels, lab, cent, min_area=40, max_de=float(merge_de))
    pal = palette_rgb(cent)
    h, w = labels.shape
    S = 2
    labs = np.repeat(np.repeat(labels, S, axis=0), S, axis=1)
    counts = np.bincount(labels.ravel(), minlength=len(pal))
    order = [int(i) for i in np.argsort(-counts) if counts[i] > 0]

    step("平滑大色块：L/A/Q 长弧拟合…")
    fills = []
    nsub = 0
    for k in order:
        mask = (labs == k).astype(np.uint8)
        if not mask.any():
            continue
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        d = []
        h0h = hier[0] if hier is not None else []
        for ci, c in enumerate(cnts):
            area = abs(cv2.contourArea(c))
            is_hole = bool(h0h[ci][3] != -1) if len(h0h) > ci else False
            if is_hole and area < 24 * S * S:
                continue
            if len(c) < 4 or area < 8 * S * S:
                continue
            pts = cv2.approxPolyDP(c, 0.6, True).reshape(-1, 2).astype(float) / S
            if len(pts) < 3:
                continue
            p = rv._arc_path(pts, float(arc_tol))
            if p:
                d.append(p)
                nsub += 1
        if not d:
            continue
        rgbv = '#%02x%02x%02x' % tuple(int(v) for v in pal[k])
        fills.append('<path d="%s" fill="%s" fill-rule="evenodd"/>' % (''.join(d), rgbv))
        if progress and len(fills) % 60 == 0:
            step("  已拟合 %d 层…" % len(fills))

    bg = order[0] if order else 0
    body = '<rect x="0" y="0" width="%d" height="%d" fill="#%02x%02x%02x"/>\n' % (
        w, h, *tuple(int(v) for v in pal[bg]))
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
           'width="%d" height="%d" viewBox="0 0 %d %d">\n%s%s\n</svg>' % (
               w, h, w, h, body, '\n'.join(fills)))

    step("完成 %d 层（统一 %.0f%%，渐变保留 %.0f%%）" % (
        len(fills), 100.0 * unified / labels.size,
        100.0 * gradient_kept / labels.size))
    return {'svg': svg, 'layers': len(fills), 'subpaths': nsub,
            'colors': len(fills), 'ms': int((time.time() - t0) * 1000),
            'smooth_ratio': round(100.0 * unified / labels.size, 1),
            'gradient_ratio': round(100.0 * gradient_kept / labels.size, 1)}
