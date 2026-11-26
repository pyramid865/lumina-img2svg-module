# -*- coding: utf-8 -*-
"""实验：色带式描摹引擎（MODE_BANDS 候选，暂未接入 core/GUI）。

## 为什么做
v7.9 的 `smooth_tracer` 是「低梯度区统一主色」：皮肤干净了，但柔和阴影会被压成
一块硬边大色块。Lumina-Vec 0.9.2 的 bands 路线相反——**保留渐变的多个色阶**，
只把色带边界做成平滑等值线：
  1. OKLab 频率平坦化采样 + k-means 量化（稀有高光/虹膜不被大面积肤色淹没）；
  2. 低对比区：距边缘自适应的边缘遮罩模糊 + 空间多数投票 → 平滑等值线边界；
     高对比区（线/纹理/眼睛）保留逐像素标签；
  3. 小碎块并入最相似邻居，但与邻居色差大的小点（眼高光/鼻点/嘴线）保留；
  4. 每个色块的最终颜色 = 该区域原图像素的中位数（不是量化中心）。

本模块为**独立自研实现**，只借鉴上述算法思想，未复制 Lumina-Vec 源码；
依赖均为项目已有的 numpy / OpenCV / Pillow / scipy。

## 用法
    import band_tracer
    res = band_tracer.trace(rgb, colors=112)
    open('out.svg','w').write(res['svg'])
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

from cel import quantize as _quantize
from color import kmeans_palette, palette_rgb, srgb_to_oklab
import regionvec as rv


def _assign_lab(lab: np.ndarray, cent: np.ndarray) -> np.ndarray:
    """Nearest palette entry in OKLab (one matmul per chunk, no NxKx3 array)."""
    h, w = lab.shape[:2]
    flat = lab.reshape(-1, 3).astype(np.float32)
    c = np.ascontiguousarray(cent, np.float32)
    cn = (c * c).sum(1)
    out = np.empty(len(flat), np.int32)
    for s in range(0, len(flat), 500_000):
        seg = flat[s:s + 500_000]
        out[s:s + 500_000] = (cn[None, :] - 2.0 * (seg @ c.T)).argmin(1)
    return out.reshape(h, w)


def _masked_blur(rgb: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution over `mask` pixels only; stays clear of edges so
    dark linework doesn't bleed into skin."""
    m = cv2.dilate((~mask).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
    mf = m.astype(np.float32)
    num = cv2.GaussianBlur(rgb.astype(np.float32) * mf[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(mf, (0, 0), sigma)[..., None]
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), rgb)


def _adaptive_blur(rgb: np.ndarray, low: np.ndarray, sigma: float,
                   near_sigma: float = 2.0, ramp=(4, 24)) -> np.ndarray:
    """Blur radius grows with distance from the nearest edge: small near outlines
    (keeps real shadows, no pale halo), large in open areas (smooth contours)."""
    d = ndi.distance_transform_edt(low)
    w = np.clip((d - ramp[0]) / float(ramp[1] - ramp[0]), 0, 1)[..., None].astype(np.float32)
    near = _masked_blur(rgb, low, near_sigma)
    far = _masked_blur(rgb, low, sigma)
    return np.clip(near * (1 - w) + far * w, 0, 255).astype(np.uint8)


def _soft_vote(labels: np.ndarray, sigma: float, mask: np.ndarray) -> np.ndarray:
    """Gaussian-weighted majority vote, only among `mask` pixels."""
    if sigma <= 0:
        return labels
    best = np.full(labels.shape, -1.0, np.float32)
    out = labels.copy()
    for c in np.unique(labels[mask]):
        ind = (labels == c).astype(np.float32) * mask
        p = cv2.GaussianBlur(ind, (0, 0), sigma)
        upd = (p > best) & mask
        out[upd] = c
        best[upd] = p[upd]
    return out


def _merge_specks(labels: np.ndarray, rgb: np.ndarray, max_area: int = 48,
                  keep_de: float = 0.10, tiny: int = 12, iters: int = 2) -> np.ndarray:
    """Small regions join their closest-colour neighbour, unless they contrast
    strongly with it (eye glints, moles, mouth lines stay)."""
    out = labels.copy()
    lab = srgb_to_oklab(rgb.astype(np.float32) / 255.0)
    for _ in range(int(iters)):
        reg = ndi.label(out + 1, structure=np.ones((3, 3)))[0]
        n_reg = int(reg.max()) + 1
        area = np.bincount(reg.ravel(), minlength=n_reg)
        small = area < int(max_area)
        small[0] = False
        if not small[reg].any():
            break
        objs = ndi.find_objects(reg)
        changed = 0
        for r in np.flatnonzero(small):
            sl = objs[r - 1]
            if sl is None:
                continue
            m = reg[sl] == r
            ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
            if not ring.any():
                continue
            nb, counts = np.unique(out[sl][ring], return_counts=True)
            nb = nb[(nb >= 0) & (nb != out[sl][m][0])]
            if not len(nb):
                continue
            my = lab[sl][m].mean(0)
            best, bd = -1, 1e9
            for q in nb:
                qm = (out[sl] == q) & ring
                if not qm.any():
                    continue
                c = lab[sl][qm].mean(0)
                d = float(np.sqrt(((c - my) ** 2).sum()))
                if d < bd:
                    bd, best = d, int(q)
            if best < 0:
                continue
            # a small/thin region clearly distinct from every neighbour is a detail, not noise
            if bd > float(keep_de) and int(area[r]) >= int(tiny):
                continue
            sub = out[sl]
            sub[m] = best
            out[sl] = sub
            changed += 1
        if not changed:
            break
    return out


def _region_palette(labels: np.ndarray, rgb: np.ndarray, mode: str = "median") -> np.ndarray:
    """Final colour per label from that region's ORIGINAL pixels.

    median: robust to a few dark pixels (line remnants) inside a fill.
    mean:   keeps the overall tone of a soft-light region (pale skin is not
            pulled to its brightest plateau); better for shading bands.
    core:   mean of the region's *interior* pixels (distance-transform weighted).
            A 1–2px outline averages its dark core together with anti-aliased
            light neighbours, so the drawn line comes out washed-out and the
            picture reads "soft"; taking only the interior keeps the true dark
            value. Big fills are unaffected (their interior is the fill).
    """
    n = int(labels.max()) + 1
    pal = np.zeros((n, 3), np.uint8)
    flat = labels.ravel()
    pix = rgb.reshape(-1, 3)
    if mode == "core":
        # 逐标签在自身 bbox 内做距离变换：整图做 128 次太慢（4M px × 128）。
        objs = ndi.find_objects(labels, max_label=n)
        for c in np.unique(labels):
            sl = objs[int(c)] if int(c) < len(objs) else None
            if sl is None:
                continue
            m = (labels[sl] == int(c)).astype(np.uint8)
            if not m.any():
                continue
            dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
            keep = dt >= max(0.9, 0.45 * float(dt.max()))
            if not keep.any():
                keep = m > 0
            v = rgb[sl][keep]
            pal[c] = v.mean(0).astype(np.uint8)
        return pal
    for c in np.unique(labels):
        sel = flat == c
        if not sel.any():
            continue
        v = pix[sel]
        pal[c] = (np.median(v, 0) if mode == "median" else v.mean(0)).astype(np.uint8)
    return pal


def trace(rgb: np.ndarray, colors: int = 112, grid: int = 1200,
          edge_thr: float = 0.40, blur_sigma: float = 12.0, vote_sigma: float = 3.0,
          near_sigma: float = 2.0, speck_area: int = 48, speck_keep_de: float = 0.10,
          highlight_l: float = 0.0, palette_mode: str = "median", detail_colors: int = 48,
          poly_eps: float = 1.0, corner_deg: float = 45.0, contour_smooth: int = 1,
          line_layer: bool = False, line_keep: float = 0.55, line_seed: float = 0.85,
          stroke_layer: bool = False, stroke_thr: float = 0.34,
          stroke_max_width: float = 4.6, stroke_min_len: float = 9.0,
          stroke_inpaint: bool = True, denoise: bool = True,
          arc_tol: float = 1.0, subpixel: int = 1, progress=None) -> dict:
    """Raster -> SVG（色带式）. Returns {svg, layers, subpaths, colors, ms}."""
    t0 = time.time()

    def step(msg):
        if progress:
            progress(msg)

    im = Image.fromarray(rgb).convert('RGB')
    # grid<=0：按原分辨率描摹（不缩放）。v7.11 起由 core 传入"原生尺寸（上限
    # 2600）"，因为死钉 1600 会把 1400+ 宽的原图降采样、把小图插值放大，两者
    # 都让成品发软——这是用户反馈"糊"的主因之一。
    if grid and im.width != int(grid):
        im = im.resize((int(grid), max(1, round(im.height * int(grid) / im.width))), Image.LANCZOS)
    if denoise:
        # 中值滤波压 JPEG 噪点；但高清化之后它会把刚补出来的细节又磨掉，
        # v7.11 起可关（denoise=0），由 HD 预处理负责降噪。
        im = im.filter(ImageFilter.MedianFilter(3))
    arr = np.asarray(im)
    h, w = arr.shape[:2]

    lab = srgb_to_oklab(arr.astype(np.float32) / 255.0)
    L = lab[..., 0].astype(np.float32)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.dilate(np.hypot(gx, gy), np.ones((3, 3), np.uint8))

    step("色带引擎：量化 %d 色…" % int(colors))
    # 频率平坦化采样 + k-means：稀有但独特的颜色（虹膜/高光）保留自己的入口
    _raw, cent = _quantize(lab, int(colors), power=0.4)
    if detail_colors and edge.any():
        # 高对比像素单独再分一组色：眼睛/高光/线条本来只占几像素，主色板给不了
        # 它们自己的入口，会被附近的填充色吸走（高光变成一块死白）。
        sel = lab[edge > 0]
        if len(sel) >= int(detail_colors) * 4:
            rng = np.random.default_rng(3)
            sub = sel[rng.choice(len(sel), min(40000, len(sel)), replace=False)]
            dcent = kmeans_palette(sub, int(detail_colors), iters=12, seed=3)
            cent = np.concatenate([cent, dcent.astype(np.float32)], 0)
    raw_labels = _assign_lab(lab, cent)

    step("色带引擎：自适应模糊 + 空间投票…")
    low = edge <= float(edge_thr)          # 低对比：色带可平滑
    if highlight_l > 0:
        # 镜面高光（眼高光/皮肤反光）不参与模糊投票，否则会被抹成一块死白
        low &= ~(L > float(highlight_l))
    if blur_sigma > 0:
        smooth_rgb = _adaptive_blur(arr, low, float(blur_sigma), near_sigma=float(near_sigma))
        smooth_lab = srgb_to_oklab(smooth_rgb.astype(np.float32) / 255.0)
        smooth_labels = _assign_lab(smooth_lab, cent)
        voted = _soft_vote(np.where(low, smooth_labels, raw_labels),
                           float(vote_sigma), low)
        labels = np.where(low, voted, raw_labels)
    else:
        labels = raw_labels

    step("色带引擎：清理碎块（高对比小点保留）…")
    if speck_area:
        labels = _merge_specks(labels, arr, max_area=int(speck_area),
                               keep_de=float(speck_keep_de))

    pal = _region_palette(labels, arr, mode=palette_mode)
    line_id = -1
    stroke_d = ''
    if stroke_layer:
        # v7.11：细线（睫毛/发丝/轮廓）改用中心线等宽描边，不再靠色块边界承担。
        # 描边覆盖处的色块先向内补平（最近邻标签），否则"面的边 + 线的边"
        # 会叠成双边缘，反而更脏。
        import stroke_tracer as _skt
        sr = _skt.strokes_from_labels(labels, arr,
                                      max_width=float(stroke_max_width),
                                      min_len=float(stroke_min_len),
                                      progress=step)
        if sr['count']:
            mk = sr['mask'] > 0
            if stroke_inpaint and mk.any() and float(mk.mean()) < 0.35:
                idx = ndi.distance_transform_edt(
                    mk, return_distances=False, return_indices=True)
                labels = labels[tuple(idx)]
                pal = _region_palette(labels, arr, mode=palette_mode)
            stroke_d = sr['d']
    if line_layer:
        # 睫毛/发丝只有 1–2px 宽，色带量化会把它们切成断续短段；形态学黑帽 +
        # 滞后连接（项目自有 line_layer）先把整条线连起来，作为独立顶层。
        import line_layer as _ll
        m = _ll.stroke_mask(arr, keep=float(line_keep), seed=float(line_seed), min_area=12)
        if m.any() and float(m.mean()) <= 0.22:
            # 线条 mask 边缘仍是像素阶梯：先闭合断口，再用小半径高斯把轮廓化开，
            # 之后轮廓拟合才不会一圈毛刺。
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            m = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8) > 0.45).astype(np.uint8)
            line_id = int(labels.max()) + 1
            labels[m > 0] = line_id
            pal = _region_palette(labels, arr, mode=palette_mode)
    S = max(1, int(subpixel))
    labs = np.repeat(np.repeat(labels, S, axis=0), S, axis=1) if S > 1 else labels
    counts = np.bincount(labels.ravel(), minlength=len(pal))
    order = [int(i) for i in np.argsort(-counts) if counts[i] > 0]

    step("色带引擎：L/A/Q 长弧拟合…")
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
            is_line = (int(k) == int(line_id))
            pe = 0.8 if is_line else float(poly_eps)
            cd = 55.0 if is_line else float(corner_deg)
            cs = 2 if is_line else int(contour_smooth)
            pts = cv2.approxPolyDP(c, pe, True).reshape(-1, 2).astype(float) / S
            if len(pts) < 3:
                continue
            if cs > 0 and len(pts) >= 8 and (is_line or area >= 200 * S * S):
                # 闭合 Chaikin 圆角：像素阶梯的 90° 拐角先化开，再交给转角检测，
                # 否则每个阶梯折点都会被当成真转角 -> 碎成短 L/Q 段。
                pts = rv._chaikin(pts, cs)
                pts = pts[::max(1, len(pts) // 200)]
            p = rv._arc_path(pts, float(arc_tol), corner_deg=cd)
            if p:
                d.append(p)
                nsub += 1
        if not d:
            continue
        rgbv = '#%02x%02x%02x' % tuple(int(v) for v in pal[k])
        fills.append('<path d="%s" fill="%s" fill-rule="evenodd"/>' % (''.join(d), rgbv))

    bg = order[0] if order else 0
    body = '<rect x="0" y="0" width="%d" height="%d" fill="#%02x%02x%02x"/>\n' % (
        w, h, *tuple(int(v) for v in pal[bg]))
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
           'width="%d" height="%d" viewBox="0 0 %d %d">\n%s%s%s\n</svg>' % (
               w, h, w, h, body, '\n'.join(fills),
               ('\n' + stroke_d) if stroke_d else ''))

    step("色带引擎完成 %d 层" % len(fills))
    return {'svg': svg, 'layers': len(fills), 'subpaths': nsub,
            'colors': len(fills), 'width': w, 'height': h,
            'strokes': len(stroke_d.split('<path')) - 1 if stroke_d else 0,
            'ms': int((time.time() - t0) * 1000)}
