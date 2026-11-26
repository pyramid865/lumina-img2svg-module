# -*- coding: utf-8 -*-
"""中心线描边层（自研，v7.11）——专治"细线糊、断、淡"。

## 为什么需要它
色带引擎把画面切成色块，而细线（睫毛、发丝、轮廓线、眼线）只有 1–2px 宽：
- 作为"面"输出时，轮廓要沿像素阶梯走，必然产生毛刺与断口；
- `mean` 取色把深色核心和抗锯齿浅边一起平均 → 描边发灰发淡，整张图显"糊"；
- 小碎块清理还会把断开的线段并进邻近肤色。

细线本来就该用"线"表示：骨架化取中心线 → 平滑样条 → 按实测宽度等宽描边。
这样线条是**几何光滑**的，颜色是**核心色**，不再被抗锯齿洗淡。

## 流程
1. 取色带引擎已分好的标签，逐标签找连通块；
2. 用块内距离变换判定"细不细"：最大厚度 ≤ max_width 才算线；
3. 骨架化 → 8 连通遍历 → 拆折线；
4. `splprep` 平滑样条重采样，按导数生成三次贝塞尔（真正光滑，不是阶梯拟合）；
5. 颜色取块内核心像素（dt 大的部分）均值，保证是深核不是混色；
6. 按「颜色 + 宽度」分组输出 `<path stroke=... stroke-width=...>`，圆头圆角。

只依赖项目已有的 numpy / scipy / skimage / OpenCV，无模型。
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from skimage.morphology import skeletonize

from color import srgb_to_oklab

_NB = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _skeleton_paths(sk: np.ndarray):
    """8-connected skeleton -> list of [(y, x), ...] polylines."""
    pts = np.argwhere(sk)
    if not len(pts):
        return []
    S = set(map(tuple, pts.tolist()))
    deg = {}
    for p in S:
        y, x = p
        d = 0
        for dy, dx in _NB:
            if (y + dy, x + dx) in S:
                d += 1
        deg[p] = d

    def nbrs(p):
        y, x = p
        for dy, dx in _NB:
            q = (y + dy, x + dx)
            if q in S:
                yield q

    seen = set()
    paths = []

    def walk(a, b):
        path = [a, b]
        seen.add((a, b)); seen.add((b, a))
        prev, cur = a, b
        while deg[cur] == 2:
            nxt = None
            for q in nbrs(cur):
                if q != prev:
                    nxt = q
                    break
            if nxt is None or (cur, nxt) in seen:
                break
            seen.add((cur, nxt)); seen.add((nxt, cur))
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for n in [p for p in S if deg[p] != 2]:
        for q in nbrs(n):
            if (n, q) not in seen:
                paths.append(walk(n, q))
    for p in S:
        for q in nbrs(p):
            if (p, q) not in seen:
                paths.append(walk(p, q))
    return paths


def _arc_len(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def _bezier_path(xy: np.ndarray, smooth: float = 0.7, step: float = 7.0,
                 simplify: float = 0.6) -> str | None:
    """Smoothing spline through the centreline -> cubic beziers."""
    if len(xy) < 3:
        if len(xy) == 2:
            return 'M%.2f %.2fL%.2f %.2f' % (xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1])
        return None
    keep = np.ones(len(xy), bool)
    keep[1:] = (np.abs(np.diff(xy, axis=0)).sum(1) > 1e-6)
    xy = xy[keep]
    if len(xy) < 3:
        return None
    n = len(xy)
    try:
        from scipy.interpolate import splev, splprep
        k = 3 if n > 3 else n - 1
        tck, _ = splprep([xy[:, 0], xy[:, 1]], s=float(n) * float(smooth) ** 2, k=k)
    except Exception:
        return 'M' + 'L'.join('%.2f %.2f' % (p[0], p[1]) for p in xy)
    length = _arc_len(xy)
    m = int(np.clip(round(length / max(1.0, step)), 2, 48))
    u = np.linspace(0.0, 1.0, m + 1)
    px, py = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    px = np.asarray(px); py = np.asarray(py)
    dx = np.asarray(dx); dy = np.asarray(dy)
    out = ['M%.2f %.2f' % (px[0], py[0])]
    for i in range(m):
        du = u[i + 1] - u[i]
        out.append('C%.2f %.2f %.2f %.2f %.2f %.2f'
                   % (px[i] + dx[i] * du / 3.0, py[i] + dy[i] * du / 3.0,
                      px[i + 1] - dx[i + 1] * du / 3.0, py[i + 1] - dy[i + 1] * du / 3.0,
                      px[i + 1], py[i + 1]))
    return ''.join(out)


def strokes_from_ridges(rgb: np.ndarray, sigmas=(0.7, 1.1, 1.7),
                        thr_high: float = 0.26, thr_low: float = 0.10,
                        max_width: float = 4.6, min_width: float = 0.55,
                        min_len: float = 14.0, smooth: float = 0.7, step: float = 7.0,
                        progress=None) -> dict:
    """Thin dark/light ridges -> centreline strokes (hysteresis keeps lines whole).

    Sato 的多尺度脊线响应沿一条线是起伏的，单阈值会把线切成短段（第一版就
    断成虚线）。这里用滞后阈值：高阈值取种子，低阈值内做形态学重建生长，
    整条线先连通再骨架化，出来才是连续的一笔。
    """
    from skimage.filters import sato
    from skimage.morphology import reconstruction

    t0 = time.time()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    resp = np.maximum(sato(gray, sigmas=list(sigmas), black_ridges=True),
                      sato(gray, sigmas=list(sigmas), black_ridges=False))
    if resp.max() <= 0:
        return {'d': '', 'count': 0, 'groups': 0, 'mask': np.zeros((h, w), np.uint8), 'ms': 0}
    norm = resp / max(1e-6, float(np.percentile(resp, 99.5)))
    seed = (norm > float(thr_high)).astype(np.uint8)
    lowm = (norm > float(thr_low)).astype(np.uint8)
    if not seed.any() or not lowm.any():
        return {'d': '', 'count': 0, 'groups': 0, 'mask': np.zeros((h, w), np.uint8), 'ms': 0}
    grown = reconstruction(seed, lowm, method='dilation').astype(np.uint8)
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if progress:
        progress("描边层：脊线连通 %.1f%% 像素" % (100.0 * grown.mean()))

    dt = cv2.distanceTransform(grown, cv2.DIST_L2, 3)
    sk = skeletonize(grown > 0)
    keep = sk & (dt * 2.0 <= float(max_width))
    if not keep.any():
        return {'d': '', 'count': 0, 'groups': 0, 'mask': np.zeros((h, w), np.uint8), 'ms': 0}

    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    groups: dict = {}
    kept = 0
    for p in _skeleton_paths(keep):
        arr = np.asarray(p, np.float32)
        xy = np.stack([arr[:, 1], arr[:, 0]], 1)
        length = _arc_len(xy)
        if length < float(min_len):
            continue
        ys = arr[:, 0].astype(np.int32).clip(0, h - 1)
        xs = arr[:, 1].astype(np.int32).clip(0, w - 1)
        width = float(np.median(dt[ys, xs]) * 2.0)
        width = max(float(min_width), min(float(max_width), width))
        # 沿线取样 + 环带取样：线比周围暗就是暗线，反之是亮线
        ring = np.zeros((h, w), np.uint8)
        ring[ys, xs] = 1
        ring = cv2.dilate(ring, np.ones((9, 9), np.uint8)).astype(bool)
        ring[ys, xs] = False
        ry, rx = np.nonzero(ring)
        if len(ry) > 4000:
            sel = np.random.default_rng(7).choice(len(ry), 4000, replace=False)
            ry, rx = ry[sel], rx[sel]
        bg = float(np.median(L[ry, rx])) if len(ry) else float(np.median(L))
        fg = float(np.median(L[ys, xs]))
        vals = L[ys, xs]
        if fg < bg:                       # 暗线：取最暗的一截
            pick = vals <= np.percentile(vals, 35)
        else:                             # 亮线：取最亮的一截
            pick = vals >= np.percentile(vals, 65)
        col = tuple(int(v) for v in np.median(rgb[ys[pick], xs[pick]], 0))
        key = (col, round(width * 4) / 4.0)
        d = _bezier_path(xy, smooth=smooth, step=step)
        if not d:
            continue
        groups.setdefault(key, []).append(d)
        kept += 1

    frag = []
    for (col, width), ds in groups.items():
        frag.append('<path d="%s" fill="none" stroke="#%02x%02x%02x" stroke-width="%.2f" '
                    'stroke-linecap="round" stroke-linejoin="round"/>'
                    % (''.join(ds), col[0], col[1], col[2], width))
    if progress:
        progress("描边层：%d 条中心线 / %d 组" % (kept, len(groups)))
    return {'d': '\n'.join(frag), 'count': kept, 'groups': len(groups),
            'mask': (keep.astype(np.uint8)), 'ms': int((time.time() - t0) * 1000)}


def strokes_from_labels(labels: np.ndarray, rgb: np.ndarray,
                        max_width: float = 4.5, min_area: int = 45,
                        min_len: float = 10.0, min_contrast: float = 0.055,
                        smooth: float = 0.7, step: float = 7.0,
                        progress=None) -> dict:
    """Thin label components -> centreline strokes.

    Returns {'d': svg fragment, 'count': n, 'groups': m, 'mask': consumed px}.
    """
    t0 = time.time()
    h, w = labels.shape[:2]
    n_lab = int(labels.max()) + 1
    counts = np.bincount(labels.ravel(), minlength=n_lab)
    lab = srgb_to_oklab(rgb.astype(np.float32) / 255.0)
    idx = labels.ravel()
    lab_mean = np.stack([np.bincount(idx, lab[..., c].ravel(), minlength=n_lab)
                         for c in range(3)], 1) / np.maximum(counts, 1)[:, None]
    consumed = np.zeros((h, w), np.uint8)
    groups: dict = {}
    kept = 0

    for k in range(n_lab):
        area_k = int(counts[k])
        if area_k < int(min_area) or area_k > 0.20 * h * w:
            continue
        m = (labels == k).astype(np.uint8)
        if not m.any():
            continue
        ncomp, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, ncomp):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if area < int(min_area):
                continue
            sub = (cc[y:y + bh, x:x + bw] == i)
            dt = cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 3)
            wmax = float(dt.max()) * 2.0
            if wmax > float(max_width):
                continue                      # 太厚：是色块，不是线
            wmed = float(np.median(dt[sub])) * 2.0
            if wmed < 0.5:
                continue
            # 对比度：与相邻标签色差太小就不画（否则等于重复画一遍填充）
            ring = cv2.dilate(sub.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~sub
            nb = labels[y:y + bh, x:x + bw][ring]
            if len(nb):
                nb = nb[nb != k]
            if len(nb):
                vals, cnts = np.unique(nb, return_counts=True)
                nb_c = vals[int(np.argmax(cnts))]
                if float(np.sqrt(((lab_mean[k] - lab_mean[nb_c]) ** 2).sum())) < float(min_contrast):
                    continue
            sk = skeletonize(sub)
            if not sk.any():
                continue
            core = dt >= max(0.5, 0.6 * float(dt.max()))
            if not core.any():
                core = sub
            col = tuple(int(v) for v in rgb[y:y + bh, x:x + bw][core].mean(0))
            for p in _skeleton_paths(sk):
                arr = np.asarray(p, np.float32)
                xy = np.stack([arr[:, 1] + x, arr[:, 0] + y], 1)
                if _arc_len(xy) < float(min_len):
                    continue
                d = _bezier_path(xy, smooth=smooth, step=step)
                if not d:
                    continue
                key = (col, round(max(0.6, min(wmed, max_width)) * 4) / 4.0)
                groups.setdefault(key, []).append(d)
                kept += 1
            consumed[y:y + bh, x:x + bw][sub] = 1

    frag = []
    for (col, width), ds in groups.items():
        frag.append('<path d="%s" fill="none" stroke="#%02x%02x%02x" stroke-width="%.2f" '
                    'stroke-linecap="round" stroke-linejoin="round"/>'
                    % (''.join(ds), col[0], col[1], col[2], width))
    if progress:
        progress("描边层：%d 条中心线 / %d 组" % (kept, len(groups)))
    return {'d': '\n'.join(frag), 'count': kept, 'groups': len(groups),
            'mask': consumed, 'ms': int((time.time() - t0) * 1000)}
