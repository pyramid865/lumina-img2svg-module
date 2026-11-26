"""智能碎片归并 (自研) — 借鉴「近色加权归并 + 五官钉色保护」的算法思路，
用本项目自研资产 (numpy/scipy/skimage + color.OKLab) 独立实现，不依赖任何网页版源码。

解决的问题：fusion 量化后，渐变区产生大量小碎片，直接用 vtracer 描摹会
「发碎」。本模块在描摹前把碎片并给颜色最接近的大邻居，同时归并前先把
瞳孔/嘴角(小暗核)、眼高光/齿白(极小亮核)钉住，归并时只许同色焊接，
从而做到「色块整、边缘顺、五官还在」。

对外接口：
    pin_features(labels, rgb, pal) -> (labels, pal, guard_set)
    merge_fragments(labels, pal, min_area, guard) -> labels
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

import color


def _luma(rgb: np.ndarray) -> np.ndarray:
    """感知亮度 0-255（RGB 数组 float/uint8 均可）。"""
    a = rgb.astype(np.float32)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def _keep_small(mask: np.ndarray, max_area: int, min_area: int = 2) -> np.ndarray:
    """保留面积在 [min_area, max_area] 的连通域，其余清零。"""
    reg = measure.label(mask, connectivity=1)
    n = int(reg.max())
    if n == 0:
        return np.zeros_like(mask, bool)
    area = np.bincount(reg.ravel(), minlength=n + 1)
    keep = (area >= min_area) & (area <= max_area)
    keep[0] = False
    return keep[reg]


def pin_features(labels: np.ndarray, rgb: np.ndarray, pal: np.ndarray,
                 pin_dark_max: int = 500, pin_light_max: int = 64,
                 ws2: int = 1):
    """归并前钉住五官小特征：小暗核→真黑、极小亮核→近纯白。

    rgb/pal 为 0-255。ws2 = 高清倍率²，面积上限按比例放大。
    返回 (labels2, pal2, guard)：guard 是钉住标签下标集合，归并时只许同色合并。
    """
    labels = labels.copy()
    pal = pal.copy()
    guard = []
    gray = _luma(rgb)
    # 平滑后亮度：抹掉噪点，让"2px 瞳孔放大后的中暗光晕"仍能被识别为暗核
    pl = ndi.gaussian_filter(gray, 1.0)
    # ---- 钉黑：够暗(<=110) 且是孤立小连通域的像素（瞳孔/嘴角/鼻孔） ----
    dark = (pl <= 110.0)
    dark_small = _keep_small(dark, int(pin_dark_max * ws2))
    # 加一条真黑（不复用调色板里可能的深棕/深蓝）
    if int(dark_small.sum()) >= 32:
        pal = np.vstack([pal, np.array([[9, 9, 9]], dtype=pal.dtype)])
        bi = len(pal) - 1
        labels[dark_small] = bi
        guard.append(bi)
        # 向内焊 1px：吃掉 LANCZOS 光晕造成的中暗碎标签，但只长进中暗像素
        core = labels == bi
        grown = ndi.binary_dilation(core, structure=np.ones((3, 3), bool))
        weld = grown & (pl <= 160.0) & ~core
        labels[weld] = bi
    # ---- 钉白：原图极亮(>=235) + 近旁有暗(局部最小值<=140) + 极小孤立亮核 ----
    lo5 = ndi.minimum_filter(gray, 5)
    light = (gray >= 235.0) & (lo5 <= 140.0)
    light_small = _keep_small(light, int(pin_light_max * ws2)) & ~_keep_small(light, 2)
    if int(light_small.sum()) >= 12:
        pal = np.vstack([pal, np.array([[253, 253, 253]], dtype=pal.dtype)])
        wi = len(pal) - 1
        labels[light_small] = wi
        guard.append(wi)
        # 白核向外焊 1px，把 1px 高光撑到描摹能留下的块大小
        core = labels == wi
        grown = ndi.binary_dilation(core, structure=np.ones((3, 3), bool))
        labels[grown & ~core] = wi
    return labels, pal, set(guard)


def merge_fragments(labels: np.ndarray, pal: np.ndarray, min_area: int,
                    guard: set | None = None, iters: int = 2) -> np.ndarray:
    """近色加权碎片归并：小连通域并给「共享边长/(16+OKLab色距)」最大的邻居。

    与 geom_trace.merge_small 的区别：投票显式计入共享边长；钉住的黑/白
    只许并同色（瞳孔/高光碎片互相焊成整块，绝不并进底色）。
    """
    guard = set(guard or ())
    out = labels.copy()
    labpal = color.srgb_to_oklab(np.clip(pal, 0, 255).astype(np.float32) / 255.0)
    for _ in range(iters):
        reg = measure.label(out.astype(np.int32) + 1, connectivity=1)
        n = int(reg.max())
        if n <= 1:
            break
        area = np.bincount(reg.ravel(), minlength=n + 1)
        # 全分辨率统计共享边长（相邻不同连通域的像素对数）
        votes: dict[int, dict[int, float]] = {}
        for axis in (0, 1):
            a = reg[:-1, :] if axis == 0 else reg[:, :-1]
            b = reg[1:, :] if axis == 0 else reg[:, 1:]
            m = (a != b) & (a > 0) & (b > 0)
            if not m.any():
                continue
            k = a[m].astype(np.int64) * (n + 1) + b[m]
            uk, cnt = np.unique(k, return_counts=True)
            for key, c in zip(uk.tolist(), cnt.tolist()):
                x, y = divmod(key, n + 1)
                votes.setdefault(x, {})[y] = votes.setdefault(x, {}).get(y, 0) + c
                votes.setdefault(y, {})[x] = votes.setdefault(y, {}).get(x, 0) + c
        # 每个连通域的主色：向量化实现（原逐域 bincount 在 2x 网格上万域时极慢）
        cval = np.zeros(n + 1, np.int32)
        flat_idx = reg.ravel().astype(np.int64)
        flat_val = out.ravel().astype(np.int64)
        # 用 (区域id, 颜色id) 联合计数，取每区域出现最多的颜色
        joint = flat_idx * (len(pal) + 1) + flat_val
        jc = np.bincount(joint)
        kJ, jC = np.divmod(np.flatnonzero(jc), len(pal) + 1)
        counts = jc[jc > 0]
        # 按区域取最大计数的颜色
        order = np.lexsort((-counts, kJ))
        kJs, jCs = kJ[order], jC[order]
        first = np.ones(len(kJs), bool); first[1:] = kJs[1:] != kJs[:-1]
        cval[kJs[first]] = jCs[first]
        small = np.flatnonzero(area < min_area)
        small = small[small != 0]
        target = {}
        for r in small.tolist():
            v = votes.get(r)
            if not v:
                continue
            c1 = labpal[cval[r]]
            best, best_s = None, -1.0
            for nb, cnt in v.items():
                d = float(np.sqrt(((labpal[cval[nb]] - c1) ** 2).sum()))
                sc = cnt / (16.0 + d)          # 近色加权：共享边长÷(16+色距)
                if sc > best_s:
                    best_s, best = sc, nb
            if best is None or best == r:
                continue
            # 钉住的黑/白只许并同色
            if cval[r] in guard and cval[r] != cval[best]:
                continue
            target[r] = best
        # 解链：A→B、B→C 则 A→C
        for _ in range(6):
            moved = False
            for r in list(target):
                if target[r] in target:
                    target[r] = target[target[r]]
                    moved = True
                if target[r] == r:
                    del target[r]
            if not moved:
                break
        if not target:
            break
        lut = np.arange(n + 1, dtype=reg.dtype)
        for r, t in target.items():
            lut[r] = t
        out = cval[lut[reg]].astype(out.dtype)
    return out
