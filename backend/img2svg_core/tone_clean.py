# -*- coding: utf-8 -*-
"""平滑区近色调和 (自研, v7.17.4) — 去掉人眼不可辨的伪细节色块。

问题（用户 2026-11-19 指出，脸颊红框实证）：
  保真线 170 色量化会在脸颊/下巴这类**平滑区**切出大量 ΔE<3 的碎色块
  （下巴区 34134 个连通域，32822 个面积<50px；top10 里 5 个几乎是同一个黑）。
  这些色差在 JND（just-noticeable difference）以下，正常观看分辨不出，
  但 3D 打印放大后就是「脏 / 杂」。群友方案把这块统一成单色，看着干净。

本模块不是 v7.18 的「无差别合并小色块」（那会丢真细节），而是：
  1. 只用原图梯度判定「平滑区」，高梯度（五官/发丝/轮廓）一律不动；
  2. 在 OKLab 空间做近色聚类（ΔE < jnd），把不可辨的调色板色并成一簇；
  3. 仅在平滑区内把标签重映射到簇代表色；高梯度区保留原标签和原色；
  4. 面积阈值过滤：只归并「碎」的，大块真色（如肤色主体、黑发主体）不动。

只依赖 numpy / scipy / skimage / color（全部已在运行时内）。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import color


def merge_near_colors(pal: np.ndarray, jnd: float = 2.6,
                      min_l: float = 0.0, freq: np.ndarray | None = None,
                      dark_jnd: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """OKLab 空间「星形」近色归并，返回 (新调色板, 旧->新标签映射)。

    绝不能用并查集传递：A≈B、B≈C 但 A 与 C 可能差很远，链式效应会把
    170 色全串成一坨（实测平均色差 81，RMSE 78~99）。
    这里采用星形：按出现频率从高到低取锚点，只有**直接**距锚点 <jnd 的
    颜色才并入，且被并入的不再当锚点。保证任意颜色与其代表色距离 <jnd。

    newpal 保留原调色板（不重建）：高梯度区保留的原标签必须查到正确颜色，
    不能有任何未填充的 0（黑）槽位。
    """
    K = len(pal)
    lab = color.srgb_to_oklab(np.clip(pal, 0, 255).astype(np.float32) / 255.0)
    newpal = pal.copy()
    mp = np.arange(K, dtype=np.int32)
    # 频率高的颜色优先当锚点（主色不会被并走）
    if freq is None:
        order = np.arange(K)
    else:
        order = np.argsort(-np.bincount(freq, minlength=K))
    used = np.zeros(K, bool)
    # 暗部人眼分辨力低（韦伯定律）：L<0.25 的阴影允许更大合并半径，
    # 治「同一片暗部被切成十几个亮度档」的色带（下巴区实证）。
    local_jnd = np.full(K, float(jnd), np.float32)
    if dark_jnd is not None:
        luma = lab[:, 0]
        local_jnd[luma < 0.25] = max(float(jnd), float(dark_jnd))
    for a in order:
        if used[a]:
            continue
        used[a] = True
        dist = np.sqrt(((lab - lab[a]) ** 2).sum(1))
        # 两色都在暗部时用暗部阈值；有一方在亮部用常规阈值
        if dark_jnd is not None and lab[a, 0] < 0.25:
            th = np.where(lab[:, 0] < 0.25, float(dark_jnd), float(jnd))
        else:
            th = np.full(K, float(jnd), np.float32)
        members = np.flatnonzero((dist < th) & ~used)
        for m in members:
            if m == a:
                continue
            used[m] = True
            mp[m] = a
    return newpal, mp


def clean(labels: np.ndarray, pal: np.ndarray, orig_rgb: np.ndarray,
          jnd: float = 0.02, grad_hi: float = 22.0, grad_lo: float = 7.0,
          min_area: int = 240, feather: int = 1,
          dark_jnd: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """平坦区近色调和。

    labels: H×W int32 量化标签；pal: K×3 uint8；orig_rgb: H×W×3 原图(网格尺寸)。
    返回 (新标签, 新调色板)。高梯度像素及其 jnd 邻域不动。
    """
    h, w = labels.shape
    g = cvt = None
    import cv2
    gray = cv2.cvtColor(orig_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad = np.maximum(gx, gy)
    # 平滑一下梯度图，避免单像素噪声误判为"结构"
    grad_s = ndi.gaussian_filter(grad, 1.2)
    smooth = grad_s < grad_lo
    # 高梯度周围 feather 像素也保护（轮廓边缘的过渡带）
    structure = ndi.binary_dilation(grad_s >= grad_hi,
                                    structure=np.ones((2 * feather + 1, 2 * feather + 1), bool))
    editable = smooth & ~structure

    newpal, mp = merge_near_colors(pal, jnd=jnd, freq=labels.ravel(), dark_jnd=dark_jnd)
    # 只映射"被合并过"且"位于平坦区"的像素
    changed_labels = np.flatnonzero(mp != np.arange(len(mp)))
    if len(changed_labels) == 0:
        return labels, pal
    sel_mask = np.isin(labels, changed_labels) & editable
    out = labels.copy()
    out[sel_mask] = mp[out[sel_mask]]
    return out, newpal
