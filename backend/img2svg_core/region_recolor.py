# -*- coding: utf-8 -*-
"""平滑区区域重取色 (自研, v7.17.4 修复版) — 治「脸颊色块太杂」。

背景（用户 2026-11-19 指出，下巴区实测，口径已在本机复现）：
  群友成品下巴 33 色 / 主色 57.4%，我们 v7.17.3 是 120 色 / 14.0%。
  同一片平滑阴影被 170 色量化切成十几个亮度档 = 色带伪细节。
  纯 JND 近色归并（tone_clean）改善轻微；色阶归整会偏灰绿。
  正解：**按原图平滑区的连通域整片重取主色**，不是在旧调色板里挑色。

v7.17.4 重写要点（修掉首轮 A/B 的两处错误）：
  - 旧版按「量化标签 × 连通域」分区域，每个区域再分配一个新颜色，
    颜色数不降反升（下巴 100→138），方向就是错的。本版改为按
    **原图平滑掩码的连通域**分块，整块统一取原图中位数色。
  - 旧版 `newpal[tgt] = med` 复用旧标签改全局调色板，会污染非平滑区
    同标签像素。本版新色从 K 之后追加，旧调色板一字不改。
  - 区域间主色合并用**星形聚类**（面积最大者为簇心，近色者归簇），
    不用并查集链式传递——HANDOFF §42 已证实链式会把全部颜色串成一坨。

`smooth_override` 可传入外部平滑区掩码（如 CNN 分割输出），便于 A/B
只替换「平滑区判定」这一个变量。

依赖 numpy / scipy / skimage / color（运行时已具备）。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import color


def _smooth_mask(arr: np.ndarray, grad_lo: float = 9.0,
                 protect: tuple[float, float] = (0.94, 0.06)) -> np.ndarray:
    """原图梯度判定平滑区；极亮/极暗（高光/瞳孔）永不参与。"""
    import cv2
    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad = ndi.gaussian_filter(np.maximum(gx, gy), 1.2)
    smooth = grad < grad_lo
    lum = arr.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    smooth &= (lum < 255 * protect[0]) & (lum > 255 * protect[1])
    return smooth


def recolor(labels: np.ndarray, pal: np.ndarray, orig: np.ndarray,
            jnd: float = 0.035, grad_lo: float = 9.0,
            min_area: int | None = None, cluster_jnd: float = 0.05,
            use_median: bool = True,
            smooth_override: np.ndarray | None = None,
            smooth_max_de: float | None = 2.5
            ) -> tuple[np.ndarray, np.ndarray]:
    """平滑区整片重取主色。返回 (新标签, 新调色板)。

    labels H×W int32, pal K×3 uint8, orig H×W×3 uint8。
    min_area 默认按图像面积自适应；cluster_jnd 为 OKLab 空间星形聚类的色距阈值。
    smooth_override 给定时完全替代梯度掩码（形状须与 labels 一致）。
    smooth_max_de 是渐变保护阈值：连通域内部颜色跨度（OKLab ΔE×100，p5~p95）
    超过它就不整片刷平——见下面步骤 2 的说明，None 关闭保护。
    """
    h, w = labels.shape
    if smooth_override is None:
        smooth = _smooth_mask(orig, grad_lo=grad_lo)
    else:
        smooth = np.asarray(smooth_override, bool)
        if smooth.shape != (h, w):
            raise ValueError("smooth_override 形状必须与 labels 一致")
    out = labels.copy()
    newpal = np.asarray(pal, np.uint8).copy()
    K = len(newpal)
    if not smooth.any():
        return out, newpal
    if min_area is None:
        min_area = max(8, int(labels.size * 0.00015))

    # 1) 按原图平滑掩码的连通域分块 —— 关键：不看量化标签
    reg, n = ndi.label(smooth, structure=np.ones((3, 3), np.int32))
    if n == 0:
        return out, newpal
    area = np.bincount(reg.ravel(), minlength=n + 1)
    area[0] = 0

    # 2) 每块用原图像素取一个主色（中位数抗高光/暗斑，保肤色暖调）
    #    渐变保护（v7.17.13）：区域内部颜色跨度大，说明它不是「平涂面」而是渐变。
    #    整片刷成一色会丢掉整段过渡，并和未刷区域之间留下台阶色带。
    #    实测（p5~p95 的 OKLab ΔE）真平涂 0.00、淡彩肤色 4~8、
    #    水面 26、合成渐变 41、天空 53 —— 阈值 2.5（≈1.25 JND）能干净分开。
    #    没有这道保护时，整幅渐变图会被 CNN 判成「100% 平滑」连成一个连通域，
    #    取中位色刷平 → 输出只剩 1 个色块的纯色矩形（color=1 / path=1）。
    colors = np.zeros((n + 1, 3), np.uint8)
    for r in np.flatnonzero(area >= min_area).tolist():
        pix = orig[reg == r].astype(np.float32)
        if smooth_max_de is not None:
            lab_r = color.srgb_to_oklab(pix / 255.0)
            span = np.percentile(lab_r, 95, axis=0) - np.percentile(lab_r, 5, axis=0)
            if float(np.sqrt((span ** 2).sum())) * 100.0 > smooth_max_de:
                area[r] = 0          # 交给原量化结果，不整片刷平
                continue
        colors[r] = np.clip(np.median(pix, 0) if use_median else pix.mean(0),
                            0, 255).astype(np.uint8)
    lab = color.srgb_to_oklab(colors.astype(np.float32) / 255.0)

    # 3) 星形聚类：面积大者当簇心，近色块归簇；绝不链式传递
    order = np.argsort(-area)
    center_of = np.full(n + 1, -1, np.int32)
    centers: list[int] = []
    for r in order:
        if area[r] < min_area or center_of[r] >= 0:
            continue
        ci = len(centers)
        centers.append(int(r))
        center_of[r] = ci
        for s in order:
            if area[s] < min_area or center_of[s] >= 0:
                continue
            de = float(np.sqrt(((lab[r] - lab[s]) ** 2).sum()))
            if de < cluster_jnd:
                center_of[s] = ci

    # 4) 平滑区写回簇色（新标签 ID）；非平滑区完全不动
    newcols: list[np.ndarray] = []
    for ci, r in enumerate(centers):
        newcols.append(colors[r])
    for ci in range(len(centers)):
        m = np.isin(reg, np.flatnonzero(center_of == ci))
        out[m] = K + ci
    if newcols:
        newpal = np.vstack([newpal, np.stack(newcols)])
    return out, newpal
