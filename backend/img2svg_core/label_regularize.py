"""描摹前的量化标签图规整（源头消除像素阶梯）。

背景（HANDOFF §22）：vtracer 忠实追随量化标签图的逐像素硬边界，短 C 因此
把「像素阶梯」原样描进 SVG，4× 放大就是锯齿。在**成品 SVG 上**做任何曲线
平滑 / 几何重拟合都会失败——stacked 是层叠覆盖，边界一动就露出下层颜色。

这里换个位置动手：在 vtracer 之前规整 ``labels``。所有 path 都从同一张
规整后的标签图生成，边界天然一致，不会出现「各自偏移露底色」。

算法：边缘保护的空间多数投票。
  1) 用**未量化的原图**算梯度：真实轮廓/线条梯度高 → 保护；量化造成的
     阶梯边界在原图里往往是柔和渐变（梯度低）→ 允许规整。
  2) 4 邻域里有 3 个同标签、中心却不同 → 判为孤立像素 / 阶梯凸起，改成
     邻居标签。迭代若干轮逐步把阶梯抹平。
  3) 只改满足条件的少数像素，色板与颜色分层完全不变， RMSE 风险可控。

纯 numpy，无新增依赖。IMG2SVG_LABEL_REG=0/1/2/3 控制强度，默认 0（关闭）。
"""
from __future__ import annotations

import numpy as np


def _gradient(img: np.ndarray) -> np.ndarray:
    """灰度梯度幅值（用原图，不用量化图）。"""
    a = np.asarray(img, np.float32)
    if a.ndim == 3:
        g = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    else:
        g = a
    gy, gx = np.gradient(g)
    return np.sqrt(gx * gx + gy * gy)


def edge_protect(img: np.ndarray, percentile: float = 70.0) -> np.ndarray:
    """返回「真实边缘」布尔掩码：梯度高于全图分位数的像素要保护，不被规整改写。

    用**未量化的原图**算梯度是关键：量化造成的像素阶梯在原图里通常只是柔和
    渐变（梯度低，允许抹平），而轮廓线 / 发丝 / 花纹是真实高频（梯度高，保护）。
    """
    g = _gradient(img)
    sample = g[::8, ::8].ravel()
    thr = max(1.0, float(np.percentile(sample, percentile))) if sample.size else 1.0
    return g > thr


# 旧名保留，便于已有实验脚本调用
_protect_mask = edge_protect


def regularize(labels: np.ndarray, img: np.ndarray, strength: int = 1,
               protect_pct: float = 75.0) -> tuple[np.ndarray, dict]:
    """返回 (规整后的 labels, 统计信息)。strength=0 时原样返回。"""
    lab = np.asarray(labels)
    info = {"strength": int(strength), "changed": 0, "protect_pct": protect_pct}
    if strength <= 0:
        return lab, info
    protect = _protect_mask(img, protect_pct)
    total = 0
    for _ in range(int(strength)):
        P = np.pad(lab, 1, mode="edge")
        up, dn = P[:-2, 1:-1], P[2:, 1:-1]
        lf, rt = P[1:-1, :-2], P[1:-1, 2:]
        v = lab.copy()
        m1 = (up == dn) & (dn == lf)
        v[m1] = up[m1]
        m2 = (up == dn) & (dn == rt)
        v[m2] = up[m2]
        m3 = (up == lf) & (lf == rt)
        v[m3] = up[m3]
        m4 = (dn == lf) & (lf == rt)
        v[m4] = dn[m4]
        change = (m1 | m2 | m3 | m4) & ~protect & (v != lab)
        n = int(change.sum())
        if n == 0:
            break
        lab[change] = v[change]
        total += n
    info["changed"] = total
    info["pct"] = round(100.0 * total / lab.size, 3)
    return lab, info


def from_env() -> int:
    import os
    try:
        v = int(os.environ.get("IMG2SVG_LABEL_REG", "0"))
    except ValueError:
        v = 0
    return max(0, min(6, v))
