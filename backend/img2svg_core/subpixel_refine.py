"""几何层亚像素边缘细化（自研，借鉴 Vecto stage 9b 的公开思路）。

对共边等值线上的每个点，沿边界法线在【原图】上双线性采样，找到该处
从区域色 C_in 过渡到邻居色 C_out 的 50% 覆盖位置，把点沿法线滑到那里。
这样把 vtracer/量化丢失的抗锯齿边缘位置信息重新找回来，治「边缘糊、
跳变只有原图一半」。

关键（区别于失败的标签层简化版）：
- 作用在几何轮廓点上（共边链），不是改像素标签归属；
- 两侧颜色 C_in/C_out 由轮廓点所在的两个区域给出（或就地采样推断）；
- 沿法线滑点，限幅，角点/交叉处降低位移避免撕裂。
"""
import numpy as np
from scipy import ndimage as ndi


def _bilinear(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """双线性采样，img 为 float32 HxWx3，坐标以像素中心为整数。"""
    h, w = img.shape[:2]
    x = min(max(x, 0.0), float(w - 1)); y = min(max(y, 0.0), float(h - 1))
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1); y1 = min(y0 + 1, h - 1)
    x0 = max(x0, 0); y0 = max(y0, 0)
    fx = min(max(x - x0, 0.0), 1.0); fy = min(max(y - y0, 0.0), 1.0)
    a = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    b = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return a * (1 - fy) + b * fy


def refine_chain(xy: np.ndarray, orig: np.ndarray, labels: np.ndarray,
                 pal: np.ndarray, c_in: int, max_shift: float = 0.75,
                 n_steps: int = 4, step: float = 0.35,
                 corner_guard: bool = True) -> np.ndarray:
    """细化一条闭合共边链。

    xy: (N,2) 浮点 [x,y]，共边等值线（像素分界坐标）
    orig: 原图 float32 HxWx3（工作分辨率，与 labels 同尺寸）
    labels/pal: 量化标签与调色板（工作分辨率）
    c_in: 这条链包围的区域标签
    返回细化后的 (N,2)。
    """
    P = np.asarray(xy, np.float64).copy()
    n = len(P)
    if n < 8:
        return P
    closed = np.allclose(P[0], P[-1], atol=1e-6)
    core = P[:-1] if closed else P
    m = len(core)

    # 两侧颜色：区域内均值色 C_in，邻域主色 C_out（沿法线外侧采样标签众数）
    def region_color(x, y, default):
        xi = int(round(x)); yi = int(round(y))
        if 0 <= yi < labels.shape[0] and 0 <= xi < labels.shape[1]:
            return pal[labels[yi, xi]].astype(np.float64)
        return default

    cin = pal[c_in].astype(np.float64) if c_in < len(pal) else region_color(*core[0], np.zeros(3))
    # 切线 / 法线
    prev = np.roll(core, 1, axis=0); nxt = np.roll(core, -1, axis=0)
    tang = nxt - prev
    tl = np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-9)
    tang = tang / tl[:, None]
    nrm = np.c_[tang[:, 1], -tang[:, 0]]  # 一种法线方向，符号由覆盖测试定

    # 转角（角点降低位移，避免撕裂）
    if corner_guard:
        a1 = np.arctan2(tang[:, 1], tang[:, 0])
        turn = np.abs(np.arctan2(np.sin(np.roll(a1, -1) - a1), np.cos(np.roll(a1, -1) - a1)))
        guard = np.clip(1.0 - turn / 1.6, 0.15, 1.0)
    else:
        guard = np.ones(m)

    out = core.copy()
    for i in range(m):
        nx, ny = nrm[i]
        # 试两个方向，找颜色差异更大的一侧作为"向外"
        p_plus = core[i] + np.array([nx, ny]) * step * 2
        p_minus = core[i] - np.array([nx, ny]) * step * 2
        cp = _bilinear(orig, p_plus[0], p_plus[1])
        cm = _bilinear(orig, p_minus[0], p_minus[1])
        # 哪一侧更远离 cin，就是外侧
        dp = np.linalg.norm(cp - cin); dm = np.linalg.norm(cm - cin)
        sign = 1.0 if dp >= dm else -1.0
        cout = cp if dp >= dm else cm
        cvec = cout - cin
        cvlen2 = float(np.dot(cvec, cvec))
        if cvlen2 < 25:  # 两侧色几乎一样，不细化
            continue
        # 沿法线找覆盖率从 0.5(Cin侧) 到 0.5(Cout侧) 的过渡中点
        # t>0 朝 sign 方向；用投影在 cvec 上的标量作为覆盖度
        def cover_at(t):
            q = core[i] + sign * np.array([nx, ny]) * t
            c = _bilinear(orig, q[0], q[1])
            return float(np.dot(c - cin, cvec) / cvlen2)
        # 从 -R 到 +R 采样，找 cover 跨过 0.5 的位置
        R = max_shift + 1.0
        ts = np.linspace(-R, R, 2 * n_steps + 3)
        cs = np.array([cover_at(t) for t in ts])
        tgt = 0.5
        # 取最接近 0.5 的采样点附近做线性插值
        k = int(np.argmin(np.abs(cs - tgt)))
        if k == 0 or k == len(ts) - 1:
            continue
        t0, t1 = ts[k - 1], ts[k + 1]
        c0, c1 = cs[k - 1], cs[k + 1]
        if abs(c1 - c0) < 1e-6:
            tstar = ts[k]
        else:
            tstar = t0 + (tgt - c0) * (t1 - t0) / (c1 - c0)
        tstar = float(np.clip(tstar, -max_shift, max_shift)) * guard[i]
        out[i] = core[i] + sign * np.array([nx, ny]) * tstar
    if closed:
        out = np.vstack([out, out[0]])
    return out


def refine_labels_contours(contours, orig, labels, pal, c_in, **kw):
    return [refine_chain(xy, orig, labels, pal, c_in, **kw) for xy in contours]
